# -*- coding: utf-8 -*-
"""
ui/viewer.py —— fNIRS 查看器页（PySide6 + pyqtgraph，深色主题）

参照 Brainstorm（顶部事件条 + 下方波形）与 mne-qt-browser（深色主题）的思路。
数据层复用 core/data.py 的 FNIRSData，主题复用 ui/theme.py。
"""

import os

from PySide6 import QtCore, QtGui, QtWidgets

import pyqtgraph as pg
import numpy as np

from ui import theme
from ui.theme import _CN_FONT, _rgba255
from core import data as core  # 别名：保持原代码中 core.XXX 写法不变


_GRAY = (0.85, 0.85, 0.85, 1.0)

PRIMARY_MODES = (
    (core.MODE_ORIG, '原始'),
    (core.MODE_INTENSITY, '光强'),
    (core.MODE_FNIRS, '血氧'),
)
INTENSITY_VARIANTS = (
    ('原始 ADC', core.MODE_INTENSITY),
    ('相对变化 (%)', core.MODE_RELATIVE),
    ('光密度 (OD)', core.MODE_OD),
)


class _JumpSlider(QtWidgets.QSlider):
    """水平时间滑块：点击轨道任意位置直接跳转到该处。

    默认 QSlider 点击轨道只按 pageStep 翻页，需按住 handle 拖动。
    本类将轨道点击映射为 value 跳转；点击 handle 本身仍保留拖动能力。
    """

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self._handle_rect().contains(event.position().toPoint()):
                super().mousePressEvent(event)  # 点中 handle：交给默认逻辑（可拖动）
                return
            self.setValue(self._value_from_x(event.position().x()))
            event.accept()
            return
        super().mousePressEvent(event)

    def _slider_option(self):
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        return opt

    def _handle_rect(self):
        return self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider,
            self._slider_option(),
            QtWidgets.QStyle.SubControl.SC_SliderHandle, self)

    def _value_from_x(self, x):
        opt = self._slider_option()
        groove = self.style().subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt,
            QtWidgets.QStyle.SubControl.SC_SliderGroove, self)
        handle = self._handle_rect()
        span = max(1, groove.width() - handle.width())
        ratio = (x - handle.width() / 2.0 - groove.x()) / span
        ratio = max(0.0, min(1.0, ratio))
        value = self.minimum() + (self.maximum() - self.minimum()) * ratio
        return int(round(value))


class _SelectRegion(pg.LinearRegionItem):
    """自定义选区框：按下在边界线附近=调整大小，按下在中间=移动位置。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._resize_line = None  # None=移动, 0=调左线, 1=调右线

    def mouseDragEvent(self, ev):
        if not self.movable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()

        if ev.isStart():
            bdp = ev.buttonDownPos()
            vb = self.getViewBox()
            self._resize_line = None
            if vb is not None:
                mouse_x = vb.mapSceneToView(bdp).x()
                r0, r1 = sorted(self.getRegion())
                (x0, x1) = vb.viewRange()[0]
                w = vb.boundingRect().width()
                if x1 > x0 and w > 0:
                    threshold = 6.0 * (x1 - x0) / w  # 6px 换算成数据坐标
                    if abs(mouse_x - r0) <= threshold:
                        self._resize_line = 0
                    elif abs(mouse_x - r1) <= threshold:
                        self._resize_line = 1
            self.cursorOffsets = [l.pos() - bdp for l in self.lines]
            self.startPositions = [l.pos() for l in self.lines]
            self.moving = True

        if not self.moving:
            return

        self.blockLineSignal = True
        if self._resize_line is None:
            for i, l in enumerate(self.lines):
                l.setPos(self.cursorOffsets[i] + ev.pos())
        else:
            i = self._resize_line
            self.lines[i].setPos(self.cursorOffsets[i] + ev.pos())
        self.prepareGeometryChange()
        self.blockLineSignal = False

        if ev.isFinish():
            self.moving = False
            self._resize_line = None
            self.sigRegionChangeFinished.emit(self)
        else:
            self.sigRegionChanged.emit(self)


class _ViewBox(pg.ViewBox):
    """滚轮：普通=时间移动、Ctrl=时间缩放、Alt=子图高度。"""

    def __init__(self, wheel_cb):
        super().__init__()
        self.wheel_cb = wheel_cb

    def wheelEvent(self, ev, axis=None):
        ctrl = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        alt = bool(ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)
        self.wheel_cb(ev.delta(), ctrl, alt, ev.scenePos(), self)
        ev.accept()


class _TrialHitItem(pg.GraphicsObject):
    """试次条的透明点击区；悬停时仅显示轻微底色。"""

    def __init__(self, x0, x1, color, callback, tooltip=''):
        super().__init__()
        self._rect = QtCore.QRectF(float(x0), -0.15,
                                   max(0.0, float(x1) - float(x0)), 1.15)
        self._color = pg.mkColor(_rgba255(color))
        self._callback = callback
        self._hovered = False
        self.setAcceptHoverEvents(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setZValue(4)
        if tooltip:
            self.setToolTip(tooltip)

    def boundingRect(self):
        return self._rect

    def paint(self, painter, _option, _widget=None):
        if not self._hovered:
            return
        color = QtGui.QColor(self._color)
        color.setAlpha(28 if theme.current_theme == 'light' else 42)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(color))
        painter.drawRect(self._rect)

    def hoverEvent(self, event):
        hovered = not event.isExit()
        if hovered != self._hovered:
            self._hovered = hovered
            self.update()
        event.acceptClicks(QtCore.Qt.MouseButton.LeftButton)

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._callback()
            event.accept()
        else:
            event.ignore()


class ViewerWindow(QtWidgets.QMainWindow):
    def __init__(self, data: core.FNIRSData, stream_dir: str):
        super().__init__()
        self.data = data
        self.stream_dir = stream_dir
        self.selections = []  # [(start_seconds, end_seconds), ...]
        self.channel_plot_height = 110
        self.channel_plots = []
        self.channel_curves = []
        self.channel_regions = []
        self.saved_region_items = []
        self._plot_signature = None
        self._syncing_regions = False
        self._region_bounds = (0.2, 0.5)
        self._selection_by_group = {'orig': None, 'processed': None}
        self.current_epoch_record = None  # 当前点击的任务 epoch record

        self.setWindowTitle('fNIRS 查看器')
        # 标准窗口按钮：QMainWindow 默认自带最小化/最大化/关闭
        # 初始尺寸不超过可用屏幕，避免高度超出屏幕放不下
        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, screen.width()), min(750, screen.height()))

        self._setup_theme()
        self._build_ui()
        self._build_legend()

        # 绑定交互
        self._install_shortcuts()

        self._apply_theme()  # 应用当前主题（QSS + 背景 + 重绘）

    # ---------------------------------------------------------------
    # 主题 & UI
    # ---------------------------------------------------------------
    def _setup_theme(self):
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('leftButtonPan', False)  # 禁用左键平移，避免与选区冲突

    def _apply_theme(self):
        """统一刷新窗口、侧栏、图例、全部子图和选择区域。"""
        self.setStyleSheet(theme.qss())
        pg.setConfigOptions(background=theme.color('plot_bg'),
                            foreground=theme.pg_foreground())
        self.channel_glw.setBackground(pg.mkBrush(theme.color('plot_bg')))
        for plot in (self.trial_plot, self.time_plot):
            plot.setBackground(theme.color('plot_bg'))
        self._style_fixed_axes()
        self._style_channel_plots()
        if hasattr(self, 'info_label'):
            self.info_label.setStyleSheet(
                f'color: {theme.color("muted")}; font-size: 12px;')
        if hasattr(self, 'region_label'):
            self.region_label.setStyleSheet(
                f'color: {theme.color("info")}; font-size: 12px;')
        # 主题按钮文案跟随状态
        if hasattr(self, 'btn_theme'):
            self.btn_theme.setText('浅色模式' if theme.current_theme == 'dark' else '深色模式')
        if hasattr(self, 'btn_modes'):
            self._style_mode_buttons()
        if hasattr(self, 'legend_layout'):
            self._build_legend()
        self._update_plot()

    def _style_fixed_axes(self):
        fg_pen = pg.mkPen(theme.pg_foreground(), width=0.8)
        subtle = pg.mkPen(theme.color('axis_subtle'), width=0.6)
        self.trial_plot.getAxis('left').setPen(None)
        self.trial_plot.getAxis('bottom').setPen(None)
        self.time_plot.getAxis('left').setPen(None)
        self.time_plot.getAxis('bottom').setPen(fg_pen)
        self.time_plot.getAxis('bottom').setTextPen(theme.color('axis_text'))
        self.time_plot.getAxis('left').setTextPen(theme.color('axis_text'))

    def _style_channel_plots(self):
        for index, plot in enumerate(self.channel_plots):
            plot.setTitle(self._plot_signature[index][0] if self._plot_signature else '',
                          color=theme.color('text'), size='9pt')
            axis = plot.getAxis('left')
            axis.setPen(pg.mkPen(theme.color('axis_subtle'), width=0.6))
            axis.setTextPen(theme.color('axis_text'))
            plot.getAxis('bottom').setPen(None)
            for curve, kind in self.channel_curves[index]:
                curve.setPen(pg.mkPen(theme.trace_color(kind), width=1.0))
        self._style_regions()

    def _style_regions(self):
        brush = pg.mkBrush(58, 134, 255, 52 if theme.current_theme == 'light' else 68)
        pen = pg.mkPen(theme.color('accent'), width=1.0)
        for region in self.channel_regions:
            region.setBrush(brush)
            for line in region.lines:
                line.setPen(pen)

    def _toggle_theme(self):
        theme.toggle_theme()
        self._apply_theme()

    def _style_mode_buttons(self):
        """保证 Fusion 样式在运行时换主题后仍重绘选中文字。"""
        for button in self.btn_modes.values():
            if button.isChecked():
                button.setStyleSheet(
                    'QPushButton {'
                    f'background-color: {theme.color("accent")}; color: #ffffff; '
                    f'border: 1px solid {theme.color("accent")};'
                    '}')
            else:
                button.setStyleSheet('')

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 16)

        # 原生菜单栏（QMainWindow 自带的窗口菜单栏）
        self._build_menu_bar()

        # 左侧：图 + 控制栏
        left = QtWidgets.QVBoxLayout()

        # 顶部工具行：右上角浅色/深色切换按钮
        top_row = QtWidgets.QHBoxLayout()
        top_row.addStretch(1)
        self.btn_theme = QtWidgets.QPushButton('深色模式')
        self.btn_theme.setFixedWidth(96)
        self.btn_theme.clicked.connect(self._toggle_theme)
        top_row.addWidget(self.btn_theme)
        left.addLayout(top_row)

        # 顶部试次条、中央通道子图、底部唯一时间轴共享 X 范围。
        self._trial_vb = _ViewBox(self._on_wheel)
        self.trial_plot = pg.PlotWidget(viewBox=self._trial_vb)
        self.trial_plot.setFixedHeight(44)

        # 事件条样式（完全无轴：只显示试次色带 + 文字）
        self.trial_plot.showAxis('left')
        self.trial_plot.getAxis('left').setStyle(showValues=False)
        self.trial_plot.getAxis('left').setWidth(62)
        self.trial_plot.hideAxis('bottom')
        self.trial_plot.hideButtons()
        self.trial_plot.setMouseEnabled(x=False, y=False)

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.plot_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.channel_glw = pg.GraphicsLayoutWidget()
        self.glw = self.channel_glw  # 兼容已有测试/外部引用
        self.plot_scroll.setWidget(self.channel_glw)

        self._time_vb = _ViewBox(self._on_wheel)
        self.time_plot = pg.PlotWidget(viewBox=self._time_vb)
        self.time_plot.setFixedHeight(42)
        self.time_plot.hideButtons()
        self.time_plot.setMouseEnabled(x=False, y=False)
        self.time_plot.setLabel('bottom', '时间 (s)')
        self.time_plot.getAxis('bottom').enableAutoSIPrefix(False)
        self.time_plot.getAxis('bottom').setStyle(tickFont=QtGui.QFont(_CN_FONT, 9))
        self.time_plot.showAxis('left')
        self.time_plot.getAxis('left').setStyle(showValues=False)
        self.time_plot.getAxis('left').setWidth(62)

        left.addWidget(self.trial_plot)
        left.addWidget(self.plot_scroll, stretch=1)
        left.addWidget(self.time_plot)

        # 控制栏
        ctrl_row = QtWidgets.QHBoxLayout()

        self.btn_modes = {}
        for mode, text in PRIMARY_MODES:
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setSizePolicy(QtWidgets.QSizePolicy.Fixed,
                            QtWidgets.QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, m=mode: self._set_mode(m))
            ctrl_row.addWidget(b)
            self.btn_modes[mode] = b
        self.btn_modes[core.MODE_ORIG].setChecked(True)

        self.intensity_variant = QtWidgets.QComboBox()
        for text, mode in INTENSITY_VARIANTS:
            self.intensity_variant.addItem(text, mode)
        self.intensity_variant.setFixedWidth(126)
        self.intensity_variant.currentIndexChanged.connect(
            self._on_intensity_variant_changed)
        self.intensity_variant.setVisible(False)
        ctrl_row.addWidget(self.intensity_variant)

        ctrl_row.addSpacing(12)

        self.slider = _JumpSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1)
        self.slider.setMinimumWidth(240)
        self.slider.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Fixed)
        self.slider.valueChanged.connect(self._on_slider)
        ctrl_row.addWidget(self.slider, stretch=1)

        left.addLayout(ctrl_row)

        # 状态文本单独占一行。若与模式按钮/滑块同排，高 DPI 下其 sizeHint 会挤压时间滑块。
        self.info_label = QtWidgets.QLabel('')
        self.info_label.setStyleSheet(f'color: {theme.color("muted")}; font-size: 12px;')
        self.info_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight
                                     | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.info_label.setMinimumWidth(0)
        self.info_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                                      QtWidgets.QSizePolicy.Fixed)
        left.addWidget(self.info_label)

        # 底部按钮行
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(6)
        self.btn_region = QtWidgets.QPushButton('选区')
        self.btn_region.setCheckable(True)
        self.btn_region.toggled.connect(self._toggle_region)
        btn_row.addWidget(self.btn_region)
        self.region_label = QtWidgets.QLabel('')
        self.region_label.setStyleSheet(f'color: {theme.color("info")}; font-size: 12px;')
        self.region_label.setVisible(False)
        btn_row.addWidget(self.region_label)
        self.btn_save_sel = QtWidgets.QPushButton('保存选区')
        self.btn_save_sel.clicked.connect(self._save_selection)
        self.btn_sel_manager = QtWidgets.QPushButton('选区管理')
        self.btn_sel_manager.clicked.connect(self._open_selection_manager)
        self.btn_annotation_mgr = QtWidgets.QPushButton('标注管理')
        self.btn_annotation_mgr.clicked.connect(self._open_annotation_manager)
        self.btn_filter = QtWidgets.QPushButton('血氧滤波')
        self.btn_filter.clicked.connect(self._open_filter_dialog)
        self.btn_exit = QtWidgets.QPushButton('返回列表')
        self.btn_exit.clicked.connect(self._on_exit)
        self.btn_prev_label = QtWidgets.QPushButton('上一标签')
        self.btn_prev_label.clicked.connect(lambda: self._goto_label(-1))
        self.btn_next_label = QtWidgets.QPushButton('下一标签')
        self.btn_next_label.clicked.connect(lambda: self._goto_label(1))
        for b in (self.btn_save_sel, self.btn_sel_manager, self.btn_annotation_mgr,
                  self.btn_filter, self.btn_prev_label, self.btn_next_label,
                  self.btn_exit):
            btn_row.addWidget(b)

        btn_row.addStretch(1)
        left.addLayout(btn_row)

        root.addLayout(left, stretch=1)

        # 右侧图例
        self.legend_widget = QtWidgets.QWidget()
        self.legend_widget.setObjectName('legendPanel')
        self.legend_widget.setFixedWidth(250)
        self.legend_layout = QtWidgets.QVBoxLayout(self.legend_widget)
        self.legend_layout.setContentsMargins(10, 10, 10, 10)
        self.legend_layout.setSpacing(4)
        root.addWidget(self.legend_widget)

    def _build_menu_bar(self):
        """原生菜单栏：帮助(H) → 快捷键提示。"""
        menu_help = self.menuBar().addMenu('帮助(&H)')
        action = menu_help.addAction('快捷键提示')
        action.triggered.connect(self._show_shortcut_hints)

    def _show_shortcut_hints(self):
        """弹出当前支持的快捷键列表。"""
        rows = [
            ('滚轮', '前后移动时间窗口'),
            ('Ctrl + 滚轮', '时间轴缩放'),
            ('Alt + 滚轮', '调整子图高度'),
            ('← / → 方向键', '微调时间窗口（±100 ms）'),
            ('Esc', '退出查看器'),
        ]
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('快捷键提示')
        dlg.resize(420, 260)
        lay = QtWidgets.QVBoxLayout(dlg)
        table = QtWidgets.QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(['快捷键', '功能'])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for r, (key, desc) in enumerate(rows):
            table.setItem(r, 0, QtWidgets.QTableWidgetItem(key))
            table.setItem(r, 1, QtWidgets.QTableWidgetItem(desc))
        lay.addWidget(table)

        btn_row = QtWidgets.QHBoxLayout()
        btn_close = QtWidgets.QPushButton('关闭')
        btn_close.clicked.connect(dlg.accept)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec()

    def _open_filter_dialog(self):
        """弹出血氧滤波控制对话框。

        采样率固定为 processed_fs（5 Hz）；带通上下限频率可输入，
        默认 0.01–0.5 Hz（Butterworth 4 阶 + filtfilt 零相位）。
        """
        fs = float(self.data.processed_fs)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('血氧滤波控制')
        dlg.resize(380, 220)
        lay = QtWidgets.QVBoxLayout(dlg)

        # 采样率提示（只读）
        lbl_fs = QtWidgets.QLabel(
            f'数据采样率：{fs:g} Hz（奈奎斯特频率 {fs/2:g} Hz）')
        lbl_fs.setStyleSheet(
            f'color: {theme.color("muted")}; font-size: 12px;')
        lay.addWidget(lbl_fs)

        # 当前生效的滤波范围，用于预填；未启用时给默认 0.01–0.5 Hz
        current = self.data.blood_oxygen_filter
        if current is not None:
            low_default, high_default = current
        else:
            low_default, high_default = 0.01, 0.5

        form = QtWidgets.QFormLayout()
        spin_low = QtWidgets.QDoubleSpinBox()
        spin_low.setRange(0.001, fs / 2.0 - 0.001)
        spin_low.setDecimals(3)
        spin_low.setSingleStep(0.01)
        spin_low.setValue(low_default)
        spin_low.setSuffix(' Hz')
        spin_high = QtWidgets.QDoubleSpinBox()
        spin_high.setRange(0.001, fs / 2.0 - 0.001)
        spin_high.setDecimals(3)
        spin_high.setSingleStep(0.05)
        spin_high.setValue(high_default)
        spin_high.setSuffix(' Hz')
        form.addRow('下限频率 (Hz)：', spin_low)
        form.addRow('上限频率 (Hz)：', spin_high)
        lay.addLayout(form)

        def apply():
            try:
                self.data.apply_blood_oxygen_filter(
                    spin_low.value(), spin_high.value())
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(dlg, '血氧滤波', str(exc))
                return
            self._plot_signature = None
            self._update_plot()
            dlg.accept()

        def clear():
            self.data.clear_blood_oxygen_filter()
            self._plot_signature = None
            self._update_plot()
            dlg.accept()

        btn_row = QtWidgets.QHBoxLayout()
        btn_apply = QtWidgets.QPushButton('应用')
        btn_apply.clicked.connect(apply)
        btn_clear = QtWidgets.QPushButton('关闭滤波')
        btn_clear.clicked.connect(clear)
        btn_close = QtWidgets.QPushButton('关闭')
        btn_close.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_clear)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)
        dlg.exec()

    def _build_legend(self):
        # 清空旧图例。必须递归处理 addLayout() 加入的行，否则旧色块、
        # 标签和复选框会在多次切换模式/主题后叠在右栏顶部。
        def clear_layout(layout):
            while layout.count():
                item = layout.takeAt(0)
                child_layout = item.layout()
                if child_layout is not None:
                    clear_layout(child_layout)
                    child_layout.deleteLater()
                widget = item.widget()
                if widget is not None:
                    widget.hide()
                    widget.setParent(None)
                    widget.deleteLater()

        clear_layout(self.legend_layout)

        def add_title(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f'font-weight: bold; font-size: 15px; color: {theme.color("title")}; margin-top: 8px;')
            self.legend_layout.addWidget(lbl)

        def add_row(color, text):
            row = QtWidgets.QHBoxLayout()
            swatch = QtWidgets.QLabel()
            swatch.setFixedSize(32, 22)
            if isinstance(color, str):
                rgba255 = pg.mkColor(color).getRgb()
            else:
                rgba255 = tuple(int(c * 255) for c in color)
            swatch.setStyleSheet('background-color: rgba(%d,%d,%d,%d); border-radius: 3px;' %
                                 rgba255)
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(f'font-size: 14px; color: {theme.color("text")};')
            row.addWidget(swatch)
            row.addWidget(lbl, stretch=1)
            row.addStretch(0)
            self.legend_layout.addLayout(row)

        # 试次
        add_title('试次 / Trial')
        bmap = core.BEHAVIOR_LABELS.get(self.data.task_type, {})
        for m, (zh, en) in bmap.items():
            add_row(core.BEHAVIOR_COLORS.get(m, (0.85, 0.85, 0.85, 0.9)), f'{zh} {en}')

        # 任务 epoch 人工质检标注面板
        self._build_annotation_panel()

        # 光分组（已弃用，注释保留）
        # add_title('光分组 / TDM')
        # for m in self.data.unique_markers:
        #     if m == 0:
        #         continue
        #     add_row(core.MARKER_COLORS[m], core.MARKER_LABELS.get(m, str(m)))

        # 曲线颜色（随模式动态）
        add_title('曲线颜色 / Trace')
        for rgba, text in self._trace_legend_rows():
            add_row(rgba, text)

        # 通道显示选择
        self._build_channel_selector()

    def _build_channel_selector(self):
        """构建内嵌通道复选框；不再提供功能重复的选择弹窗。"""
        title = QtWidgets.QLabel('通道显示 / Channels')
        title.setStyleSheet(f'font-weight: bold; font-size: 15px; color: {theme.color("title")}; margin-top: 8px;')
        self.legend_layout.addWidget(title)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_ch_all = QtWidgets.QPushButton('全选')
        self.btn_ch_none = QtWidgets.QPushButton('清空')
        self.btn_ch_all.clicked.connect(lambda: self._set_all_channels(True))
        self.btn_ch_none.clicked.connect(lambda: self._set_all_channels(False))
        btn_row.addWidget(self.btn_ch_all)
        btn_row.addWidget(self.btn_ch_none)
        self.legend_layout.addLayout(btn_row)

        self._channel_scroll = QtWidgets.QScrollArea()
        self._channel_scroll.setObjectName('channelScroll')
        self._channel_scroll.setWidgetResizable(True)
        self._channel_scroll.setMinimumHeight(320)
        self._channel_scroll.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                           QtWidgets.QSizePolicy.Expanding)
        self._channel_container = QtWidgets.QWidget()
        self._channel_container.setObjectName('channelContainer')
        self._channel_box_layout = QtWidgets.QVBoxLayout(self._channel_container)
        self._channel_box_layout.setContentsMargins(4, 4, 4, 4)
        self._channel_box_layout.setSpacing(1)
        self._channel_scroll.setWidget(self._channel_container)
        self.legend_layout.addWidget(self._channel_scroll, stretch=1)
        self._populate_channel_checkboxes()

    def _populate_channel_checkboxes(self):
        """按当前模式重建复选框列表，并依据 data.selected_channels 勾选。"""
        while self._channel_box_layout.count():
            item = self._channel_box_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self.channel_checkboxes = []
        selected = self.data.selected_channels
        for name in self.data.channel_options():
            cb = QtWidgets.QCheckBox(name)
            cb.setStyleSheet(f'color: {theme.color("text")}; font-size: 12px;')
            cb.blockSignals(True)
            cb.setChecked(selected is None or name in selected)
            cb.blockSignals(False)
            cb.toggled.connect(lambda _=False, n=name: self._on_channel_toggled(n))
            self._channel_box_layout.addWidget(cb)
            self.channel_checkboxes.append((name, cb))
        self._channel_box_layout.addStretch(1)

    def _refresh_channel_checkboxes(self):
        """仅同步复选框勾选状态（不重建）。"""
        selected = self.data.selected_channels
        for name, cb in self.channel_checkboxes:
            cb.blockSignals(True)
            cb.setChecked(selected is None or name in selected)
            cb.blockSignals(False)

    def _set_all_channels(self, checked):
        self.data.selected_channels = None if checked else set()
        group = 'orig' if self.data.current_mode == core.MODE_ORIG else 'processed'
        self._selection_by_group[group] = self.data.selected_channels
        self._refresh_channel_checkboxes()
        self._plot_signature = None
        self._update_plot()

    def _on_channel_toggled(self, _name):
        names = {name for name, cb in self.channel_checkboxes if cb.isChecked()}
        self.data.selected_channels = names if names else set()
        group = 'orig' if self.data.current_mode == core.MODE_ORIG else 'processed'
        self._selection_by_group[group] = self.data.selected_channels
        self._plot_signature = None
        self._update_plot()

    def _trace_legend_rows(self):
        """图例与实际曲线共用 theme.trace_color。"""
        m = self.data.current_mode
        if m == core.MODE_FNIRS:
            return [(theme.trace_color('HbO'), 'HbO 含氧'),
                    (theme.trace_color('HbR'), 'HbR 脱氧')]
        if m in (core.MODE_INTENSITY, core.MODE_RELATIVE, core.MODE_OD):
            return [(theme.trace_color('850'), '850nm'),
                    (theme.trace_color('780'), '780nm')]
        if m == core.MODE_780:
            return [(theme.trace_color('780'), '780nm')]
        if m == core.MODE_850:
            return [(theme.trace_color('850'), '850nm')]
        return [(_GRAY, '通道')]

    # ---------------------------------------------------------------
    # Epoch 标注面板
    # ---------------------------------------------------------------
    def _build_annotation_panel(self):
        """在右侧图例栏构建固定的任务 epoch 标注面板。"""
        title = QtWidgets.QLabel('Epoch 标注')
        title.setStyleSheet(
            f'font-weight: bold; font-size: 15px; color: {theme.color("title")}; margin-top: 8px;')
        self.legend_layout.addWidget(title)

        self.annotation_epoch_label = QtWidgets.QLabel('未选中任务 epoch')
        self.annotation_epoch_label.setWordWrap(True)
        self.annotation_epoch_label.setStyleSheet(
            f'font-size: 12px; color: {theme.color("muted")};')
        self.legend_layout.addWidget(self.annotation_epoch_label)

        # 标签单选按钮组（互斥）
        self.annotation_label_buttons = {}
        self.annotation_label_group = QtWidgets.QButtonGroup(self)
        self.annotation_label_group.setExclusive(True)
        btn_row = QtWidgets.QHBoxLayout()
        for key, zh in core.ANNOTATION_LABELS.items():
            b = QtWidgets.QPushButton(zh)
            b.setCheckable(True)
            self.annotation_label_group.addButton(b)
            self.annotation_label_buttons[key] = b
            btn_row.addWidget(b)
        self.legend_layout.addLayout(btn_row)

        self.annotation_note_edit = QtWidgets.QLineEdit()
        self.annotation_note_edit.setPlaceholderText('备注（可选）')
        self.legend_layout.addWidget(self.annotation_note_edit)

        action_row = QtWidgets.QHBoxLayout()
        self.btn_annotation_save = QtWidgets.QPushButton('保存标注')
        self.btn_annotation_clear = QtWidgets.QPushButton('清除标注')
        self.btn_annotation_save.clicked.connect(self._save_annotation)
        self.btn_annotation_clear.clicked.connect(self._clear_annotation)
        action_row.addWidget(self.btn_annotation_save)
        action_row.addWidget(self.btn_annotation_clear)
        self.legend_layout.addLayout(action_row)

        self._refresh_annotation_panel()

    def _refresh_annotation_panel(self):
        """根据当前选中 epoch 刷新标注面板（文案、预选标签、回填备注、控件可用性）。"""
        rec = self.current_epoch_record
        has_epoch = bool(rec and rec.get('epoch_index') is not None)
        if has_epoch:
            self.annotation_epoch_label.setText(
                f'{rec["display_label"]} · '
                f'{rec["start_time"]:.1f}s–{rec["end_time"]:.1f}s')
        else:
            self.annotation_epoch_label.setText('未选中任务 epoch')

        ann = rec.get('annotation') if rec else None
        label = ann.get('label') if ann else None
        note = ann.get('note', '') if ann else ''

        # exclusive 组禁止取消最后一个选中按钮，先解除互斥再重置选中态
        self.annotation_label_group.setExclusive(False)
        for key, b in self.annotation_label_buttons.items():
            b.blockSignals(True)
            b.setChecked(key == label)
            b.blockSignals(False)
            b.setEnabled(has_epoch)
        self.annotation_label_group.setExclusive(True)

        self.annotation_note_edit.blockSignals(True)
        self.annotation_note_edit.setText(note)
        self.annotation_note_edit.blockSignals(False)
        self.annotation_note_edit.setEnabled(has_epoch)
        self.btn_annotation_save.setEnabled(has_epoch)
        self.btn_annotation_clear.setEnabled(has_epoch and label is not None)

    def _on_epoch_selected(self, record):
        """点击任务 epoch 后更新标注面板（选区分隔逻辑由 _select_trial_record 负责）。"""
        self.current_epoch_record = record
        if hasattr(self, 'annotation_epoch_label'):
            self._refresh_annotation_panel()

    def _save_annotation(self):
        rec = self.current_epoch_record
        if not rec or rec.get('epoch_index') is None:
            return
        label = None
        for key, b in self.annotation_label_buttons.items():
            if b.isChecked():
                label = key
                break
        if label is None:
            QtWidgets.QMessageBox.information(self, 'Epoch 标注', '请先选择标注标签')
            return
        note = self.annotation_note_edit.text().strip()
        self.data.set_epoch_annotation(rec['epoch_index'], label, note)
        rec['annotation'] = self.data.get_epoch_annotation(rec['epoch_index'])
        self._refresh_annotation_panel()
        self._update_plot()

    def _clear_annotation(self):
        rec = self.current_epoch_record
        if not rec or rec.get('epoch_index') is None:
            return
        self.data.clear_epoch_annotation(rec['epoch_index'])
        rec['annotation'] = None
        self._refresh_annotation_panel()
        self._update_plot()

    def _install_shortcuts(self):
        # 方向键：±100ms
        for key, delta in [(QtCore.Qt.Key_Left, -0.1), (QtCore.Qt.Key_Right, 0.1)]:
            sc = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            sc.activated.connect(lambda d=delta: self._nudge(d))
        # Esc 退出查看器
        sc = QtGui.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self)
        sc.activated.connect(self._on_exit)

    # ---------------------------------------------------------------
    # 交互回调
    # ---------------------------------------------------------------
    def _set_mode(self, mode):
        if mode not in core.DISPLAY_MODES:
            return
        old_group = 'orig' if self.data.current_mode == core.MODE_ORIG else 'processed'
        self._selection_by_group[old_group] = self.data.selected_channels
        if mode == core.MODE_INTENSITY:
            mode = self.intensity_variant.currentData()
        self.data.current_mode = mode
        new_group = 'orig' if mode == core.MODE_ORIG else 'processed'
        self.data.selected_channels = self._selection_by_group[new_group]
        primary = (core.MODE_ORIG if mode == core.MODE_ORIG else
                   core.MODE_FNIRS if mode == core.MODE_FNIRS else
                   core.MODE_INTENSITY)
        for m, b in self.btn_modes.items():
            b.setChecked(m == primary)
        self._style_mode_buttons()
        self.intensity_variant.setVisible(primary == core.MODE_INTENSITY)
        self._plot_signature = None
        self.plot_scroll.verticalScrollBar().setValue(0)
        self._build_legend()
        self._update_plot()

    def _on_intensity_variant_changed(self, _index):
        if self.intensity_variant.isVisible():
            self._set_mode(self.intensity_variant.currentData())

    def _on_slider(self, value):
        # QSlider value = 窗口起始样本
        self.data.current_start = value
        self._update_plot()

    def _nudge(self, delta_sec):
        current_time = float(self.data.raw_time[self.data.current_start])
        new = int(np.searchsorted(self.data.raw_time, current_time + delta_sec))
        new = max(0, min(new, self.data.n_samples - self.data.samples_per_window))
        self.data.current_start = new
        self._update_plot()

    def _goto_label(self, direction):
        """跳转到上一个/下一个行为试次标签（100/110/111/112）位置。direction: -1 上一、1 下一。"""
        positions = self.data.behavior_label_positions()
        if not positions:
            return
        cur = self.data.current_start
        if direction < 0:
            candidates = [p for p in positions if p < cur]
            target = max(candidates) if candidates else None
        else:
            candidates = [p for p in positions if p > cur]
            target = min(candidates) if candidates else None
        if target is None:
            return
        max_start = max(0, self.data.n_samples - self.data.samples_per_window)
        self.data.current_start = max(0, min(target, max_start))
        self._update_plot()

    def _on_wheel(self, delta, ctrl, alt, pos, source_viewbox):
        step = 1 if delta > 0 else -1
        if alt:
            self.channel_plot_height = max(
                80, min(220, self.channel_plot_height + step * 10))
            self._apply_channel_height()
            return
        if ctrl:
            # Ctrl+滚轮 = 缩放
            factor = 0.8 if step > 0 else 1.25
            new_width = int(self.data.samples_per_window * factor)
            new_width = max(self.data.min_win_samples, min(self.data.max_win_samples, new_width))
            if new_width == self.data.samples_per_window:
                return
            old_width = self.data.samples_per_window
            old_start = self.data.current_start
            # 鼠标锚定
            mouse_x = source_viewbox.mapSceneToView(pos).x()
            sample_mouse = int(np.searchsorted(self.data.raw_time, mouse_x))
            new_start = int(round(sample_mouse - (sample_mouse - old_start) * (new_width / old_width)))
            new_start = max(0, min(new_start, self.data.n_samples - new_width))
            self.data.samples_per_window = new_width
            self.data.current_start = new_start
        else:
            # 普通滚轮 = 前后进度（向上=后退、向下=前进）
            delta_samples = int(self.data.samples_per_window * 0.5) * (-1 if step > 0 else 1)
            new = max(0, min(self.data.current_start + delta_samples,
                             self.data.n_samples - self.data.samples_per_window))
            self.data.current_start = new
        self._update_plot()

    def _sync_slider(self):
        maxv = max(0, self.data.n_samples - self.data.samples_per_window)
        self.slider.blockSignals(True)
        self.slider.setRange(0, maxv)
        self.slider.setValue(self.data.current_start)
        self.slider.blockSignals(False)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'channel_glw'):
            self._apply_channel_height()

    def _save_selection(self):
        a, b = sorted(self._region_bounds)
        if b > a:
            self.selections.append((float(a), float(b)))
            self._draw_selections()

    def _open_selection_manager(self):
        if not self.selections:
            QtWidgets.QMessageBox.information(self, '选区', '暂无已保存的选区')
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('选区管理')
        lay = QtWidgets.QVBoxLayout(dlg)
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(['#', '起始(s)', '结束(s)', '长度(s)'])
        for i, (a, b) in enumerate(self.selections):
            it = QtWidgets.QTreeWidgetItem(
                [str(i + 1), f'{a:.3f}', f'{b:.3f}', f'{b - a:.3f}'])
            tree.addTopLevelItem(it)
        lay.addWidget(tree)

        row = QtWidgets.QHBoxLayout()
        btn_del = QtWidgets.QPushButton('删除选中')
        btn_clear = QtWidgets.QPushButton('清空全部')
        row.addWidget(btn_del)
        row.addWidget(btn_clear)
        row.addStretch(1)
        lay.addLayout(row)

        def delete_sel():
            idxs = sorted([tree.indexOfTopLevelItem(it) for it in tree.selectedItems()], reverse=True)
            for k in idxs:
                if 0 <= k < len(self.selections):
                    del self.selections[k]
            tree.clear()
            for i, (a, b) in enumerate(self.selections):
                tree.addTopLevelItem(QtWidgets.QTreeWidgetItem(
                    [str(i + 1), f'{a:.3f}', f'{b:.3f}', f'{b - a:.3f}']))
            self._draw_selections()

        def clear_all():
            self.selections.clear()
            tree.clear()
            self._draw_selections()

        btn_del.clicked.connect(delete_sel)
        btn_clear.clicked.connect(clear_all)
        dlg.exec()

    def _open_annotation_manager(self):
        """查看当前 stream 中所有已标注的任务 epoch。"""
        records = self.data._trial_records_in_window(0, self.data.n_samples)
        annotated = [r for r in records if r.get('annotation')]
        if not annotated:
            QtWidgets.QMessageBox.information(self, '标注管理', '暂无 Epoch 标注')
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('标注管理')
        dlg.resize(560, 440)
        lay = QtWidgets.QVBoxLayout(dlg)
        tree = QtWidgets.QTreeWidget()
        tree.setRootIsDecorated(False)
        tree.setHeaderLabels(['Epoch', '标签', '备注', '时间区间 (s)'])
        for r in annotated:
            ann = r['annotation']
            zh = core.ANNOTATION_LABELS.get(ann.get('label'), ann.get('label'))
            note = ann.get('note', '') or ''
            it = QtWidgets.QTreeWidgetItem([
                r['display_label'], zh, note,
                f"{r['start_time']:.1f} – {r['end_time']:.1f}"])
            tree.addTopLevelItem(it)
        for i in range(4):
            tree.resizeColumnToContents(i)
        # 内容列设最小宽度，备注列拉伸占满剩余空间，避免列宽过窄
        for i, w in ((0, 90), (1, 70), (3, 150)):
            if tree.columnWidth(i) < w:
                tree.setColumnWidth(i, w)
        header = tree.header()
        header.setSectionResizeMode(QtWidgets.QHeaderView.Interactive)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        lay.addWidget(tree)

        row = QtWidgets.QHBoxLayout()
        btn_close = QtWidgets.QPushButton('关闭')
        btn_close.clicked.connect(dlg.accept)
        row.addStretch(1)
        row.addWidget(btn_close)
        lay.addLayout(row)
        dlg.exec()

    def _toggle_region(self, checked):
        for region in self.channel_regions:
            region.setVisible(checked)
        self.region_label.setVisible(checked)
        if checked:
            self._on_region_changed()

    def _on_channel_region_changed(self, source):
        if self._syncing_regions:
            return
        self._syncing_regions = True
        self._region_bounds = tuple(sorted(source.getRegion()))
        for region in self.channel_regions:
            if region is not source:
                region.setRegion(self._region_bounds)
        self._syncing_regions = False
        self._on_region_changed()

    def _on_region_changed(self):
        a, b = sorted(self._region_bounds)
        i0 = int(np.searchsorted(self.data.raw_time, a, side='left'))
        i1 = int(np.searchsorted(self.data.raw_time, b, side='right'))
        n = max(0, i1 - i0)
        self.region_label.setText(f'选区 {a:.2f}s ~ {b:.2f}s ({n} 样本)')

    def _on_exit(self):
        self.close()

    # ---------------------------------------------------------------
    # 绘图
    # ---------------------------------------------------------------
    def _unit_for_mode(self):
        return ('μM' if self.data.current_mode == core.MODE_FNIRS else
                '%' if self.data.current_mode == core.MODE_RELATIVE else
                'OD' if self.data.current_mode == core.MODE_OD else
                'ADC')

    @staticmethod
    def _signature_for_rows(rows):
        return tuple((row['label'], tuple(s['kind'] for s in row['series']))
                     for row in rows)

    def _rebuild_channel_plots(self, rows):
        self.channel_glw.clear()
        self.channel_plots = []
        self.channel_curves = []
        self.channel_regions = []
        self.saved_region_items = []
        self._plot_signature = self._signature_for_rows(rows)
        master_plot = None
        for row_index, row in enumerate(rows):
            plot = self.channel_glw.addPlot(
                row=row_index, col=0, viewBox=_ViewBox(self._on_wheel))
            plot.hideButtons()
            plot.setMouseEnabled(x=False, y=False)
            plot.hideAxis('bottom')
            plot.getAxis('left').setWidth(62)
            plot.getAxis('left').setStyle(
                tickFont=QtGui.QFont(_CN_FONT, 8),
                textFillLimits=[(0, 1.0)])
            plot.setTitle(row['label'], color=theme.color('text'), size='9pt')
            if master_plot is None:
                master_plot = plot
            else:
                plot.setXLink(master_plot)

            curves = []
            for series in row['series']:
                curve = plot.plot([], [],
                                  pen=pg.mkPen(theme.trace_color(series['kind']), width=1.0),
                                  connect='finite')
                curve.setClipToView(True)
                curve.setDownsampling(auto=True, method='peak')
                curves.append((curve, series['kind']))
            self.channel_plots.append(plot)
            self.channel_curves.append(curves)

            region = _SelectRegion(
                values=self._region_bounds, brush=pg.mkBrush(58, 134, 255, 52),
                pen=pg.mkPen(theme.color('accent')), movable=True)
            region.setZValue(10)
            region.setVisible(self.btn_region.isChecked())
            region.setCursor(QtCore.Qt.CursorShape.OpenHandCursor)
            for line in region.lines:
                line.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)
            region.sigRegionChanged.connect(
                lambda *_, r=region: self._on_channel_region_changed(r))
            plot.addItem(region)
            self.channel_regions.append(region)

        if master_plot is not None:
            self.trial_plot.setXLink(master_plot)
            self.time_plot.setXLink(master_plot)
        else:
            self.trial_plot.setXLink(None)
            self.time_plot.setXLink(None)
        self._apply_channel_height()
        self._style_channel_plots()
        self._draw_selections()

    def _apply_channel_height(self):
        if not self.channel_plots:
            self.channel_glw.setMinimumHeight(1)
            return
        for plot in self.channel_plots:
            plot.setMinimumHeight(self.channel_plot_height)
            plot.setMaximumHeight(self.channel_plot_height)
        required = len(self.channel_plots) * self.channel_plot_height
        self.channel_glw.setMinimumHeight(
            max(required, max(1, self.plot_scroll.viewport().height())))

    def _format_y_value(self, value):
        unit = self._unit_for_mode()
        if unit in ('ADC', 'raw') and abs(value) >= 100:
            return f'{value:.0f}'
        if unit == 'OD':
            return f'{value:.4f}'
        if value == 0:
            return '0'
        if abs(value) < 1e-3:
            return f'{value:.1e}'
        if abs(value) >= 100:
            return f'{value:.1f}'
        if abs(value) >= 1:
            return f'{value:.2f}'
        return f'{value:.3f}'

    def _set_channel_y_range(self, plot, row):
        parts = [s['y'][np.isfinite(s['y'])] for s in row['series']]
        parts = [part for part in parts if part.size]
        if not parts:
            lo, hi = -1.0, 1.0
        else:
            values = np.concatenate(parts)
            lo, hi = float(np.min(values)), float(np.max(values))
            if self.data.current_mode in (core.MODE_RELATIVE, core.MODE_OD, core.MODE_FNIRS):
                lo, hi = min(lo, 0.0), max(hi, 0.0)
            span = hi - lo
            pad = (max(abs(lo), abs(hi), 1.0) * 0.05
                   if span <= max(1e-12, abs(lo) * 1e-12, abs(hi) * 1e-12)
                   else span * 0.08)
            lo, hi = lo - pad, hi + pad
        ticks = [lo, 0.0, hi] if lo < 0.0 < hi else [lo, (lo + hi) / 2.0, hi]
        # 刻度若正好位于 ViewBox 边界，文字会被上下裁掉。
        outer_pad = max((hi - lo) * 0.15, 1e-12)
        plot.setYRange(lo - outer_pad, hi + outer_pad, padding=0)
        plot.getAxis('left').setTicks([[
            (value, self._format_y_value(value)) for value in ticks]])

    def _select_trial_record(self, record):
        """将点击的完整试次设为活动选区，但不自动保存。"""
        start = float(record['start_time'])
        end = float(record['end_time'])
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            return
        self._region_bounds = (start, end)
        self._syncing_regions = True
        try:
            for region in self.channel_regions:
                region.setRegion(self._region_bounds)
        finally:
            self._syncing_regions = False
        if not self.btn_region.isChecked():
            self.btn_region.setChecked(True)
        else:
            for region in self.channel_regions:
                region.setVisible(True)
            self._on_region_changed()
        self._on_epoch_selected(record)

    def _draw_trials(self, trial_records, t_lo, t_hi):
        self.trial_plot.clear()
        self.trial_plot.setYRange(-0.15, 1, padding=0)
        for record in trial_records:
            s = record['visible_start']
            e = record['visible_end']
            rgba = record['color']
            label = record['display_label']
            color255 = _rgba255((rgba[0], rgba[1], rgba[2], 1.0))
            pen = pg.mkPen(color255, width=2)
            self.trial_plot.plot([s, e], [0.25, 0.25], pen=pen)
            self.trial_plot.plot([s, s], [0.25, -0.09], pen=pen)
            self.trial_plot.plot([e, e], [0.25, -0.09], pen=pen)
            if label and (e - s) > (t_hi - t_lo) * 0.03:
                item = pg.TextItem(label, color=pg.mkColor(color255), anchor=(0.5, 0.5))
                item.setFont(QtGui.QFont(_CN_FONT, 9, QtGui.QFont.Weight.Bold))
                self.trial_plot.addItem(item)
                item.setPos(max(t_lo, min(t_hi, (s + e) / 2)), 0.45)

            # 已标注 epoch 的视觉高亮（顶部标签色条）
            ann = record.get('annotation')
            if ann:
                acolor = core.ANNOTATION_LABEL_COLORS.get(
                    ann.get('label'), (1.0, 1.0, 1.0, 1.0))
                self.trial_plot.plot(
                    [s, e], [0.7, 0.7],
                    pen=pg.mkPen(
                        _rgba255((acolor[0], acolor[1], acolor[2], 1.0)),
                        width=4))

            tooltip = (f'{label}\n{record["start_time"]:.3f}s – '
                       f'{record["end_time"]:.3f}s')
            if ann:
                zh = core.ANNOTATION_LABELS.get(ann.get('label'), ann.get('label'))
                tooltip += f'\n标注: {zh}'
                if ann.get('note'):
                    tooltip += f' · {ann["note"]}'
            hit_item = _TrialHitItem(
                s, e, rgba,
                lambda r=record: self._select_trial_record(r), tooltip)
            self.trial_plot.addItem(hit_item)

    def _update_plot(self):
        if not hasattr(self, 'slider'):
            return
        self._sync_slider()
        data = self.data
        win_start = data.current_start
        win_end = min(win_start + data.samples_per_window, data.n_samples)
        wd = data.window_data(win_start, win_end)
        rows = wd['rows']
        if self._signature_for_rows(rows) != self._plot_signature:
            self._rebuild_channel_plots(rows)

        if wd['n_win'] > 0:
            t_lo, t_hi = float(wd['time_vec'][0]), float(wd['time_vec'][-1])
        else:
            t_lo = float(data.raw_time[min(win_start, data.n_samples - 1)])
            t_hi = float(data.raw_time[min(max(win_start, win_end - 1), data.n_samples - 1)])
        if t_hi <= t_lo:
            t_hi = t_lo + 1.0 / max(1.0, wd['display_fs'])

        for row_index, row in enumerate(rows):
            for (curve, _kind), series in zip(self.channel_curves[row_index], row['series']):
                curve.setData(wd['time_vec'], series['y'])
            self._set_channel_y_range(self.channel_plots[row_index], row)
        if self.channel_plots:
            self.channel_plots[0].setXRange(t_lo, t_hi, padding=0)
        else:
            self.time_plot.setXRange(t_lo, t_hi, padding=0)
            self.trial_plot.setXRange(t_lo, t_hi, padding=0)

        ra, rb = self._region_bounds
        if rb <= t_lo or ra >= t_hi:
            span = t_hi - t_lo
            self._region_bounds = (t_lo + span * 0.2, t_lo + span * 0.4)
            self._syncing_regions = True
            for region in self.channel_regions:
                region.setRegion(self._region_bounds)
            self._syncing_regions = False
            self._on_region_changed()
        self._draw_trials(wd['trial_records'], t_lo, t_hi)

        lost, raw_n_win = wd['lost'], wd['raw_n_win']
        loss_pct = (lost / (lost + raw_n_win) * 100.0
                    if (lost + raw_n_win) > 0 else 0.0)
        status = (f'已选 {wd["n_rows"]}/{len(data.channel_options())} · '
                  f'{wd["display_fs"]:g} Hz · {self._unit_for_mode()} · '
                  f'丢包 {lost} ({loss_pct:.2f}%) · 试次 {len(data.behavior_trials)}')
        if self.selections:
            status += f' · 选区 {len(self.selections)}'
        flt = data.blood_oxygen_filter
        if flt is not None:
            status += f' · 血氧滤波 {flt[0]:g}-{flt[1]:g} Hz'
        self.info_label.setText(status)
        self.info_label.setToolTip(
            f'模式: {data.current_mode}\n原始样本: {win_start}-{win_end}/{data.n_samples}\n'
            f'窗口: {t_lo:.3f}-{t_hi:.3f}s\n丢包跳变: {wd["jumps"]}')

    def _draw_selections(self):
        for plot, item in self.saved_region_items:
            plot.removeItem(item)
        self.saved_region_items = []
        for (a, b) in self.selections:
            for plot in self.channel_plots:
                region = pg.LinearRegionItem(
                    values=(a, b), brush=pg.mkColor(120, 120, 120, 48),
                    pen=pg.mkPen(theme.color('muted'), width=1,
                                 style=QtCore.Qt.PenStyle.DashLine),
                    movable=False)
                region.setZValue(-5)
                plot.addItem(region)
                self.saved_region_items.append((plot, region))
