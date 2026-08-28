# -*- coding: utf-8 -*-
"""
theme.py —— 浅色/深色主题管理器（供选择页 ui/selector.py 与查看器页 ui/viewer.py 共用）

通过 current_theme / set_theme / toggle_theme 维护全局主题状态；
qss() 生成当前主题的 QSS；pg_background()/pg_foreground() 供 pyqtgraph 使用；
color(key) 供内联样式读取当前主题的次级颜色。
"""

# 中文字体
_CN_FONT = 'Microsoft YaHei'


def _rgba255(rgba):
    """0-1 浮点 RGBA → 0-255 整数 RGBA（pyqtgraph 的 mkColor 需要 0-255）。"""
    return tuple(int(round(c * 255)) for c in rgba)


# 两套调色板
_THEMES = {
    'light': {
        'bg': '#f5f6f8',          # 窗口背景
        'panel': '#ffffff',       # 面板（图例等）
        'plot_bg': '#ffffff',     # 波形子图
        'fg': '#222222',          # 前景 / 主文字
        'muted': '#8a8a8a',       # 次级/提示文字
        'text': '#555555',        # 正文
        'title': '#222222',       # 标题
        'accent': '#3a86ff',      # 强调色
        'info': '#3a86ff',        # 信息高亮（选区标签等）
        'border': '#d5d5d5',      # 边框
        'scroll_bg': '#ffffff',   # 滚动区背景
        'grid': '#d0d0d0',        # 网格/分隔线
        'axis_subtle': '#c8cdd4', # 子图 Y 轴
        'axis_text': '#8a9098',   # 子图 Y 数值
        'ch': '#666666',          # 通道（Ch）曲线颜色
        'trace_red': '#e53935',
        'trace_blue': '#2979ff',
        'btn_bg': '#ffffff',
        'btn_hover': '#e9e9e9',
        'btn_pressed': '#d8d8d8',
        'btn_disabled_bg': '#efefef',
        'btn_disabled_fg': '#b0b0b0',
        'input_bg': '#ffffff',
        'table_bg': '#ffffff',
        'header_bg': '#eeeeee',
        'slider_groove': '#d5d5d5',
    },
    'dark': {
        'bg': '#2b3038',
        'panel': '#343b45',
        'plot_bg': '#303640',
        'fg': '#e5e9f0',
        'muted': '#aeb6c2',
        'text': '#d5dbe4',
        'title': '#f0f3f7',
        'accent': '#3a86ff',
        'info': '#7fb3ff',
        'border': '#596270',
        'scroll_bg': '#303640',
        'grid': '#596270',
        'axis_subtle': '#596270',
        'axis_text': '#aeb6c2',
        'ch': '#d8dee9',
        'trace_red': '#ff6b6b',
        'trace_blue': '#6ea8ff',
        'btn_bg': '#3a424d',
        'btn_hover': '#46505d',
        'btn_pressed': '#252a31',
        'btn_disabled_bg': '#303640',
        'btn_disabled_fg': '#7f8996',
        'input_bg': '#303640',
        'table_bg': '#343b45',
        'header_bg': '#3a424d',
        'slider_groove': '#596270',
    },
}

# 默认浅色
current_theme = 'light'


def set_theme(mode):
    """设置主题，mode 为 'light' 或 'dark'。"""
    global current_theme
    if mode in _THEMES:
        current_theme = mode


def toggle_theme():
    """在浅色/深色间切换，返回切换后的主题名。"""
    set_theme('dark' if current_theme == 'light' else 'light')
    return current_theme


def color(key, default='#000000'):
    """返回当前主题下指定 key 的颜色。"""
    return _THEMES[current_theme].get(key, default)


def pg_background():
    """pyqtgraph 背景色。"""
    return color('bg')


def trace_color(kind):
    """波长/血红蛋白曲线颜色；图例与曲线必须共用此入口。"""
    if kind in ('HbO', '850', '%850', 'OD850'):
        return color('trace_red')
    if kind in ('HbR', '780', '%780', 'OD780'):
        return color('trace_blue')
    return color('ch')


def pg_foreground():
    """pyqtgraph 前景色（文字/坐标轴）。"""
    return color('fg')


def qss():
    """生成当前主题的 QSS 字符串。"""
    p = _THEMES[current_theme]
    return """
QMainWindow, QDialog { background-color: %(bg)s; color: %(fg)s; }
QMenuBar { background-color: %(bg)s; color: %(fg)s; border-bottom: 1px solid %(border)s; }
QMenuBar::item { background: transparent; padding: 4px 10px; }
QMenuBar::item:selected { background: %(btn_hover)s; border-radius: 4px; }
QMenuBar::item:pressed { background: %(accent)s; color: white; }
QMenu { background-color: %(panel)s; color: %(fg)s; border: 1px solid %(border)s; }
QMenu::item { background: transparent; padding: 6px 24px 6px 20px; }
QMenu::item:selected { background-color: %(accent)s; color: white; }
QMenu::separator { height: 1px; background: %(border)s; margin: 4px 8px; }
QWidget { color: %(fg)s; font-family: "Microsoft YaHei"; }
QWidget#legendPanel {
    background-color: %(panel)s; border: 1px solid %(border)s; border-radius: 6px;
}
QScrollArea#channelScroll {
    background-color: %(scroll_bg)s; border: 1px solid %(border)s; border-radius: 4px;
}
QWidget#channelContainer { background-color: %(scroll_bg)s; border: none; }
QPushButton {
    background-color: %(btn_bg)s; color: %(fg)s; border: 1px solid %(border)s;
    border-radius: 4px; padding: 5px 12px; font-size: 13px;
}
QPushButton:hover { background-color: %(btn_hover)s; }
QPushButton:pressed { background-color: %(btn_pressed)s; }
QPushButton:checked { background-color: %(accent)s; color: white; border-color: %(accent)s; }
QPushButton:disabled { color: %(btn_disabled_fg)s; background-color: %(btn_disabled_bg)s; }
QSlider::groove:horizontal { height: 6px; background: %(slider_groove)s; border-radius: 3px; }
QSlider::handle:horizontal { background: %(accent)s; width: 16px; margin: -5px 0; border-radius: 8px; }
QSlider::sub-page:horizontal { background: %(accent)s; border-radius: 3px; }
QSlider::groove:vertical { width: 6px; background: %(slider_groove)s; border-radius: 3px; }
QSlider::handle:vertical { background: %(accent)s; height: 16px; margin: 0 -5px; border-radius: 8px; }
QSlider::sub-page:vertical { background: %(accent)s; border-radius: 3px; }
QLabel { font-size: 13px; }
QTableWidget, QTreeWidget { background-color: %(table_bg)s; gridline-color: %(border)s; font-size: 13px; }
QHeaderView::section { background-color: %(header_bg)s; color: %(fg)s; border: 1px solid %(border)s; padding: 4px; }
QLineEdit, QComboBox { background-color: %(input_bg)s; color: %(fg)s; border: 1px solid %(border)s; padding: 3px; }
QComboBox QAbstractItemView {
    background-color: %(input_bg)s; color: %(fg)s;
    border: 1px solid %(border)s; outline: 0;
    selection-background-color: %(accent)s; selection-color: white;
}
QComboBox QAbstractItemView::item { min-height: 26px; padding: 3px 8px; }
QComboBox QAbstractItemView::item:hover {
    background-color: %(btn_hover)s; color: %(fg)s;
}
QComboBox QAbstractItemView::item:selected {
    background-color: %(accent)s; color: white;
}
QDialog { background-color: %(bg)s; }
QScrollArea { background-color: %(bg)s; }
QCheckBox { color: %(fg)s; }
""" % p
