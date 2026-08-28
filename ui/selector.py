# -*- coding: utf-8 -*-
"""
ui/selector.py —— 数据选择页（列出 data/raw 下所有 stream）

从 qt_main.py 提取出来的 StreamSelectorDialog，主题复用 ui/theme.py。
"""

import os

from PySide6 import QtCore, QtWidgets

from ui import theme


class StreamSelectorDialog(QtWidgets.QDialog):
    """数据选择界面：QTreeWidget 列出所有 stream。"""

    def __init__(self, streams):
        super().__init__()
        self.streams = streams
        self.selected = None

        self.setWindowTitle('fNIRS 数据选择')
        self.resize(880, 540)
        self.setStyleSheet(theme.qss())

        lay = QtWidgets.QVBoxLayout(self)

        hint = QtWidgets.QLabel('选择一条 stream 打开 fNIRS 查看器；查看完毕后点 Exit 可返回此处重新选择')
        hint.setStyleSheet(f'color: {theme.color("muted")};')
        lay.addWidget(hint)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['任务', '被试', 'Stream 目录'])
        self.tree.setColumnWidth(0, 60)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 560)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        for i, (task, subj, stream_dir) in enumerate(streams):
            it = QtWidgets.QTreeWidgetItem([task, subj, os.path.basename(stream_dir)])
            it.setData(0, QtCore.Qt.UserRole, i)
            self.tree.addTopLevelItem(it)
        self.tree.itemDoubleClicked.connect(lambda *_: self._open())
        lay.addWidget(self.tree)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_open = QtWidgets.QPushButton('打开查看器')
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open)
        btn_quit = QtWidgets.QPushButton('退出程序')
        btn_quit.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_open)
        btn_row.addWidget(btn_quit)
        lay.addLayout(btn_row)

        self.status = QtWidgets.QLabel(f'共 {len(streams)} 条 stream')
        self.status.setStyleSheet(f'color: {theme.color("muted")};')
        lay.addWidget(self.status)

        self.tree.itemSelectionChanged.connect(self._on_select)

    def _on_select(self):
        self.btn_open.setEnabled(bool(self.tree.selectedItems()))

    def _open(self):
        items = self.tree.selectedItems()
        if not items:
            return
        idx = items[0].data(0, QtCore.Qt.UserRole)
        self.selected = self.streams[idx]
        self.accept()
