# -*- coding: utf-8 -*-
"""
main.py —— fNIRS 查看器入口（PySide6 版）

扫描 data/raw → 数据选择界面 → 查看器，查看器关闭后返回选择界面循环。
等价于旧版 qt_main.py 的 main() + StreamSelector。
"""

import os
import sys
import traceback

from PySide6 import QtCore, QtWidgets

from core import data as core
from ui.selector import StreamSelectorDialog
from ui.viewer import ViewerWindow


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')

    # data 目录已移入 fNIRSViewer 内部
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_raw_dir = os.path.join(_BASE_DIR, 'data', 'raw')

    print('扫描 data/raw ...', flush=True)
    streams = core.discover_streams(data_raw_dir)
    if not streams:
        print('未找到任何含 data.csv 的 stream')
        return

    while True:
        sel = StreamSelectorDialog(streams)
        if sel.exec() != QtWidgets.QDialog.Accepted:
            print('已退出程序')
            break

        task, subj, stream_dir = sel.selected
        print(f'打开查看器: {task}/{subj}/{os.path.basename(stream_dir)}', flush=True)
        try:
            data = core.FNIRSData(stream_dir)
        except Exception:
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(None, '加载失败', '加载失败，返回选择界面')
            continue

        win = ViewerWindow(data, stream_dir)
        # QMainWindow 无 exec()：show 后进入局部事件循环，关闭即销毁并返回选择界面
        win.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
        win.show()
        loop = QtCore.QEventLoop()
        win.destroyed.connect(loop.quit)
        loop.exec()


if __name__ == '__main__':
    main()
