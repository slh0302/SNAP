# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 7/28/19 2:52 PM

import time
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import *

class draw_state():
    draw_done = False
    draw_flag = False


class Runthread(QtCore.QThread):
    _signal = pyqtSignal(str)
    def __init__(self):
        super(Runthread, self).__init__()
    def __del__(self):
        self.wait()
    def run(self):
        while True:
            time.sleep(0.2)
            self._signal.emit()