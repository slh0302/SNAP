# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 7/28/19 1:55 PM

import cv2
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap
from ui.draw_ctrl import draw_state

class drawLabel(QLabel):
    def __init__(self, parent):
        super(drawLabel, self).__init__(parent)
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0
        self.draw_flag = False
        self.draw_done = True
        self.cur_img = None
        self.img_scale = []
        self.x1_finals, self.y1_finals = 0, 0

    def return_x_y(self):
        return self.x0, self.y0, self.x1_finals, self.y1_finals

    def setdrawDoneStatus(self, img=None, img_scale=None):
        if self.draw_done == True:
            if len(self.img_scale) > 0:
                self._redraw_img(img.copy())
            self.draw_done = False
            draw_state.draw_done = False
            self.cur_img = img.copy()
            self.img_scale = img_scale

    def setNewFrames(self):
        self.draw_flag = False
        self.draw_done = True
        draw_state.draw_done = False

    def mousePressEvent(self, event):
        if self.draw_done == False:
            self.draw_flag = True
            self.x0 = event.x()
            self.y0 = event.y()
            print(self._coord_to_text())

    def mouseReleaseEvent(self,event):
        if self.draw_done == False:
            self.draw_flag = False
            self.draw_done = True
            draw_state.draw_done = True
            x1, y1 = self.x1, self.y1
            self.x1, self.y1 = self.x0, self.y0
            self.repaint()
            self.x1_finals, self.y1_finals = x1, y1
            frame = self.cur_img.copy()
            ori_x, ori_x1 = self.x0 * self.img_scale[0], self.x1_finals * self.img_scale[0]
            ori_y, ori_y1 = self.y0 * self.img_scale[1], self.y1_finals * self.img_scale[1]
            cv2.rectangle(frame, (int(ori_x), int(ori_y)), (int(ori_x1), int(ori_y1)), (0, 0, 255), 4)
            self._redraw_img(frame.copy())

    def _redraw_img(self, frame):
        self.clear()
        height, width, bytesPerComponent = frame.shape
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
        bytesPerLine = 3 * width
        QImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pix = QPixmap.fromImage(QImg)
        self.setPixmap(pix)
        self.update()

    def mouseMoveEvent(self,event):
        if self.draw_flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()

    def _coord_to_text(self):
        return "X: %d, Y: %d" % (int(self.x0), int(self.y0))

    def paintEvent(self, event):
        super().paintEvent(event)
        rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 4, Qt.SolidLine))
        painter.drawRect(rect)
        # painter.eraseRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
