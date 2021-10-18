# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 7/28/19 11:11 AM

import os
import cv2
import sys
sys.path.append("..\\")
import torch
import time
import threading
from ui.ui_main import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QApplication, QDialog
from PyQt5.QtGui import QPixmap, QImage, QPixmap
from PyQt5.QtCore import QCoreApplication, QBasicTimer, Qt
from PyQt5.QtPrintSupport import QPrinter, QPrintDialog
from model.tracker_tools import init_net, init_track, rect_box
from model.siamTracker import SiamRPN_track
# from model.train_detector import train_size
from model.utils import cxy_wh_2_rect, rect_2_cxy_wh
from ui_func.fileProcess import check_filters, save_files
from ui.draw_ctrl import draw_state
from ui.train_dialog import Ui_Dialog
__prefix_path__, _ = os.path.split(os.path.realpath(__file__))



class Tracker_window(QtWidgets.QMainWindow):
    def __init__(self):
        super(Tracker_window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.cwd = os.getcwd()
        self.scene = None
        self.printer = None
        self.draw_w = 0
        self.draw_h = 0
        self._func_bind()
        self.cv_cap_loader = None
        self.video_path = ""
        self.save_path = self.cwd
        self.save_name = 'results.txt'
        self.tacker_status = False
        self.cuda_state = torch.cuda.is_available()
        self.tracker_net = init_net(gpus=0,
                                    rel_path=os.path.join(__prefix_path__, '../checkpoint/'),
                                    cuda_state= self.cuda_state)
        self.total_frames = 0
        self.before_img = None
        self.cur_img = None
        self.cur_frames = 0
        """ reinit status """
        self.skip_frame = 0
        self.before_frame = 0
        self.move_value = 1
        self.reinit_status = False
        """ reinit status """
        self.frameRate = 0
        self.tracker_res = []
        self.thread_status = []
        self.img_scale_ratio = [] # w, h
        self.frame_state = []
        self.stopEvent = threading.Event()
        self.stopEvent.clear()
        self.timer = QBasicTimer()
        self.pause_status = False
        self.timer.start(1000, self)

        # train detector model
        self.train_dialog_ui = Ui_Dialog()
        self.train_dialog = QDialog()
        self.train_dialog_ui.setupUi(self.train_dialog)
        self.model_name = ""
        self.LabelFilePath = "./"
        self.outModelPath = "./"
        self.detect_status = False
        self._func_bind_train_dialog()


    def _func_bind(self):
        self.ui.progressBar.setEnabled(False)
        self.ui.displayLabel.setScaledContents(True)
        # self.ui.plainTextEdit.setEnabled(False)
        self._button_status_change(False)
        self.ui.actionOpen_video.triggered.connect(self.open_event)
        self.ui.actionSave_video.triggered.connect(self.save_event)
        self.ui.actionTrain_Faster_RCNN.triggered.connect(self.showTrainDialog)
        self.ui.redrawPushButton.clicked.connect(self._redraw_button)
        self.ui.startPushButton.clicked.connect(self.start_tracking)
        self.ui.pausPushButton.clicked.connect(self.pause_tracking)
        self.ui.stoPushButton.clicked.connect(self.stop_tracking)
        self.ui.chooseModelButton.clicked.connect(self.open_model_event)
        self.ui.nextPushButton.clicked.connect(self.next_tracking)
        self.ui.prePushButton.clicked.connect(self.prev_tracking)
        self.ui.FramesSpinBox.valueChanged.connect(self.spinValueChange)


    def _func_bind_train_dialog(self):
        self.train_dialog_ui.pushButtonOK.clicked.connect(self.train_ok)
        self.train_dialog_ui.pushButtonCancel.clicked.connect(self.train_cancel)
        self.train_dialog_ui.pushButtonLabelPath.clicked.connect(self.trainDialog_openLabel)
        self.train_dialog_ui.pushButtonOutModel.clicked.connect(self.trainDialog_outputPath)


    def _button_status_change(self, status, need_detect=False):
        self.ui.startPushButton.setEnabled(status)
        self.ui.pausPushButton.setEnabled(status)
        self.ui.redrawPushButton.setEnabled(status)
        self.ui.stoPushButton.setEnabled(status)
        self.ui.prePushButton.setEnabled(status)
        self.ui.nextPushButton.setEnabled(status)
        # self.ui.chooseModelButton.setEnabled(status)
        self.ui.FramesSpinBox.setEnabled(status)
        if not need_detect:
            self.ui.turnDetectorButton.setEnabled(status)


    def _write_logs_to_plain_text(self, text):
        self.ui.textBrowser.append(text)


    def _redraw_button(self):
        self.ui.displayLabel.setdrawDoneStatus(self.cur_img.copy(), self.img_scale_ratio)
        if self.cur_frames != 0:
            self.reinit_status = True
            self.ui.startPushButton.setText("re-start")
            self.ui.startPushButton.setEnabled(True)


    def _coord_to_text(self, x, y):
        return "X: %d, Y: %d" % (int(x),int(y))


    def _fresh_progresssBar(self):
        if self.total_frames != 0:
            self.ui.progressBar.setValue(int(self.cur_frames / self.total_frames * 100))


    def _set_img_ratio(self, img_wh):
        img_w = img_wh[1]
        img_h = img_wh[0]
        self.draw_h = self.ui.displayLabel.size().height()
        self.draw_w = self.ui.displayLabel.size().width()
        self.img_scale_ratio = [1.0 * img_w / self.draw_w,
                                1.0 * img_h / self.draw_h ]
        self._write_logs_to_plain_text("Img scale w: %.2f h: %.2f" % tuple(self.img_scale_ratio))


    def open_event(self):
        fileName_, f_type = QFileDialog.getOpenFileName(self,
                                                        "Open videos",
                                                        self.cwd,
                                                        'All Files (*);;Mp4 (*.mp4);;Avi (*.avi)')
        if fileName_ == "":
            print("no files")
            return

        if check_filters(fileName_):
            self.video_path = fileName_
            self.init_trackers()
            print(fileName_, f_type)
        else:
            print(fileName_, ' is wrong!')
            return


    def open_model_event(self):
        fileName_, f_type = QFileDialog.getOpenFileName(self,
                                                        "Open models",
                                                        self.cwd,
                                                        'All Files (*);;pth (*.pth);;Avi (*.model)')
        if fileName_ == "":
            print("no files")
            return

        if os.path.getsize(fileName_) > 100:
            self.model_name = fileName_
            if self.load_model(self.model_name):
                self.ui.turnDetectorButton.setEnabled(True)

            print(fileName_, f_type)
        else:
            print(fileName_, ' is wrong!')
            return


    def load_model(self, filename):
        if os.path.exists(filename):
            # try:
            #     # with open(filename, 'rb') as f:
            #
            # except:

            return True
        else:
            return False


    def save_event(self):
        fileName_, f_type = QFileDialog.getSaveFileName(self,
                                                        "Save videos",
                                                        self.cwd,
                                                        initialFilter='*.txt')
        if fileName_ == "":
            print('use default files')
            out_res_path = os.path.join(self.save_path, self.save_name)
        else:
            self.save_path = os.path.dirname(fileName_)
            self.save_name = os.path.basename(fileName_)
            out_res_path = os.path.join(self.save_path, self.save_name)
            self.LabelFilePath = out_res_path
            print('new path: ', out_res_path)

        if len(self.tracker_res) != 0:
            t1 = threading.Thread(target=save_files, args=(out_res_path, self.tracker_res))
            t1.start()
        else:
            print(fileName_, ' can\'t save, because the empty res')

        print(fileName_)


    def reinit_tracker(self):
        self.before_img = None
        self.cur_img = None
        self.cur_frames = 0
        self.frameRate = 0
        self.skip_frame = 0
        self.before_frame = 0
        # self.tracker_res = []


    def init_trackers(self):
        if self.total_frames != 0 and self.cv_cap_loader.isOpened():
            self.cv_cap_loader.release()
            self.cur_frames = 0

        if self.cur_frames != 0:
            self.cur_frames = 0

        self.tracker_res = []

        self.cv_cap_loader = cv2.VideoCapture(self.video_path)
        self.total_frames = self.cv_cap_loader.get(7)
        self.frameRate = self.cv_cap_loader.get(cv2.CAP_PROP_FPS)
        if self.cv_cap_loader.isOpened():
            ret, frame = self.cv_cap_loader.read()
            if ret:
                # self.scene.clear()
                self.cur_img = frame.copy()
                height, width, bytesPerComponent = frame.shape
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
                bytesPerLine = 3 * width
                QImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pix = QPixmap.fromImage(QImg)
                self.ui.displayLabel.setPixmap(pix)
                self._write_logs_to_plain_text('Read Video done.')
                self.ui.startPushButton.setEnabled(True)
                self.ui.progressBar.setEnabled(True)
                self.ui.redrawPushButton.setEnabled(True)
                self._fresh_progresssBar()
                self._set_img_ratio(frame.shape)
                self.ui.displayLabel.setNewFrames()
                self.ui.displayLabel.setdrawDoneStatus(self.cur_img.copy(), self.img_scale_ratio)


    def start_tracking(self):

        if self.stopEvent.is_set():
            self.stopEvent.clear()

        if self.ui.startPushButton.text() != 'Start':
            if self.before_frame != 0:
                self.cur_frames = self.before_frame

            self.tracker_res = self.tracker_res[:self.cur_frames + 1]
            self.ui.progressBar.setValue(int(self.cur_frames / self.total_frames * 100))
            if draw_state.draw_done:
                x0, y0, x1, y1 = self.ui.displayLabel.return_x_y()
            else:
                x0, y0, x1, y1 = self.tracker_res[self.cur_frames][1:-1]

            self.tracker_res.pop()
            self._write_logs_to_plain_text("re-start at: " + self._coord_to_text(x0, y0))
            self._write_logs_to_plain_text("re-start at: " + self._coord_to_text(x0, y0))
            ori_x, ori_x1 = x0 * self.img_scale_ratio[0], x1 * self.img_scale_ratio[0]
            ori_y, ori_y1 = y0 * self.img_scale_ratio[1], y1 * self.img_scale_ratio[1]
            frame_bbox = [ori_x, ori_y, ori_x1 - ori_x, ori_y1 - ori_y]
            self.tracker_res.append([self.cur_frames + 1] + frame_bbox + [2.0])
            self.frame_state = init_track(self.tracker_net, self.cur_img, frame_bbox)
            self.ui.startPushButton.setText("Start")
            self.ui.startPushButton.setEnabled(False)
            self.ui.pausPushButton.click()
        else:
            if draw_state.draw_done:
                self.ui.progressBar.setValue(0)
                x0, y0, x1, y1 = self.ui.displayLabel.return_x_y()
                self._write_logs_to_plain_text(self._coord_to_text(x0, y0))
                self._write_logs_to_plain_text(self._coord_to_text(x1, y1))
                ori_x, ori_x1 = x0 * self.img_scale_ratio[0], x1*  self.img_scale_ratio[0]
                ori_y, ori_y1 = y0 * self.img_scale_ratio[1], y1*  self.img_scale_ratio[1]
                frame_bbox = [ori_x, ori_y, ori_x1 - ori_x, ori_y1 - ori_y]
                self.tracker_res.append([self.cur_frames + 1] + frame_bbox + [2.0])
                self.frame_state = init_track(self.tracker_net, self.cur_img, frame_bbox, cuda_state=self.cuda_state)
                self.ui.startPushButton.setEnabled(False)
                self.ui.pausPushButton.setEnabled(True)
                self.ui.redrawPushButton.setEnabled(False)
                self.ui.stoPushButton.setEnabled(True)
                th = threading.Thread(target=self.tracking)
                th.start()
            else:
                self._write_logs_to_plain_text("Bbox undone")


    def tracking(self):
        while self.cv_cap_loader.isOpened():
            if True == self.stopEvent.is_set():
                self.stopEvent.clear()
                break
            success, frame = self.cv_cap_loader.read()
            if success:
                self.cur_frames += 1
                self.before_img = self.cur_frames
                self.cur_img = frame.copy()
                self.frame_state = SiamRPN_track(self.frame_state, frame, cuda_state=self.cuda_state)
                res = cxy_wh_2_rect(self.frame_state['target_pos'], self.frame_state['target_sz'])
                self.tracker_res.append([self.cur_frames + 1] + list(res) + [self.frame_state['score']])
                if self.cur_frames % 2 == 0:
                    # TODO: skip frame changes
                    height, width, bytesPerComponent = frame.shape
                    frame = rect_box(frame, res, self.frame_state['score'], self.cur_frames + 1)
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
                    bytesPerLine = 3 * width
                    QImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(QImg)
                    self.ui.displayLabel.setPixmap(pix)
            else:
                self.cv_cap_loader.release()
                self.ui.stoPushButton.click()
                # self._button_status_change(False)
                break

            if self.stopEvent.is_set() == False:
                time.sleep(0.02)


    def call_backlog(self, msg):
        self.pbar.setValue(int(msg))


    def timerEvent(self, event):
        if self.total_frames != 0:
            self._fresh_progresssBar()
            if not self.cv_cap_loader.isOpened():
                self.cur_frames = 0


    def pause_tracking(self):
        if self.ui.pausPushButton.text() == 'Pause':
            self.stopEvent.set()
            time.sleep(0.5)
            self.pause_status = True
            """ draw res """
            frame = self.cur_img.copy()
            res = self.tracker_res[self.cur_frames]
            height, width, bytesPerComponent = frame.shape
            frame = rect_box(frame, res[1:-1], res[-1], self.cur_frames + 1)
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
            bytesPerLine = 3 * width
            QImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pix = QPixmap.fromImage(QImg)
            self.ui.displayLabel.setPixmap(pix)
            """ end draw res """
            self.ui.pausPushButton.setText('Continue')
            self.ui.prePushButton.setEnabled(True)
            self.ui.nextPushButton.setEnabled(True)
            self.ui.FramesSpinBox.setEnabled(True)
            self.ui.redrawPushButton.setEnabled(True)
        else:
            self.pause_status = False
            self.ui.pausPushButton.setText('Pause')
            self.ui.FramesSpinBox.setEnabled(False)
            self.ui.prePushButton.setEnabled(False)
            self.ui.nextPushButton.setEnabled(False)
            self.ui.redrawPushButton.setEnabled(False)
            self.skip_frame = 0
            self.before_frame = 0
            self.cv_cap_loader.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frames + 1)
            self.stopEvent.clear()
            th = threading.Thread(target=self.tracking)
            th.start()


    def next_tracking(self):
        frame_skip = self.move_value
        if frame_skip <= 0:
            frame_skip = 1

        if self.skip_frame == 0:
            self.skip_frame = self.cur_frames
        else:
            self.skip_frame = self.before_frame

        if self.skip_frame + frame_skip < self.total_frames:
            self.cv_cap_loader.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frame + frame_skip)
            _, tmp_frame = self.cv_cap_loader.read()
            self.cur_img = tmp_frame.copy()
            res_id = self.skip_frame + frame_skip
            if res_id < self.cur_frames:
                height, width, bytesPerComponent = tmp_frame.shape
                frame = rect_box(tmp_frame, self.tracker_res[res_id][1:-1],
                                 self.tracker_res[res_id][-1], res_id + 1)
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
                bytesPerLine = 3 * width
                QImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pix = QPixmap.fromImage(QImg)
                self.ui.displayLabel.setPixmap(pix)
                self.before_frame = self.skip_frame + frame_skip
            else:
                print("undone!")
        else:
            print(" frames can't over the total videos !")


    def prev_tracking(self):
        frame_skip = self.move_value
        if frame_skip <= 0:
            frame_skip = 1

        if self.skip_frame == 0:
            self.skip_frame = self.cur_frames
        else:
            self.skip_frame = self.before_frame

        if self.skip_frame - frame_skip >= 0:
            self.cv_cap_loader.set(cv2.CAP_PROP_POS_FRAMES, self.skip_frame - frame_skip)
            _, tmp_frame = self.cv_cap_loader.read()
            self.cur_img = tmp_frame.copy()
            res_id = self.skip_frame - frame_skip
            if res_id >= 0:
                height, width, bytesPerComponent = tmp_frame.shape
                frame = rect_box(tmp_frame, self.tracker_res[res_id][1:-1],
                                 self.tracker_res[res_id][-1], res_id + 1)
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
                bytesPerLine = 3 * width
                QImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pix = QPixmap.fromImage(QImg)
                self.ui.displayLabel.setPixmap(pix)
                self.before_frame = self.skip_frame - frame_skip
        else:
            print(" frames can't less the zero !")


    def spinValueChange(self):
        self.move_value = self.ui.FramesSpinBox.value()


    def stop_tracking(self):
        if self.ui.pausPushButton.text() == 'Continue':
            self.ui.pausPushButton.setText('Pause')

        if self.cv_cap_loader.isOpened():
            self.stopEvent.set()
            self.reinit_tracker()
            self.cv_cap_loader.release()
        else:
            self.reinit_tracker()
            self._write_logs_to_plain_text("Done all videos !!")
        self._button_status_change(False, need_detect=self.detect_status)



    """ train dialog"""
    def showTrainDialog(self):
        self.train_dialog.setWindowModality(Qt.ApplicationModal)
        self.train_dialog_ui.lineEditLabelPath.setText(self.LabelFilePath)
        self.train_dialog.exec_()


    def trainDialog_openLabel(self):
        fileName_, f_type = QFileDialog.getOpenFileName(self,
                                                        "Open Labels",
                                                        self.cwd,
                                                        'All Files (*);;TXT (*.txt)')
        if fileName_ == "":
            print("no files")
            return

        if ".txt" in fileName_:
            self.LabelFilePath = fileName_
            # self.init_trackers()
            print(fileName_, f_type)
        else:
            print(fileName_, ' is wrong!')
            return


    def trainDialog_outputPath(self):
        fileName_, f_type = QFileDialog.getOpenFileName(self,
                                                        "Out Model",
                                                        self.cwd,
                                                        'All Files (*);;PTH (*.pth)')
        if fileName_ == "":
            print("no files")
            return

        if ".txt" in fileName_:
            self.outModelPath = fileName_
            # self.init_trackers()
            print(fileName_, f_type)
        else:
            print(fileName_, ' is wrong!')
            return


    def train_ok(self):
        self.train_dialog.close()


    def train_cancel(self):
        self.train_dialog.close()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    app = QtWidgets.QApplication(sys.argv)
    myshow = Tracker_window()
    myshow.show()
    sys.exit(app.exec_())
