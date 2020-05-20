# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 7/28/19 11:12 AM

import os
import numpy as np
import cv2
import torch
from model.net import SiamRPNBIG
from model.siamTracker import SiamRPN_init

def IOU(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def check_confidence(res):
    new_res = {}
    for key, value in res.items():
        max_value = 0
        max_coord = []
        if len(value) > 0:
            for i in value:
                cdx = i[-1]
                if max_value < cdx:
                    max_value = cdx
                    max_coord = i[:-1]
                    max_coord[2] = max_coord[2] - max_coord[0]
                    max_coord[3] = max_coord[3] - max_coord[1]
        new_res[key] = max_coord
    return new_res

def calc_distance(list1, list2):
    xc1 = float(list1[0] + list1[2] / 2)
    yc1 = float(list1[1] + list1[3] / 2)

    xc2 = float(list2[0] + list2[2] / 2)
    yc2 = float(list2[1] + list2[3] / 2)

    return np.sqrt((xc1 - xc2) ** 2 + (yc1 - yc2) ** 2)

def judge_track_frame(track_res, detect_res, foot_type):
    """
    Handle track success frame
    :param track_res: [x, y, w, h]
    :param detect_res: {'foot_type' : [x,y,w,h]}
    :param foot_type: foot type in ['front_foot_right', 'front_foot_left','back_foot_right', 'back_foot_left']
    :return: checked frame results
    """
    all_type = ['back_foot', 'front_foot']
    checked_res = {}
    for foot in all_type:
        if foot == foot_type:
            checked_res[foot_type] = track_res
            continue

        if not foot in detect_res.keys():
            checked_res[foot] = []
            continue

        other_type_res = detect_res[foot]
        iou_res = IOU(np.array(track_res), np.array(other_type_res))
        if iou_res[0] > 0.8:
            checked_res[foot] = []
        else:
            checked_res[foot] = other_type_res
    return  checked_res

def judge_detect_frame(before_res, detect_res, foot_type):
    """
    Handle track failed frame
    :param before_res: T-1 frame results
    :param detect_res: T frame results
    :param foot_type: target type
    :return: [x,y,w,h] for target type
    """
    checked_res = []
    foot_res = []
    if foot_type in detect_res.keys():
        foot_res = detect_res[foot_type]
        checked_res = foot_res
    else:
        if foot_type in before_res.keys():
            return before_res[foot_type]
        else:
            return [-1, -1, -1, -1]
    for foot in before_res.keys():
        if foot != foot_type and len(before_res[foot]) > 0:
            iou_res = IOU(np.array(before_res[foot]), np.array(foot_res))
            if iou_res[0] > 0.8:
               checked_res = [-1, -1, -1, -1]
            else:
               checked_res = foot_res
    return  checked_res

def judge_track_res(track_res, detect_res, foot_type):
    """
    Handle track failed frame
    :param track_res: T frame track results
    :param detect_res: T frame results of foot type
    :param foot_type: target type
    :return: [x,y,w,h] for target type
    """
    checked_res = track_res
    foot_res = []
    if foot_type in detect_res.keys():
        foot_res = detect_res[foot_type]
    else:
        return track_res

    iou_foot = IOU(np.array(track_res), np.array(foot_res))

    for foot in detect_res.keys():
        if foot != foot_type and len(detect_res[foot]) > 0:
            iou_res = IOU(np.array(detect_res[foot]), np.array(track_res))
            if iou_res[0] > 0.5 and iou_foot[0] < 0.3:
               checked_res = foot_res
            else:
               checked_res = track_res
    return  checked_res

def save_results(results, video_name, save_path, label_type=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if label_type:
        save_file = video_name.split('.')[0] + "_" + label_type + ".txt"
    else:
        save_file = video_name.split('.')[0] + ".txt"
    # tmp_file = os.path.join(save_path, save_file)
    # if os.path.exists(tmp_file):
    #     with open(tmp_file, 'rb') as f:
    #         offs = -100
    #         last = ""
    #         while True:
    #             f.seek(offs, 2)
    #             lines = f.readlines()
    #             if len(lines) > 1:
    #                 last = lines[-1]
    #                 break
    #             offs *= 2
    #         frame_id = int(last.split()[0])
    #         results = results.copy()
    #         results = results[frame_id: ]

    with open(os.path.join(save_path, save_file), 'w') as f:
        for item in results:
            if len(item) < 5:
                print("Warning: results length is less than 5, may be not right.")
            x = int(item[1])
            y = int(item[2])
            w = int(item[3])
            h = int(item[4])
            f.write("%d %d %d %d %d %.2f\n" % (item[0], x, y, w, h, item[-1]))



def rect_box(img, frame_box, scores=None, frame_id=None):
    img = np.copy(img)
    if scores != None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, '%d : %.2f' % (frame_id, scores), (25, 25), font, 1, (0, 0, 255), 2, False)
    x = int(frame_box[0])
    y = int(frame_box[1])
    w = int(frame_box[2])
    h = int(frame_box[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return img

def save_pictures(img, pic_path, frame_id, frame_box):
    if not os.path.exists(pic_path):
        print("No path exist: " + pic_path)
        return False
    frame_name = "%09d.jpg" % frame_id
    img = rect_box(img, frame_box)
    save_pic = os.path.join(pic_path, frame_name)
    cv2.imwrite(save_pic, img)
    return True

def init_net(gpus=0, rel_path=__file__, cuda_state=True):
    # network start
    net_file = os.path.join(rel_path, 'siam.model')
    net = SiamRPNBIG()
    net.load_state_dict(torch.load(net_file,map_location={'cuda:0': 'cpu'}))
    if cuda_state:
        net.eval().cuda(gpus)
        # warm up
        for i in range(10):
            net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
            net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())

    else:
        net.eval()
        for i in range(10):
            net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)))
            net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)))


    return net

def init_track(net, img, init_bbox, cuda_state=True):
    res_dict = {}
    res_dict['res'] = [init_bbox]
    target_pos = np.array([init_bbox[0] + 0.5 * init_bbox[2], init_bbox[1] + 0.5 * init_bbox[3]])
    target_sz = np.array(init_bbox[2:])
    print(target_pos)
    print(target_sz)
    state = SiamRPN_init(img, target_pos, target_sz, net, cuda_state)
    return state

class drawBox:
    def __init__(self, window_name, img):
        self.start_point = (-1, -1)
        self.end_point = (-1, -1)
        self.img  = img
        self.backup_img = np.copy(img)
        self.drawing = False
        self.finish = False
        self.draw_begin = False
        width = img.shape[1]
        height = img.shape[0]
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, width, height)
        cv2.setMouseCallback(window_name, self.mouse_event)

    def mouse_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.draw_begin = True
            self.start_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if not (abs(x - self.start_point[0]) < 10 and abs(y - self.start_point[1]) < 10):
                cv2.rectangle(self.img, self.start_point, (x, y), (0, 255, 0), 1)
                self.finish = True
                self.final_point = [self.start_point[0], self.start_point[1],
                                    self.end_point[0] - self.start_point[0],
                                    self.end_point[1] - self.start_point[1]]

            self.end_point = self.start_point = (-1, -1)
