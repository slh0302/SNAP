# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 7/28/19 11:11 AM


import os
import cv2
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from lib.model.nms.nms_wrapper import nms
from lib.model.utils.config import cfg, cfg_from_file
from lib.model.faster_rcnn.resnet import resnet

__detector_prefix_path__, _ = os.path.split(os.path.realpath(__file__))

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

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

class faster_rcnn:
    def __init__(self, base_batch, class_agnostic, base_net=50, cfg_file=__detector_prefix_path__ + '/model/yml/res50.yml',
                 load_name='', gpu_id=0):
        np.random.seed()
        self.batch_size = base_batch
        self.class_agnostic = class_agnostic
        cfg_from_file(cfg_file)

        self.pascal_classes = np.asarray(['__background__',  # always index 0
                                     'front_foot','back_foot'])
        self.fasterRCNN = resnet(self.pascal_classes, base_net, pretrained=False,
                           class_agnostic=False)
        self.fasterRCNN.create_architecture()
        self.fasterRCNN.cuda()
        self.fasterRCNN.eval()

        # load check point
        print("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']

    def detect_img(self, img, gpus=0):
        """
        :param img: numpy array
        :return:
        """
        im_in = img
        im = im_in[:, :, ::-1]

        blobs, im_scales = self._get_image_blob(im)
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)


        # output
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if self.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(self.pascal_classes))

            pred_boxes = self.bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = self.clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        all_res = {}
        thresh = 0.05
        for j in range(1, len(self.pascal_classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]

                res = self.fetch_dets(self.pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
                all_res = dict(all_res, **res)

        return all_res

    def detect_seq(self, seq):
        pass

    def bbox_transform_inv(self, boxes, deltas, batch_size):
        widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0::4]
        dy = deltas[:, :, 1::4]
        dw = deltas[:, :, 2::4]
        dh = deltas[:, :, 3::4]

        pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
        pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
        pred_w = torch.exp(dw) * widths.unsqueeze(2)
        pred_h = torch.exp(dh) * heights.unsqueeze(2)

        pred_boxes = deltas.clone()
        # x1
        pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def fetch_dets(self, class_name, dets, thresh=0.8, ismax=False):
        """Visual debugging of detections."""
        res = {}
        if not ismax:
            for i in range(np.minimum(10, dets.shape[0])):
                bbox = tuple(int(np.round(x)) for x in dets[i, :4])
                score = dets[i, -1]
                if score > thresh:
                    if not class_name in res.keys():
                        res[class_name] = [list(bbox[0:4]) + [score]]
                    else:
                        res[class_name].append(list(bbox[0:4]) + [score])
        else:
            if dets.shape[0] > 0:
                bbox = tuple(int(np.round(x)) for x in dets[0, :4])
                score = dets[0, -1]
                if score > thresh:
                    if not class_name in res.keys():
                        res[class_name] = [list(bbox[0:4]) + [score]]
        return res

    def clip_boxes(self, boxes, im_shape, batch_size):

        for i in range(batch_size):
            boxes[i, :, 0::4].clamp_(0, im_shape[i, 1] - 1)
            boxes[i, :, 1::4].clamp_(0, im_shape[i, 0] - 1)
            boxes[i, :, 2::4].clamp_(0, im_shape[i, 1] - 1)
            boxes[i, :, 3::4].clamp_(0, im_shape[i, 0] - 1)

        return boxes


    def nms_gpu(dets, thresh):
        keep = dets.new(dets.size(0), 1).zero_().int()
        num_out = dets.new(1).zero_().int()
        nms.nms_cuda(keep, dets, num_out, thresh)
        keep = keep[:num_out[0]]
        return keep

    def _get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
          im (ndarray): a color image in BGR order
        Returns:
          blob (ndarray): a data blob holding an image pyramid
          im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
                im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self.im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def im_list_to_blob(self, ims):
        """Convert a list of images into a network input.

        Assumes images are already prepared (means subtracted, BGR order, ...).
        """
        max_shape = np.array([im.shape for im in ims]).max(axis=0)
        num_images = len(ims)
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
        for i in range(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        return blob