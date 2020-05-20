# -*- coding: utf-8 -*-
# @Author  : Su LiHui
# @Time    : 7/28/19 11:20 AM

import os


def check_filters(video_name):
    if video_name.endswith('.mp4') or \
        video_name.endswith('.avi'):
        return True
    else:
        return False

def save_files(file_path, results):
    with open(os.path.join(file_path), 'w') as f:
        for item in results:
            if len(item) < 5:
                print("Warning: results length is less than 5, may be not right.")
            x = int(item[1])
            y = int(item[2])
            w = int(item[3])
            h = int(item[4])
            f.write("%d %d %d %d %d %.2f\n" % (item[0], x, y, w, h, item[-1]))