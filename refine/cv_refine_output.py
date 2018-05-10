import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from time import time
from imutils import contours
from skimage import measure
import imutils
import cv2
import argparse
import os
import shutil
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True,help="path to folder containing images")
parser.add_argument("--output_dir", required=True,help="path to folder containing images")

args = parser.parse_args()



def get_contour(thresh, min_numPixels=400):
    assert len(thresh.shape) == 2, len(thresh.shape)
    cnts = []
    labels = measure.label(thresh, neighbors=8, background=0)
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > min_numPixels:
            cnt = np.column_stack(np.where(labels == label))
            cnt = np.flip(cnt, 1)
            cnts.append(cnt)
    return cnts

def process_output2(img, thresh=50, k_size=10, r=10):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    thresh = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones([k_size, k_size])
    op = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cnts = get_contour(op, 1)
    mask = np.zeros_like(img, 'uint8')#cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in cnts:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx, cy = np.mean(cnt, 0).astype('int')
            cv2.circle(mask,(cx, cy), r, (255, 0, 0), -1)
    png = np.zeros([*thresh.shape, 4])
    png[...,-1] = mask
    return png
def process_output1(img, thresh=.6, k_size=3):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    blur = cv2.blur(img, 3)
    thresh = cv2.threshold(blur, int(255*thresh), 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones([k_size, k_size])
    op = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    png = np.zeros([*thresh.shape, 4])
    png[...,-1] = op
    return png[...,-1]


if __name__ == '__main__':
    paths = glob(os.path.join(args.input_dir, '*'))
    os.makedirs(args.output_dir, exist_ok=True)
    for path in paths:
        print(path)
        assert os.path.exists(os.path.join(path, 'output1.png')), os.path.join(path, 'output1.png')
        output1 = cv2.imread(os.path.join(path, 'output1.png'))
        output2 = cv2.imread(os.path.join(path, 'output2.png'))
                 
        processed1 = process_output1(output1)
        processed2 = process_output2(output2)
        
        name = os.path.split(path)[-1]
        new_path = os.path.join(args.output_dir, name)
        os.makedirs(new_path, exist_ok=True)
        cv2.imwrite(os.path.join(new_path, 'output1.png'), processed1)
        cv2.imwrite(os.path.join(new_path, 'output2.png'), processed2)
        shutil.copy(os.path.join(path, 'input.png'),os.path.join(new_path, 'input.png'))
        