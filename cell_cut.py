import numpy as np
import argparse
import os
import json
from glob import glob
import random
import collections
import math
import cv2
from skimage import measure
import time
from PIL import Image
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', '-i', required=True, help='path to input image')
parser.add_argument('--output_dir', '-o', required=True, help='path to input image')
parser.add_argument('--resize_ratio', '-r', default=0.3, type=float,
           help='image will be resized to this ratio')
parser.add_argument('--close_h', type=int, default=3)
parser.add_argument('--close_w', type=int, default=3)
args = parser.parse_args()


def read_img(path, p, is_gray=True):
    assert os.path.exists(path), path
    if is_gray:
        img = cv2.imread(path, 0)
    else:
        print('Read image:', path)
        img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), fx=p, fy=p)
    return img


def write_image(path, image):
    image = image.astype('uint8')
    return cv2.imwrite(path, image)


def show(x, method='pil'):
<<<<<<< HEAD
    
=======
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
    if method == 'pil':
        img = Image.fromarray(x)
        img.show()
    else:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(20, 20))
        if len(x.shape) == 3:
            plt.show(plt.imshow(x))
        else:
            plt.show(plt.imshow(x, cmap='gray'))


def get_dot(im):
    ret, thresh = cv2.threshold(im, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rv = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 5*5:
            x, y, w, h = cv2.boundingRect(c)
            rv.append((x+w//2, y+h//2))
    return rv


def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def sort_contours(cnts, method="left-to-right"):
<<<<<<< HEAD
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
=======
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
        
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

<<<<<<< HEAD
    # return the list of sorted contours and bounding boxes
=======
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
    return (cnts, boundingBoxes)


def process(line, dot):

    def noi(line, mode='h'):
        assert mode == 'h' or mode == 'v'
        d_kernel = np.ones([1, 20], np.uint8)
        if mode == 'v':
            d_kernel = np.transpose(d_kernel)
        errode = cv2.erode(line, d_kernel, iterations=1)
        errode = cv2.add(errode, dot)
        kernel = np.ones((1, 10), np.uint8)  # note this is a horizontal kernel
        if mode == 'v':
            kernel = np.transpose(kernel)
        d_im = cv2.dilate(errode, kernel, iterations=1)
        e_im = cv2.erode(d_im, kernel, iterations=1)

        return e_im
    h_im = noi(line, mode='h')
    show(h_im)
    v_im = noi(line, mode='v')
    merge_im = cv2.add(h_im, v_im)
    thresh = cv2.threshold(merge_im, 100, 255, cv2.THRESH_BINARY)[1]
    return thresh


def get_contour(edges):
    labels = measure.label(edges, neighbors=8, background=0)
    

    cnts = []
    for i, label in enumerate(np.unique(labels)):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros([*edges.shape, 3], dtype="uint8")
        lm = np.zeros_like(edges, dtype='uint8')
        labelMask[labels == label] = (255, 0, 0) if i % 2 == 0 else (0, 0, 255)
        lm[labels == label] = 255
        numPixels = cv2.countNonZero(
            cv2.cvtColor(labelMask, cv2.COLOR_RGB2GRAY))
        coords = np.column_stack(np.where(lm > 0))
        coords = np.column_stack([coords[:, 1], coords[:, 0]])
        
        arc = cv2.arcLength(coords, False)
        x, y, w, h = cv2.boundingRect(coords)
        cnts.append(coords)

    cnts = sort_contours(cnts, 'left-to-right')[0]
    cnts = sort_contours(cnts,'top-to-bottom')[0]

    return cnts

def read_inp():
    if os.path.isdir(args.input_dir):
        path_line = os.path.join(args.input_dir, 'refine_line.png')
        path_dot = os.path.join(args.input_dir, 'output2.png')
        path_inp = os.path.join(args.input_dir, 'input.png')
        img_inp = read_img(path_inp, args.resize_ratio, is_gray=False)
        line = read_img(path_line, args.resize_ratio)  
    else:
        img = read_img(args.input_dir, args.resize_ratio, is_gray=False)
        w = img.shape[1]
        img_inp = img[:, :w//2]
        line = img[:, w//2:, -1]
        
    return img_inp, line

def cell_cut():
    img_inp, line = read_inp()
    close = cv2.morphologyEx(line, cv2.MORPH_CLOSE, np.ones([args.close_h, args.close_w]))
    thresh = cv2.threshold(close, 50, 300, cv2.THRESH_BINARY)[-1]
    edges = auto_canny(thresh)
    cnts = get_contour(edges)
    i = 0
<<<<<<< HEAD
    # mask = np.zeros([*thresh.shape, 3], dtype="uint8")  # img_inp.copy()
    mask = img_inp.copy()
    for coords in cnts:
        # shape = sd.detect(coords)
=======
    mask = img_inp.copy()
    for coords in cnts:
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
        x,y,w,h = cv2.boundingRect(coords)
        is_convex = cv2.isContourConvex(coords)
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
<<<<<<< HEAD
        area = cv2.contourArea(coords)
=======
            
        area = cv2.contourArea(coords)
        
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
        if   h < 100 or True:
            rect = cv2.minAreaRect(coords)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            color = (255,0,0) if i % 2 == 0 else (0,255,255)
            cv2.drawContours(mask,[box],0,color,1)
            cX, cY = box.mean(axis=0).astype('int')
<<<<<<< HEAD
            # cv2.putText(mask, "#{}".format(i), (int(cX), int(cY)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)

    return mask


if __name__ == '__main__':
    mask = cell_cut()
    #resize_output = cv2.resize(mask, (0, 0), fx=args.resize_ratio, fy=args.resize_ratio)
=======

    return mask

if __name__ == '__main__':
    mask = cell_cut()
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
    resize_output = cv2.resize(mask, (0, 0), fx=.2, fy=.2)
    if os.path.isdir(args.input_dir):
        output_cell_cut = os.path.join(args.output_dir, 'cellcut_output.png')
    else:
        output_cell_cut = os.path.join(args.output_dir, '{}_cellcut_output.png'.format(os.path.split(args.input_dir)[-1]))
<<<<<<< HEAD
    print('Cell cut output: ', output_cell_cut)
    os.makedirs(os.path.split(output_cell_cut)[0], exist_ok=True)
    cv2.imwrite(output_cell_cut, mask)
    #os.system('open {}'.format(output_cell_cut))
=======
    
    print('Cell cut output: ', output_cell_cut)
    os.makedirs(os.path.split(output_cell_cut)[0], exist_ok=True)
    cv2.imwrite(output_cell_cut, mask)
>>>>>>> c4102be08e8a4d5d873699ee700313401db0b397
