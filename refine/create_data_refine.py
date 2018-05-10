import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir1', help='directory to targets')
parser.add_argument('--input_dir2', help='directory to labels')
parser.add_argument('--output_dir', help='directory to outputs')
args = parser.parse_args()

path_out = glob(os.path.join(args.input_dir1,'*'))
path_inp = [os.path.join(args.input_dir2, os.path.split(path)[-1][:-4]+'.png') for path in path_out]

assert len(path_out) == len(path_inp)

def read_image(path, img_name='output1', color=cv2.COLOR_BGR2GRAY):
#     path = os.path.join(path, img_name+'.png')
    assert os.path.exists(path)
    img = cv2.imread(path)
    w = img.shape[1]//2
    if img_name == 'input':
        rv = img[:,:w]
        rv = cv2.cvtColor(rv, color)
    elif img_name == 'output1':
        rv = img[:,-w:, -1]
    elif img_name == 'output2':
        rv = img[:,-w:, 0]
    return rv

def create_line_mask(img_inp, it=1, size=(15,60)):
    mask = np.zeros(img_inp.shape[:2], 'uint8')
    for _ in range(it):
        x1,y1 = np.random.choice(np.max(mask.shape), 2)
        x2, y2 = np.random.choice(np.max(mask.shape), 2)
        intensity = int(255*np.random.uniform(0.5, 1))
        cv2.line(mask,(x1,y1),(y2,x2),intensity,np.random.choice(size[1]-size[0])+size[0])
    return (255-mask)/255


if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    for i, (p_in, p_out) in enumerate(zip(path_inp, path_out)):
        print(p_in, p_out)
        name = os.path.split(p_out)[-1]
        img_inp1 = read_image(p_in, 'input')
        img_inp2 = read_image(p_in, 'output1')
        img_inp3 = read_image(p_in, 'output2')
        img_out1 = read_image(p_out, 'output1')
        img_out2 = read_image(p_out, 'output2')
        img_out3 = read_image(p_out, 'output2')
        for _ in range(50):
            it = np.random.choice(10)+5
            mask=create_line_mask(img_inp1, it=it)
            img_inp2_maskout = (img_inp2*mask).astype('uint8')
            img_inp = np.stack([img_inp1,img_inp2_maskout,img_inp3], axis=-1)
            img_out = np.stack([img_out1,img_out2,img_out3], axis=-1)
            img_combined = np.concatenate([img_inp, img_out], axis=1)
            img_combined = cv2.resize(img_combined, (1024*2, 2*512))
            cv2.imwrite(os.path.join(args.output_dir,'{}_{}.png'.format(name, _)), cv2.cvtColor(img_combined, cv2.COLOR_BGR2RGB))
