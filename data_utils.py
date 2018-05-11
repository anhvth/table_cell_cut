
import os
import Augmentor
import numpy as np
import cv2
from psd_tools import PSDImage
from glob import glob
import argparse
from glob import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--mode", default="combine",
                    choices=["combine", "crop"])
parser.add_argument('--input_dir', required=True,
                    help='Path to input dir')
parser.add_argument('--output_dir', required=True, help='Path to output dir')
parser.add_argument('--crop_size', type=int, help='Path to output dir')
parser.add_argument('--psd_type', type=str, default='raw',
                    help='Path to output dir')

args = parser.parse_args()


def psd_combine(path, save_path_a=None, save_path_b=None, input_idx=2, line_idx=1, dot_idx=0):
    psd = PSDImage.load(path)
    print(path)
    assert len(psd.layers) == 3, 'num of len should be 3 but found {}'.format(
        len(psd.layers)) + path

    input = psd2gray(psd, input_idx, input_layer=True)
    line = psd2gray(psd, line_idx)
    print('line: ', line.mean())
    dot = psd2gray(psd, dot_idx)
    print('dot: ', dot.mean())
    output = np.stack([line, dot, dot], axis=2)
    combine = np.concatenate([input, output], axis=1)
    if save_path_a and save_path_b is not None:
        print('------------------------')
        print('Write {} : {}'.format(save_path_a, combine.shape))
        print('Write {} : {}'.format(save_path_b, combine.shape))
        print('------------------------')
        write_image(save_path_a, input)
        write_image(save_path_b, output)
    return combine


def psd2gray(psd, layer_idx, input_layer=False):
    '''

    '''
    mask = np.zeros([psd.header.height, psd.header.width], dtype='uint8') if not input_layer else np.zeros(
        [psd.header.height, psd.header.width, 3], dtype='uint8')
    layer = np.array(psd.layers[layer_idx].as_PIL())
    print(layer.shape)
    if args.psd_type == 'raw':
        img = layer[..., -
            1] if not input_layer else cv2.cvtColor(layer, cv2.COLOR_BGRA2BGR)
    else:
        img = cv2.cvtColor(layer, cv2.COLOR_BGRA2GRAY) if not input_layer else cv2.cvtColor(
            layer, cv2.COLOR_BGRA2BGR)

    x1, y1, x2, y2 = [i for i in psd.layers[layer_idx].bbox]
    h, w = mask[y1:y2, x1:x2].shape[:2]
    mask[y1:y2, x1:x2] = cv2.resize(img, (w, h))
    return mask


def load_image(path, gray=False):
    '''
        return RGB or gray image
    '''
#     print(path, gray)

    if gray:
        img = cv2.imread(path, 0)

    else:
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    return img


def write_image(path, image):
    '''
        Input: rgb image
    '''
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, image)


def show(x):
    plt.figure(figsize=(20, 20))
    plt.show(plt.imshow(x))


def normalize(im):
    _ = np.array(im > 250, dtype='uint8')
    _ *= 255
    return _


def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name


def split(img):
    w = img.shape[1]
    return img[:, :w//2], normalize(img[:, w//2:])


def random_crop(a, b, CROP_SIZE):
    h, w = a.shape[:2]
    i1 = np.random.choice(h-CROP_SIZE)
    i2 = np.random.choice(w-CROP_SIZE)
    a_ = a[i1:i1+CROP_SIZE, i2:i2+CROP_SIZE]
    b_ = b[i1:i1+CROP_SIZE, i2:i2+CROP_SIZE]
    return a_, b_  # np.concatenate([a_, b_], axis=1)


if __name__ == '__main__':
    ext = '*.psd' if args.mode == 'combine' else '*.png'
    rg_paths = os.path.join(args.input_dir, ext)
    print('Glob({})'.format(rg_paths))
    paths = glob(rg_paths)
    if args.mode == 'combine':
        print('Num of sample:', len(paths))
        save_dir = args.output_dir
        save_dir_a = os.path.join(save_dir, 'A')
        save_dir_b = os.path.join(save_dir, 'B')
        os.makedirs(save_dir_a, exist_ok=True)
        os.makedirs(save_dir_b, exist_ok=True)
        for path in paths:
            try:
                name=os.path.split(path)[-1].split('.')[0]


                save_path_a=os.path.join(save_dir_a, name+'.png')
                save_path_b=os.path.join(save_dir_b, name+'.png')
                psd_combine(path, save_path_a,save_path_b)
            except RuntimeError:
                print('ERROR: {}'.format(path), RuntimeError)
    elif args.mode == 'crop':
        down_scale_mean=1
        i=0
        save_dir=os.path.join(args.output_dir, str(args.crop_size))
        save_dir_a=os.path.join(save_dir, 'A')
        save_dir_b=os.path.join(save_dir, 'B')
        os.makedirs(save_dir_a, exist_ok=True)
        os.makedirs(save_dir_b, exist_ok=True)
        for path in tqdm(paths):
            name=os.path.split(path)[-1].split('.')[0]
            bigimg=load_image(path)
            for _ in range(30):
                down_scale=(np.random.uniform(-.5, .5)+1)*down_scale_mean
                resized=cv2.resize(
                    bigimg, (0, 0), fx=down_scale, fy=down_scale)
                a, b=split(resized)
                for _ in range(5):
                    sample_a, sample_b=random_crop(a, b, args.crop_size)
                    output_path_a=os.path.join(
                        save_dir_a, '{}_{}.png'.format(name, i))
                    output_path_b=os.path.join(
                        save_dir_b, '{}_{}.png'.format(name, i))
                    write_image(output_path_a, sample_a)
                    write_image(output_path_b, sample_b)
                    i += 1
