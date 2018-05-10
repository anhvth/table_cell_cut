from time import time
import tensorflow as tf
import cv2
import numpy as np
import os
from glob import glob
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='separate', choices=['separate', 'join'])
parser.add_argument('--input_dir')
parser.add_argument('--output_dir')
parser.add_argument('--checkpoint')
parser.add_argument('--ext', default=None)
parser.add_argument('--down_scale', type=float, default=572/512)
args = parser.parse_args()

def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)

def load_image(path, down_scale, verbal=False):
    '''
        Return RGB image
    '''
    #print('---------------------------\nprocess:', path)
    name = path.split('/')[-1].split('.')[0]
    image = cv2.imread(path)
    image = cv2.resize(image, (0,0), fx=down_scale, fy=down_scale)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image#[:,:image.shape[1]//2]

def write(path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, image)


def get_name(path):
    name, _ = os.path.splitext(os.path.basename(path))
    return name   


def save_output(input_image, output_image, save_dir):
    print(input_image.shape, output_image.shape)
    
    out1 = output_image[:,:,0]
    out2 = output_image[:,:,-1]
    if args.mode == 'separate':
        os.makedirs(save_dir, exist_ok=True)
        write(os.path.join(save_dir, 'input.png'), input_image)
        cv2.imwrite(os.path.join(save_dir, 'output1.png'), out1)
        cv2.imwrite(os.path.join(save_dir, 'output2.png'), out2)
    elif args.mode == 'join':
        #gray_input = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        merge = np.concatenate([input_image, output_image], axis=1)
        write(save_dir+'.png', merge)
    print('Output: ', save_dir)
    
def run(sess, feed_image, batch_size=8):
    batch_input = sess.run(batch_input_tensor, {inputs:feed_image})
    rv = []
    
    for i in range(0, len(batch_input), batch_size):
        print('\r {:0.2f} %'.format(i/len(batch_input)), end='')
        rv.append(sess.run(batch_output_tensor, {batch_input_placeholder: batch_input[i:i+batch_size]}))
    output_feed = np.concatenate(rv, axis=0)
    return sess.run(outputs, {batch_output_placeholder: output_feed, inputs:feed_image})


if __name__ == '__main__':
    meta_path = os.path.join(args.checkpoint, 'export.meta')
    down_scale = args.down_scale
    print('meta path:', meta_path)
    assert os.path.exists(meta_path), meta_path+' does not exist'
    tf.train.import_meta_graph(meta_path)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(
            args.checkpoint))
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])
        print('num of params:', sess.run(parameter_count))
        start = time()
        ext = 'png' if args.ext is None else args.ext
        paths = glob(os.path.join(args.input_dir, '*.{}'.format(ext)))
        assert len(paths) > 0, 'num of example must be \
            larger than 0 check extention --ext: {} and --input_dir:{}'.format(args.ext, args.input_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        print('Num of sample:', len(paths), args.input_dir)
        inputs = get_tensor_by_name('inputs')
        outputs = get_tensor_by_name('outputs')
        batch_input_tensor = get_tensor_by_name('batch_input_tensor')
        batch_input_placeholder = get_tensor_by_name('batch_input_placeholder')
        batch_output_tensor = get_tensor_by_name('batch_output_tensor')
        batch_output_placeholder = get_tensor_by_name('batch_output_placeholder')
        start = time()
        begin = start
        for path in paths:
            print(path)
            name = path.split('/')[-1].split('.')[0]
            image = load_image(path, verbal=True, down_scale=down_scale)
            print('image:', path, '\t shape: ', image.shape)
            output_image = run(sess, image)
            save_dir = os.path.join(args.output_dir, name)
            image = cv2.resize(image, (0,0), fx=1/down_scale, fy=1/down_scale)
            output_image = cv2.resize(output_image, (0,0), fx=1/down_scale, fy=1/down_scale)
            save_output(image, output_image, save_dir)
            print('Running time:', time()-start, 'image:', image.shape)

        print('Running time:', time()-begin)
