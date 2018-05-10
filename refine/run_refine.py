import tensorflow as tf
import cv2
import numpy as np
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", default='output/refine/1/frozen/', help="")
parser.add_argument("--input_dir", default='output/shalowunet/1/frozen/temp1/', help="number of generator filters in first conv layer")
args = parser.parse_args()












def get_tensor_by_name(name):
    name_on_device = '{}:0'.format(name)
    return tf.get_default_graph().get_tensor_by_name(name_on_device)

def read_inp(p_in):
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
    
    img_inp1 = read_image(p_in, 'input')
    img_inp2 = read_image(p_in, 'output1')
    img_inp3 = read_image(p_in, 'output2')
    img_inp = np.stack([img_inp1,img_inp2,img_inp3], axis=-1)
    return img_inp


if __name__ == '__main__':
    output_dir = os.path.join(args.checkpoint, 'frozen')
    cmd = 'python refine/deepunet_refine.py --mode export --checkpoint {} --output_dir {} --ngf 32 --crop_size 1024'.format(args.checkpoint, output_dir)
    print('Export model: ',os.system(cmd)==0)
    
    tf.reset_default_graph()
    mtp = os.path.join(output_dir,'*.meta')
    print('+++++++++++++',mtp)
    meta_path = glob(mtp)[0]
    print('meta path:', meta_path)
    assert os.path.exists(meta_path)

    tf.train.import_meta_graph(meta_path)
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=1)
        print('restore: ', tf.train.latest_checkpoint(args.checkpoint))
        saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))

        inputs = get_tensor_by_name('inputs')
        outputs = get_tensor_by_name('outputs')
        paths = glob(os.path.join(args.input_dir, '*'))


        for path in paths:
            img_inp = read_inp(path)
            rv = sess.run(outputs, {inputs:img_inp})
            output_path = path[:-4]+'_refine_line.png'
            print('Output: ', output_path)
            cv2.imwrite(cv2.cvtColor(output_path, cv2.COLOR_RGB2BGR), rv[:,:,:])