import tensorflow as tf
import numpy as np
import cv2
from psd_tools import PSDImage

print('Utils: Hello world ok2')

def psd2gray(psd, layer_idx, input_layer=False):
    '''
    
    '''
    layer = np.array(psd.layers[layer_idx].as_PIL())
    img = np.array(layer)[:,:,-1] if not input_layer else  cv2.cvtColor(layer, cv2.COLOR_BGRA2BGR)
    mask = np.zeros([psd.header.height, psd.header.width]) if not input_layer else np.zeros([psd.header.height, psd.header.width, 3])
    x1, y1, x2, y2 = [i for i in psd.layers[layer_idx].bbox]
    mask[y1:y2, x1:x2] = img
    return mask
def psd_combine(path,save_path=None, input_idx=2, line_idx=1, dot_idx=0):
    psd = PSDImage.load(path)
    input =psd2gray(psd, input_idx, input_layer=True)
    line = psd2gray(psd, line_idx)
    dot = psd2gray(psd, dot_idx)
    output = np.stack([line, dot, dot], axis=2)
    combine = np.concatenate([input, output], axis=1)
    if save_path is not None:
        cv2.imwrite(save_path, combine)
    return combine

def show(img):
    plt.figure(figsize=(20, 20))
    plt.show(plt.imshow(img))

def pad(image, crop_size):
    '''
        pad an image to be multiply with the crop size 
        example: 400x1000 -> 512x1024
    '''
    
    shape = tf.shape(image)
    h, w = shape[-3], shape[-2]
    new_h = tf.cast(tf.ceil(h/crop_size)*crop_size, tf.int32)
    new_w = tf.cast(tf.ceil(w/crop_size)*crop_size, tf.int32)
    top = tf.to_int32((new_h-h)/2)
    bot = new_h-h-top
    left = tf.to_int32((new_w-w)/2)
    right = new_w-w-left
    paddings = [(0, 0),(top, bot), (left, right), (0, 0)]
    image = tf.pad(image, paddings, 'CONSTANT')
    return image, top, left, bot, right

if __name__ == '__main__':
    '''
    import numpy as np
    data = np.random.randn(32, 400, 1000, 3)
    x = tf.placeholder(tf.float32, [None, 400, 1000, 3])
    y, _, _, _, _ = pad(data, 512)
    sess = tf.Session()
    y_out = sess.run(y, {x:data})
    print('x: {}\ty:{}'.format(x.shape, y_out.shape))
    '''
    from glob import glob
    paths = glob('data/psd_training1/*')