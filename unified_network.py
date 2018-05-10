from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='data/croped_image640', help="path to folder containing images")
parser.add_argument("--mode", default="train", choices=["train", "test", "export"])
parser.add_argument("--output_dir",required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")

parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")

parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--crop_size", type=int, default=640, help="scale images to this size before cropping to 256x256")
parser.add_argument("--scale_size", type=int, default=680, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
parser.add_argument('--drop_rate', type=float, default=.5)
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
parser.add_argument("--deepunet", dest="deepunet", action="store_true", help="flip images horizontally")
parser.add_argument("--unet", dest="deepunet", action="store_false", help="flip images horizontally")
parser.add_argument("--gpu_fraction", type=float, default=0.8)
parser.add_argument("--down_scale_rate", type=float, default=4)
parser.add_argument("--outer_first", type=int, default=1000)


a = parser.parse_args()
ngf = a.ngf
EPS = 1e-12
initializer=tf.random_normal_initializer(0, 0.02)
CROP_SIZE = a.crop_size
strides=[CROP_SIZE//2, CROP_SIZE//2]

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, gen_loss_L1_inner, gen_loss_L1_outer, train, gen_train_inner, gen_train_outer")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))



def conv2d_dilated(x, filters, rate):
    weights = tf.Variable(tf.random_normal(shape=[5,5,x.get_shape().as_list()[-1],filters], mean=0.0, stddev=0.02))
    biases = tf.Variable(tf.zeros([x.get_shape().as_list()[-1]]))
    return tf.nn.atrous_conv2d(x, weights, padding='SAME', rate=rate)+biases

# def gen_conv(batch_input, out_channels, dilation_rate=1):
#     initializef = tf.random_normal_initializer(0, 0.02)

#     if dilation_rate > 1:
#         batch_input_dilated = conv2d_dilated(batch_input, batch_input.get_shape().as_list()[-1])
#         batch_input = tf.concat([batch_input, batch_input_dilated], axis=-1)

#     return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)

def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    # batch_input_dilated = conv2d_dilated(batch_input, batch_input.get_shape().as_list()[-1], dilation_rate)
    # batch_input = tf.concat([batch_input, batch_input_dilated], axis=-1)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=3, strides=(1, 1), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a=.2):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)
    
    # input_paths = input_paths[:32]
    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == "train")
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        if a.lab_colorization:
            # load color and brightness from image, no B image exists here
            lab = rgb_to_lab(raw_input)
            L_chan, a_chan, b_chan = preprocess_lab(lab)
            a_images = tf.expand_dims(L_chan, axis=2)
            b_images = tf.stack([a_chan, b_chan], axis=2)
        else:
            # break apart image pair and move to range [-1, 1]
            width = tf.shape(raw_input)[1] # [height, width, channels]
            a_images = preprocess(raw_input[:,:width//2,:])
            b_images = preprocess(raw_input[:,width//2:,:])

    if a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    elif a.which_direction == "BtoA":
        inputs, targets = [b_images, a_images]
    else:
        raise Exception("invalid direction")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image, c):
        r = image
        if a.flip:
            r =tf.image.random_flip_left_right(r, seed=seed)
        
        r=tf.cond(c, lambda:tf.image.rot90(r), lambda:r)
        
        
        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        #r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            #r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
            r = tf.random_crop(r, (CROP_SIZE, CROP_SIZE, 3), seed=seed)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    c = tf.greater(1., tf.random_uniform([], minval=0, maxval=2))
    with tf.control_dependencies([c]):
        with tf.name_scope("input_images"):
            input_images = transform(inputs, c)

        with tf.name_scope("target_images"):
            target_images = transform(targets, c)
            
    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )

def conv_bn_relu(x,filters):
    conv = gen_conv(x, filters)
    bn = batchnorm(conv)
    return tf.nn.relu(bn)




def down_block(input, padding='same', pool_size=2):
    x = tf.layers.max_pooling2d(input, pool_size, pool_size)
    temp = conv_bn_relu(x, a.ngf)

    bn = batchnorm(gen_conv(temp, a.ngf))
    bn += x
    act = tf.nn.relu(bn)
    print(act.shape)
    return bn, act

def up_block(act, bn, use_drop=False):
    bn_shape = tf.shape(bn)
    h, w = bn_shape[1], bn_shape[2]#bn.get_shape().as_list()[1:3]
    #h *= 2
    #w *= 2
    x = tf.image.resize_images(
                act,
                (h, w),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                align_corners=False
            )
    temp = tf.concat([bn, x], axis=-1)
    temp = conv_bn_relu(temp, a.ngf)
    bn = batchnorm(gen_conv(temp, a.ngf))
    output = tf.nn.relu(bn)
    if use_drop:
        output = tf.nn.dropout(output, keep_prob=0.5)
    print(output.shape)
    return output



def create_outer(generator_inputs, features, generator_outputs_channels):
    x = conv_bn_relu(generator_inputs, a.ngf)
    print(generator_inputs.shape)
    net = conv_bn_relu(x, a.ngf)
    bn1 = batchnorm(gen_conv(net, a.ngf))
    act1 = tf.nn.relu(bn1)
    bn2, act2 = down_block(act1, pool_size=1)
    bn3, act3 = down_block(act2, pool_size=2)
    '''
    bn4, act4 = down_block(act3, pool_size=2)
    bn5, act5 = down_block(act4, pool_size=2)
    bn6, act6 = down_block(act5, pool_size=2)
    
    # print('Act6:{}\t, bn5:{}'.format(act6.shape, bn5.shape))
    temp = up_block(act5, bn4, use_drop=True)
    temp = up_block(temp, bn3, use_drop=True)
    temp = up_block(temp, bn2, use_drop=True)
    '''
    fuse = act3+features#tf.concat([act3, features], axis=-1)
    temp = up_block(fuse, bn3)
    #score3 = tf.tanh(gen_conv(temp, 3))

    temp = up_block(temp, bn2)
    #score2 = tf.tanh(gen_conv(temp, 3))

    temp = up_block(temp, bn1)
    score1 = tf.tanh(gen_conv(temp, 3))
    #print([score1.shape, score2.shape, score3.shape])
    return [score1]


def create_innter(generator_inputs, generator_outputs_channels):
    x = conv_bn_relu(generator_inputs, a.ngf)
    print(generator_inputs.shape)
    net = conv_bn_relu(x, a.ngf)
    bn1 = batchnorm(gen_conv(net, a.ngf))
    act1 = tf.nn.relu(bn1)
    bn2, act2 = down_block(act1, pool_size=4)
    bn3, act3 = down_block(act2, pool_size=4)
    bn4, act4 = down_block(act3, pool_size=2)
    bn5, act5 = down_block(act4, pool_size=2)
    bn6, act6 = down_block(act5, pool_size=2)
    bn7, act7 = down_block(act6, pool_size=2)

    # print('Act6:{}\t, bn5:{}'.format(act6.shape, bn5.shape))
    temp = up_block(act7, bn6, use_drop=True)
    temp = up_block(temp, bn5, use_drop=True)
    temp = up_block(temp, bn4, use_drop=True)
    
    temp = up_block(temp, bn3)
    score3 = tf.tanh(gen_conv(temp, 3))

    temp = up_block(temp, bn2)
    score2 = tf.tanh(gen_conv(temp, 3))

    temp = up_block(temp, bn1)
    score1 = tf.tanh(gen_conv(temp, 3))
    print([score1.shape, score2.shape, score3.shape])
    return score1, score2, score3, temp


def create_generator(inputs, out_channels=3):

    print('Inputs:', inputs)
    input_shape = inputs.get_shape().as_list()#tf.shape(inputs)
    h, w = input_shape[1], input_shape[2]
    h_downscale, w_downscale = tf.to_int32(h/a.down_scale_rate), tf.to_int32(w/a.down_scale_rate)

    inputs_downscale = tf.image.resize_images(inputs, (h_downscale, w_downscale), tf.image.ResizeMethod.AREA)
    with tf.variable_scope("g_inner"):
        out_channels = out_channels#int(inputs.get_shape()[-1])
        print('Inner')
        g_inner_outputs = create_innter(inputs_downscale, out_channels)   
        inner_scores, inner_features = g_inner_outputs[:-1], g_inner_outputs[-1] 
        inner_outputs = inner_scores[0]

    with tf.variable_scope('g_outer'):
        out_channels = out_channels#,int(inputs.get_shape()[-1])
        print('Outer')
        outer_scores = create_outer(inputs, inner_features, 3)   
        outer_outputs = outer_scores[0]
        inner_outputs_upscale = tf.image.resize_images(inner_outputs, (h, h), tf.image.ResizeMethod.AREA)
        
        outputs = tf.concat([inner_outputs_upscale, outer_outputs], axis=-2)
        
    return outer_outputs

def create_model(inputs, targets):
    print('Inputs:', inputs)
    input_shape = inputs.get_shape().as_list()#tf.shape(inputs)
    h, w = input_shape[1], input_shape[2]
    h_downscale, w_downscale = tf.to_int32(h/a.down_scale_rate), tf.to_int32(w/a.down_scale_rate)

    inputs_downscale = tf.image.resize_images(inputs, (h_downscale, w_downscale), tf.image.ResizeMethod.AREA)
    with tf.variable_scope("g_inner"):
        out_channels = int(targets.get_shape()[-1])
        print('Inner')
        g_inner_outputs = create_innter(inputs_downscale, out_channels)   
        inner_scores, inner_features = g_inner_outputs[:-1], g_inner_outputs[-1] 
        inner_outputs = inner_scores[0]

    with tf.variable_scope('g_outer'):
        out_channels = int(targets.get_shape()[-1])
        print('Outer')
        outer_scores = create_outer(inputs, inner_features, 3)   
        outer_outputs = outer_scores[0]
        inner_outputs_upscale = tf.image.resize_images(inner_outputs, (h, h), tf.image.ResizeMethod.AREA)
        
        outputs = tf.concat([inner_outputs_upscale, outer_outputs], axis=-2)
    def get_tagers(target_images, scores):
        rv = []
        for score in scores:
            h,w = score.get_shape().as_list()[1:3]
            r = tf.image.resize_images(target_images, [h,w], method=tf.image.ResizeMethod.AREA)
            rv.append(r)
        return rv
    with tf.name_scope("g_inner_loss"):
        targets_inner = get_tagers(targets, inner_scores)
        gen_loss_L1_inner = tf.reduce_mean([tf.reduce_mean(tf.square(t-s)) for t,s in zip(targets_inner, inner_scores)])
        gen_loss_inner = gen_loss_L1_inner

    with tf.name_scope("gen_inner_train"):
        gen_tvars_inner = [var for var in tf.trainable_variables() if var.name.startswith("g_inner")]
        gen_optim_inner = tf.train.AdamOptimizer(a.lr, a.beta1)
        gen_grads_and_vars_inner = gen_optim_inner.compute_gradients(gen_loss_inner, var_list=gen_tvars_inner)
        gen_train_inner = gen_optim_inner.apply_gradients(gen_grads_and_vars_inner)

    with tf.name_scope("g_outer_loss"):
        targets_outer= get_tagers(targets, outer_scores)
        gen_loss_L1_outer = tf.reduce_mean([tf.reduce_mean(tf.square(t-s)) for t,s in zip(targets_outer, outer_scores)])
        gen_loss_outer = gen_loss_L1_outer

    with tf.name_scope("gen_outer_train"):
        gen_tvars_outer = [var for var in tf.trainable_variables() if var.name.startswith("g_outer")]
        gen_optim_outer = tf.train.AdamOptimizer(a.lr, a.beta1)
        gen_grads_and_vars_outer = gen_optim_outer.compute_gradients(gen_loss_outer, var_list=gen_tvars_outer)
        gen_train_outer = gen_optim_inner.apply_gradients(gen_grads_and_vars_outer)


    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([gen_loss_L1_inner, gen_loss_L1_outer])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        gen_loss_L1_inner=ema.average(gen_loss_L1_inner),
        gen_loss_L1_outer=ema.average(gen_loss_L1_outer),
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train_inner),
        gen_train_inner=gen_train_inner,
        gen_train_outer=gen_train_outer,
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export":
        if a.checkpoint is None:
            raise Exception("checkpoint required for test mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "lab_colorization"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))
    if a.mode == "export":
        # export the generator to a meta graph that can be imported later for standalone generation
        def extract_patches(image, k_size, strides):
            images = tf.extract_image_patches(tf.expand_dims(image, 0), k_size, strides, rates=[1, 1, 1, 1], padding='SAME')[0]
            images_shape = tf.shape(images)
            images_reshape = tf.reshape(images, [images_shape[0]*images_shape[1], *k_size[1:3], 3])
            images, n1, n2 = tf.cast(images_reshape, tf.uint8) , images_shape[0], images_shape[1]
            return images, n1, n2

        def join_patches(images, n1, n2, k_size, strides):

            s1 = k_size[1]//2-strides[1]//2
            s2 = k_size[2]//2-strides[2]//2
            roi = images[:, 
                        s1:s1+strides[1],\
                        s2:s2+strides[2],
                        :]
            new_shape = [n1, n2, *roi.shape[1:]]
            reshaped_roi = tf.reshape(roi, new_shape)
            reshaped_roi = tf.transpose(reshaped_roi, perm=[0,2,1,3,4])
            rs = tf.shape(reshaped_roi)
            rv = tf.reshape(reshaped_roi, [rs[0]*rs[1], rs[2]*rs[3], -1])
            return rv

        def resize(image, new_size=None):
            shape = tf.shape(image)
            h, w = shape[0], shape[1]
            if new_size is None:
                new_h = tf.cast(tf.ceil(h/CROP_SIZE)*CROP_SIZE, tf.int32)
                new_w = tf.cast(tf.ceil(w/CROP_SIZE)*CROP_SIZE, tf.int32)
            else:
                new_h, new_w = new_size
            return tf.image.resize_bilinear(tf.expand_dims(image, 0), (new_h, new_w))[0]
        # inputs = tf.placeholder(tf.float32, [None, *CROP_SIZE, 3], 'inputs')
        inputs = tf.placeholder(tf.float32, [None, None, 3], 'inputs')
        inputs_shape = tf.shape(inputs)
        input_resized = resize(inputs)
        # strides = tf.placeholder_with_default([32, 256], shape=[2], name='strides')
        images, n1, n2 = extract_patches(input_resized, [1, CROP_SIZE, CROP_SIZE,1], [1,*strides,1])

        batch_input_tensor = tf.identity(images / 255, 'batch_input_tensor')
        batch_input_placeholder = tf.placeholder(tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'batch_input_placeholder')
        batch_output = deprocess(
            create_generator(preprocess(batch_input_placeholder), 3))
        h, w = batch_input_placeholder.get_shape().as_list()[1:3]
        batch_output = tf.image.resize_bilinear(batch_output, (h, w))
        batch_output_tensor = tf.identity(batch_output, name='batch_output_tensor')
        batch_output_placeholder = tf.placeholder(tf.float32, [None, CROP_SIZE, CROP_SIZE, 3], 'batch_output_placeholder')
        batch_output = join_patches(batch_output_placeholder, n1, n2, [1, CROP_SIZE, CROP_SIZE,1], [1,*strides,1])
        batch_output = resize(batch_output, [inputs_shape[0], inputs_shape[1]])
        outputs = tf.identity(tf.cast(batch_output*255, tf.uint8), name='outputs')

        init_op = tf.global_variables_initializer()
        restore_saver = tf.train.Saver()
        export_saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            restore_saver.restore(sess, checkpoint)
            print("exporting model:", checkpoint)
            export_saver.export_meta_graph(
                filename=os.path.join(a.output_dir, "export.meta"))
            export_saver.save(sess, os.path.join(
                a.output_dir, "export"), write_meta_graph=False)

        return
    
    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    # undo colorization splitting on images that we use for display/output
    if a.lab_colorization:
        if a.which_direction == "AtoB":
            # inputs is brightness, this will be handled fine as a grayscale image
            # need to augment targets and outputs with brightness
            targets = augment(examples.targets, examples.inputs)
            outputs = augment(model.outputs, examples.inputs)
            # inputs can be deprocessed normally and handled as if they are single channel
            # grayscale images
            inputs = deprocess(examples.inputs)
        elif a.which_direction == "BtoA":
            # inputs will be color channels only, get brightness from targets
            inputs = augment(examples.inputs, examples.targets)
            targets = deprocess(examples.targets)
            outputs = deprocess(model.outputs)
        else:
            raise Exception("invalid direction")
    else:
        inputs = deprocess(examples.inputs)
        targets = deprocess(examples.targets)
        outputs = deprocess(model.outputs)

    def convert(image):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)


    tf.summary.scalar("generator_loss_L1_inner", model.gen_loss_L1_inner)
    tf.summary.scalar("generator_loss_L1_outer", model.gen_loss_L1_outer)


    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=a.gpu_fraction)

    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print("parameter_count =", sess.run(parameter_count))

        if a.checkpoint is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == "test":
            # testing
            # at most, process the test data once
            start = time.time()
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                results = sess.run(display_fetches)
                filesets = save_images(results)
                for i, f in enumerate(filesets):
                    print("evaluated image", f["name"])
                index_path = append_index(filesets)
            print("wrote index at", index_path)
            print("rate", (time.time() - start) / max_steps)
        else:
            # training
            start = time.time()

            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                options = None
                run_metadata = None
                if should(a.trace_freq):
                    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                fetches = {
                    "train": model.train,
                    "global_step": sv.global_step,
                }
                g_step = sess.run(sv.global_step)
                if g_step > a.outer_first:
                    fetches["train_outer"] = model.gen_train_outer
                if should(a.progress_freq):

                    fetches["gen_loss_L1_inner"] = model.gen_loss_L1_inner
                    fetches["gen_loss_L1_outer"] =  model.gen_loss_L1_outer

                if should(a.summary_freq):
                    fetches["summary"] = sv.summary_op

                if should(a.display_freq):
                    fetches["display"] = display_fetches

                results = sess.run(fetches, options=options, run_metadata=run_metadata)

                if should(a.summary_freq):
                    print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])

                if should(a.display_freq):
                    print("saving display images")
                    filesets = save_images(results["display"], step=results["global_step"])
                    append_index(filesets, step=True)

                if should(a.trace_freq):
                    print("recording trace")
                    sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                if should(a.progress_freq):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * a.batch_size / (time.time() - start)
                    remaining = (max_steps - step) * a.batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("gen_loss_L1_inner", results["gen_loss_L1_inner"])
                    print("gen_loss_L1_outer", results["gen_loss_L1_outer"])

                if should(a.save_freq):
                    print("saving model")
                    saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

                if sv.should_stop():
                    break

main()