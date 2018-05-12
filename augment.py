import Augmentor 
import os 
import numpy as np 
from Augmentor.Operations import Operation
from PIL import Image
import cv2
class my_resize(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, ratio_low=1, ratio_high=1):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        self.ratio_low = ratio_low
        self.ratio_high = ratio_high
        # Set your custom operation's member variables here as required:
#         self.num_of_folds = num_of_folds

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        assert self.ratio_low<self.ratio_high
        resize_rate = np.random.uniform(self.ratio_low, self.ratio_high)
        for i,im in enumerate(images):
            old_size = im.size
            new_size = [int(o*resize_rate) for o in old_size]
            # print('new size: ',new_size, '\t old size: ', old_size)
            images[i] = im.resize(new_size, Image.ANTIALIAS)
        return images

class my_combine(Operation):
    def __init__(self, probability):
        assert probability==1
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        assert len(images) == 2
        a,b = [np.array(images[i]) for i in range(2)]
        image = np.zeros_like(a)
        image[...,0] = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        image[...,1:] = b[...,0:2]
        return [Image.fromarray(image)]

class my_split(Operation):
    def __init__(self, probability):
        assert probability==1
        Operation.__init__(self, probability)

    def perform_operation(self, images):
        assert len(images) == 1
        images = np.array(images[0])
        a = images[...,0]  
        b = images[...,1]
        c = images[...,2]
        
        inputs = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
        targets = np.stack([b,c,c], axis=2)
        image = np.concatenate([inputs, targets], axis=1)
        # image = (.5*a+.5*b).astype('uint8')
        # return [Image.fromarray(a), Image.fromarray(b)]
        return [Image.fromarray(image)]

class my_crop(Operation):
    # Here you can accept as many custom parameters as required:
    def __init__(self, probability, height, width, resize=1):
        # Call the superclass's constructor (meaning you must
        # supply a probability value):
        Operation.__init__(self, probability)
        self.resize = resize
        self.h = height
        self.w = width
        # Set your custom operation's member variables here as required:
#         self.num_of_folds = num_of_folds

    # Your class must implement the perform_operation method:
    def perform_operation(self, images):
        res = []
        y = np.random.randint(0, images[0].size[1]-self.h)
        x = np.random.randint(0, images[0].size[0]-self.w)
        for im in images:
            res.append(im.crop([x,y, x+self.w, y+self.h]))  
        return res

if __name__ == '__main__':
    input_dir = 'test'
    path = os.path.join(input_dir, 'A')
    os.system('rm -r {}'.format(os.path.join(path, 'output')))
    print(path)
    p = Augmentor.Pipeline(path)
    # p.set_seed(0)
    p.ground_truth(os.path.join(input_dir, 'B'))
    p.add_operation(my_combine(1)) 
    p.add_operation(my_resize(1,.5,1))
    p.add_operation(my_crop(1.,512, 512))
    p.random_distortion(1, 5,5,15)
    p.rotate(probability=1, max_left_rotation=5, max_right_rotation=5)
    p.add_operation(my_split(1))
    
    p.sample(10, True)
    # p.process()