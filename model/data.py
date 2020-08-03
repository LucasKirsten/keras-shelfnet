''' This script hold the class to handle the dataset (pre-processing, data loader, data visualization etc). You can modify the methods of this class as you wish as long as you keep their functionalities purposes. '''

import sys
sys.path.append('./')
from utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import imgaug.parameters as iap

from glob import glob

class DataLoader(object):
    ''' Data loader class examplified when we have the data in memory '''
    def __init__(self, folders_train, folders_val):
        ''' All paths to the images to train and validate '''
        
        # map paths and split dataset
        self.path_train = []; self.path_test = []

        for path in folders_train:
            self.path_train.extend(glob(path+'/*_image.jpg'))

        for path in folders_val:
            self.path_test.extend(glob(path+'/*_image.jpg'))

        print(f'Total train: {len(self.path_train)}, Total val: {len(self.path_test)}')
        
        # options for augmentation
        self.aug = iaa.SomeOf((0,3), [
                    iaa.Affine(rotate=(-10, 10), scale={"x": (0.5, 1.2), "y": (0.5, 1.2)}),
                    iaa.AdditiveGaussianNoise(scale=0.2*255),
                    iaa.GaussianBlur(sigma=(0.0, 3.0)),
                    iaa.BlendAlphaSimplexNoise(iaa.EdgeDetect(1.0),
                                               sigmoid_thresh=iap.Normal(10.0, 5.0)),
                    iaa.Add(50, per_channel=True),
                    iaa.WithChannels(0, iaa.Add((10, 100))),
                    iaa.Sharpen(alpha=0.2),
                    iaa.Fliplr(),
                    iaa.Flipud()
                ])
        
    def data_size(self, data):
        ''' Return the number of data contained in each set '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        if data=='train':
            return len(self.path_train)
        else:
            return len(self.path_test)

    def augment(self, image, mask):
        ''' Function for augmentation, if neccessary '''
        return self.aug(image=image, segmentation_maps=mask)
        
    def norm(self, img):
        ''' Function to normalize the data '''
        return img/255.
    
    def denorm(self, img):
        ''' Function to de-normalize the data for visualization purposes '''
        return np.uint8(img*255)
    
    def flow(self, data='train', batch_size=16, resize=None):
        ''' Generator of data '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        if data=='train':
            paths = self.path_train
        else:
            paths = self.path_test
        
        batchImg = []; batchMask = []; batchCnts = []
    
        np.random.shuffle(paths)
        while True:
            for p in paths:
                # open images
                try:
                    img = imread(p, resize=resize)
                    # open mask based on the image path
                    mask = imread(p.replace('image', 'mask'), resize=resize)[...,0]
                except:
                    continue

                # adjust mask
                _,mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                mask = SegmentationMapsOnImage(np.uint8(mask/255.), shape=img.shape)
                
                # augmentation
                if data=='train':
                    img, mask = self.augment(img, mask)
                
                mask = mask.get_arr()
                mask = mask[...,np.newaxis]
                img = self.norm(img)

                batchImg.append(img); batchMask.append(mask)

                if len(batchImg)>=batch_size:
                    yield(np.float32(batchImg), np.float32(batchMask))
                    batchImg = []; batchMask = []
                    
    def view_data(self, data='train', batch_size=4, resize=None):
        ''' Method visualize the data returned by the generator (to verification purposes) '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
            
        x, [y1, y2] = next(self.flow(data, batch_size, resize))
        print('Batch X: ', x.shape, x.min(), x.max())
        print('Batch Y: ', y.shape, y.min(), y.max())
        
        plt.figure(figsize=(10,10))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0]//2+1, 2, i+1)
            plt.imshow(np.hstack([self.denorm(x[i]),
                                  cv2.merge([self.denorm(y[i][...,0])]*3)]))
            
    def predict_data(self, model, data='test', batch_size=1):
        ''' Method to visualize trained model on train/test data '''
        
        assert data=='train' or data=='test', f'Data must be train or test, received {data}'
        
        x, y = next(self.flow(data, batch_size))
        p = model.predict(x)[0]
        
        plt.figure(figsize=(10,10))
        for i in range(x.shape[0]):
            plt.subplot(x.shape[0]//2+1, 2, i+1)
            plt.imshow(np.hstack([self.denorm(x[i]), cv2.merge([self.denorm(y[i][...,0])]*3), cv2.merge([self.denorm(p[i][...,0])]*3)]))
            plt.title('Input | Ground truth | Predicted')
            
    def predict_input(self, model, input_image):
        ''' Method to visualize trained model on an input image '''
        
        input_image = self.norm(input_image)
        p = model.predict(input_image[np.newaxis,...])[0]
        
        plt.figure(figsize=(10,10))
        plt.imshow(np.hstack([self.denorm(input_image), self.denorm(p)]))
        plt.title('Input | Predicted')