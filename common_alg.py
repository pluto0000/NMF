import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import random
import copy

def load_data(root='data\ORL', reduce=4):
#def load_data(root='data\CroppedYaleB', reduce=4):
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):
        
        if not os.path.isdir(os.path.join(root, person)):
            continue
        
        for fname in os.listdir(os.path.join(root, person)):    
            
            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue
            if not fname.endswith('.pgm'):
                continue
            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L') 
            #img = img.resize([s//reduce for s in img.size])
            img = np.asarray(img)
            images.append(img)
            labels.append(i)

    return images, labels

images, labels = load_data()

def gaus_noise(imgs, means, sigma, num):
    img_copy = copy.deepcopy(imgs) 
    if len(np.asarray(img_copy).shape) != 3:
        img_copy = np.asarray(img_copy).reshape(img_copy.shape[1], np.asarray(images).shape[1], np.asarray(images).shape[2])
    for i in range(len(img_copy)):
        noise_num = int(num * img_copy[i].shape[0] * img_copy[i].shape[1])
        for j in range(noise_num):
            randX = random.randint(0, img_copy[i].shape[0] - 1)
            randY = random.randint(0, img_copy[i].shape[1] - 1)
            img_copy[i][randX, randY] = img_copy[i][randX, randY] + random.gauss(means, sigma=sigma)
            if img_copy[i][randX, randY] < 0:
                img_copy[i][randX, randY] = 0
            if img_copy[i][randX, randY]> 255:
                img_copy[i][randX, randY] = 255
        
    return img_copy

def salt_noise(imgs, per):
    img_copy = copy.deepcopy(imgs)
    
    if len(np.asarray(img_copy).shape) != 3:
        img_copy = np.asarray(img_copy).reshape(img_copy.shape[1], np.asarray(images).shape[1], np.asarray(images).shape[2])
  
    for i in range(len(img_copy)):
        noise_num = int(per * img_copy[i].shape[0] * img_copy[i].shape[1])
        for j in range(noise_num):
            randx = np.random.randint(0, img_copy[i].shape[0] - 1)
            randy = np.random.randint(0, img_copy[i].shape[1] - 1)
            if random.random() < 0.5:
                img_copy[i][randx, randy] = 0
            else:
                img_copy[i][randx, randy] = 255
    
    return img_copy


def block_noise(imgs, b):
    img_copy = copy.deepcopy(imgs)
    
    if len(np.asarray(img_copy).shape) != 3:
        img_copy = np.asarray(img_copy).reshape(img_copy.shape[1], np.asarray(images).shape[1], np.asarray(images).shape[2])
  
    for i in range(len(img_copy)):

        randx = np.random.randint(0, img_copy[i].shape[0] - b)
        randy = np.random.randint(0, img_copy[i].shape[1] - b)
        for j in range(b):
            for w in range(b):
                img_copy[i][randx+j][randy+w] = 255

    return img_copy

def test_data(imgs,lab,divided=0.7):
    imgs = np.array(imgs).transpose(1,2,0)
    imgs = np.concatenate(imgs,axis=0)
    train_size = int(imgs.shape[1]*divided)
    train_data = np.vstack((imgs,labels))
    #train_data = np.row_stack((imgs,lab))
    train_data = np.random.permutation(train_data.T)[0:train_size].T
    X_tr = train_data[:-1]
    Y_tr = train_data[-1]
    return X_tr,Y_tr

or_x_set=[]
or_y_set=[]
guas_set = []
salt_set = []
block_set = []
for i in range(5):
    X_set_or,Y_set_or = test_data(images,labels,0.8)
    or_x_set.append(X_set_or)
    or_y_set.append(Y_set_or)
    
    guas_set_d = gaus_noise(X_set_or, 100,50,1e-5).transpose(1,2,0)
    guas_set_d = np.concatenate(guas_set_d,axis=0)
    guas_set.append(guas_set_d)
    
    salt_set_d = salt_noise(X_set_or,0.1).transpose(1,2,0)
    salt_set_d = np.concatenate(salt_set_d,axis=0)
    salt_set.append(salt_set_d)
    
    block_set_d = block_noise(X_set_or,10).transpose(1,2,0)
    block_set_d = np.concatenate(block_set_d,axis=0)
    block_set.append(block_set_d)
    