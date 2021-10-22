#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate1: Evaluation for the color distances
"""
import numpy as np
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from skimage import exposure
import scipy.stats
eps = 1e-10

def rgb_hist(rgb):
    r_hist = np.histogram(rgb[:,:,0].flatten(), bins=255, range=(0,255))[0]/rgb[:,:,0].flatten().shape[0]
    g_hist = np.histogram(rgb[:,:,1].flatten(), bins=255, range=(0,255))[0]/rgb[:,:,1].flatten().shape[0]
    b_hist = np.histogram(rgb[:,:,2].flatten(), bins=255, range=(0,255))[0]/rgb[:,:,2].flatten().shape[0]
    rgb_hist = np.concatenate((r_hist,g_hist,b_hist), axis=0)
    return rgb_hist

def rgb_hist(rgb, rgb_mask):
    r_hist = np.histogram(rgb[:,:,0][rgb_mask], bins=255, range=(0,255))[0]/rgb[:,:,0][rgb_mask].shape[0]
    g_hist = np.histogram(rgb[:,:,1][rgb_mask], bins=255, range=(0,255))[0]/rgb[:,:,1][rgb_mask].shape[0]
    b_hist = np.histogram(rgb[:,:,2][rgb_mask], bins=255, range=(0,255))[0]/rgb[:,:,2][rgb_mask].shape[0]
    rgb_hist = np.concatenate((r_hist,g_hist,b_hist), axis=0)
    return rgb_hist

def hellinger(x,y):
    """Computes the Hellinger distance between two sum-normalized vectors
    
    :math:`d_h(x,y) = \\frac{1}{2} \sqrt{\sum_{i} \\left(\sqrt{x_i} - \sqrt{y_i}\\right)^2}`
    
    Equivalently, for sum-normalized vectors
    
    :math:`d_h(x,y) =  1 - \sum_{i} \\left(\sqrt{x_i} - \sqrt{y_i}\\right)`
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns: ``hellinger(x,y) = np.sqrt( 1 - np.dot(np.sqrt(x),np.sqrt(y)) )``
    """
    
    #hellinger_distance = sp.spatial.distance.euclidean(np.sqrt(x),np.sqrt(y))
    squared_hellinger_distance = 1 - np.dot(np.sqrt(x),np.sqrt(y))
    hellinger_distance = np.sqrt(squared_hellinger_distance)
    return hellinger_distance

src_path = "./data/val.txt"
dst_path = "./compares/global_transfer/color_transfer/result/test_outputs/2018-10-8"                     
style_path = './data/2018-10-8/images/DJI_0481_3_3.png'
style_mask_path = './data/2018-10-8/masks/DJI_0481_3_3.png'

style = Image.open(style_path)
style = np.array(style)
style_mask = Image.open(style_mask_path)
style_mask = np.array(style_mask)
style_mask = style_mask==1
style_hist = rgb_hist(style,style_mask)
style_hist = (style_hist+eps)/(style_hist+eps).sum()

fid = open(src_path, "r")
count = 0
KL_src_list = []
KL_out_list = []
for item in fid:
    count = count+1
    src_img = np.array(Image.open(item.strip()))
    src_mask = np.array(Image.open(item.strip().replace("images","masks")))
    src_mask = src_mask==1
    out_img = np.array(Image.open(dst_path+"/"+item.strip().split("/")[-1])) 
    if (src_mask.sum()<1000):
        continue

    src_hist = rgb_hist(src_img,src_mask)
    out_hist = rgb_hist(out_img,src_mask)
    src_hist = (src_hist+eps)/(src_hist+eps).sum()
    out_hist = (out_hist+eps)/(out_hist+eps).sum()
    KL_src = scipy.stats.entropy(src_hist, style_hist) 
    KL_out = scipy.stats.entropy(out_hist, style_hist) 
    KL_src_list.append(KL_src)
    KL_out_list.append(KL_out)
    print(count,item,src_mask.sum(),KL_src,KL_out)
fid.close()

KL_src_mean = np.array(KL_src_list).mean()
KL_out_mean = np.array(KL_out_list).mean()


style = Image.open(style_path)
style = np.array(style)
style_mask = Image.open(style_mask_path)
style_mask = np.array(style_mask)
style_mask = style_mask==1
style_hist = (style_hist+eps)/(style_hist+eps).sum()

fid = open(src_path, "r")
count = 0
HD_src_list = []
HD_out_list = []
for item in fid:
    count = count+1
    src_img = np.array(Image.open(item.strip()))
    src_mask = np.array(Image.open(item.strip().replace("images","masks")))
    src_mask = src_mask==1
    out_img = np.array(Image.open(dst_path+"/"+item.strip().split("/")[-1])) 
    if (src_mask.sum()<1000):
        continue

    src_hist = rgb_hist(src_img,src_mask)
    out_hist = rgb_hist(out_img,src_mask)
    src_hist = (src_hist+eps)/(src_hist+eps).sum()
    out_hist = (out_hist+eps)/(out_hist+eps).sum()
    HD_src = hellinger(src_hist,style_hist)
    HD_out = hellinger(out_hist,style_hist)
    HD_src_list.append(HD_src)
    HD_out_list.append(HD_out)
    print(count,item.strip().split("/")[-1],src_mask.sum(),HD_src,HD_out)
fid.close()

HD_src_mean = np.array(HD_src_list).mean()
HD_out_mean = np.array(HD_out_list).mean()



















np.sqrt(1-np.sum(np.sqrt(src_hist*style_hist)))

h2=np.sqrt(1-np.sum(np.sqrt((src_hist+eps)*(style_hist+eps))))

x = (src_hist+eps)/(src_hist+eps).sum()
y = (out_hist+eps)/(out_hist+eps).sum()

def hellinger(x,y):
    """Computes the Hellinger distance between two sum-normalized vectors
    
    :math:`d_h(x,y) = \\frac{1}{2} \sqrt{\sum_{i} \\left(\sqrt{x_i} - \sqrt{y_i}\\right)^2}`
    
    Equivalently, for sum-normalized vectors
    
    :math:`d_h(x,y) =  1 - \sum_{i} \\left(\sqrt{x_i} - \sqrt{y_i}\\right)`
    
    :param x: The first vector
    :param y: The second vector
    :type x: ndarray
    :type y: ndarray
    :returns: ``hellinger(x,y) = np.sqrt( 1 - np.dot(np.sqrt(x),np.sqrt(y)) )``
    """
    
    #hellinger_distance = sp.spatial.distance.euclidean(np.sqrt(x),np.sqrt(y))
    squared_hellinger_distance = 1 - np.dot(np.sqrt(x),np.sqrt(y))
    hellinger_distance = np.sqrt(squared_hellinger_distance)
    return hellinger_distance






























plt.figure(1)
plt.plot(src_hist)
plt.figure(2)
plt.plot(style_hist)


np.sum(np.abs(out_hist-style_hist))
np.sum(np.abs(src_hist-style_hist))














plt.figure(1)
plt.imshow(src_img)
plt.figure(2)
plt.imshow(src_mask)
plt.figure(3)
plt.imshow(out_img)



plt.figure(1)
plt.imshow(style)
plt.figure(2)
plt.imshow(style_mask)












