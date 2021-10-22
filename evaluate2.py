#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate1.py: Evaluation for the color distances
evaluate2.py: Evaluation for the content difference
"""

import numpy as np
import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
from skimage import exposure
import scipy.stats
eps = 1e-10
import cv2

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
    #hellinger_distance = sp.spatial.distance.euclidean(np.sqrt(x),np.sqrt(y))
    squared_hellinger_distance = 1 - np.dot(np.sqrt(x),np.sqrt(y))
    hellinger_distance = np.sqrt(squared_hellinger_distance)
    return hellinger_distance

def grad_xy(img):
    grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    grad = 0.5*np.abs(grad_x)+0.5*np.abs(grad_y)
    return grad


src_path = "./data/val.txt"
dst_path = "./compares/global_transfer/color_transfer/result/test_outputs/2018-10-8"
style_path = './data/2018-10-8/images/DJI_0481_3_3.png'
style_mask_path = './data/2018-10-8/masks/DJI_0481_3_3.png'

style = Image.open(style_path)
style = np.array(style)
style_mask = Image.open(style_mask_path)
style_mask = np.array(style_mask)


fid = open(src_path, "r")
count = 0
diff_grads = []
for item in fid:
    count = count+1
    src_img = np.array(Image.open(item.strip()).convert("L"))
    src_mask = np.array(Image.open(item.strip().replace("images","masks")))
    src_mask = src_mask>0
    out_img = np.array(Image.open(dst_path+"/"+item.strip().split("/")[-1]).convert("L")) 
    if (src_mask.sum()<1000):
        continue
    
    src_img = src_img.astype(np.float64)/255
    out_img = out_img.astype(np.float64)/255
    src_grad = grad_xy(src_img)
    out_grad = grad_xy(out_img)
    diff_grad = np.abs(src_grad-out_grad)
    diff_grad = np.sum(diff_grad[src_mask])/src_mask.sum()
    diff_grads.append(diff_grad)
    print(count,item)
fid.close()

Mgrad = np.mean(np.array(diff_grads))

print("good")






