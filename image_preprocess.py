#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:42:33 2022

@author: katie
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


def plot_ (img1, img2) :
    plt.figure(figsize = (20,5))
    
    plt.subplot(1,2,1)
    plt.imshow(img1, 'gray')
    
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(img2, 'gray')
    
    plt.axis('off')
    plt.show()
    

def t_img(img, k):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, 0)


def c_img(img, k):
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, k), np.uint8))

def d_img(img):
    return cv2.dilate(img, np.ones((2, 2), np.uint8), iterations=1)

def b_img(img):
    return cv2.GaussianBlur(img, (1, 1), 0)

path = './dataset/CAPTCHA_SIMPLE'
for image in os.listdir(path)[0:535]:

    if image[6:] != 'png':
        continue

    img = cv2.imread(os.path.join(path, image), cv2.IMREAD_COLOR)

    img1 = b_img(d_img(c_img(t_img(img, 145), 2)))
    img2 = b_img(d_img(c_img(t_img(img, 145), 3)))
    img3 = b_img(d_img(c_img(t_img(img, 215), 2)))
    img4 = b_img(d_img(c_img(t_img(img, 215), 3)))

    plot_4(img1, img2, img3, img4)
    plt.show()
    plt.axis('off')
    text = input("Enter 1, 2, 3, or 4:")
    if text == '1':
        denoised_img = img1
    elif text == '2':
        denoised_img = img2
    elif text == '3':
        denoised_img = img3
    elif text == '4':
        denoised_img = img4

    # denoised_img = cv2.cvtColor(denoise_mode, cv2.COLOR_BGR2RGB)
    cv2.imwrite('./denoised_samples/' + image, denoised_img)