import numpy as np 
from tqdm import tqdm
import cv2
import os
import imutils
from matplotlib import pyplot as plt
##########################################################
IMG_SIZE = 14


def denoise(img):

    denoised_img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return denoised_img
    

def crop_img(img):
    
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(img.astype("float32"), (3, 3), 0)
    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # # find contours in thresholded image, then grab the largest one
    # cnts = cv2.findContours(np.uint8(thresh.copy()), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)
    # print(len(cnts))

    # c = max(cnts, key=cv2.contourArea)

    # # find the extreme points
    # extLeft = tuple(c[c[:, :, 0].argmin()][0])
    # extRight = tuple(c[c[:, :, 0].argmax()][0])
    # extTop = tuple(c[c[:, :, 1].argmin()][0])
    # extBot = tuple(c[c[:, :, 1].argmax()][0])
    # ADD_PIXELS = 0
    # new_img =  img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    # new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
    # # new_img = denoise(new_img)

    return np.uint8(thresh)
