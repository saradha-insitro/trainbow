# -*- coding: utf-8 -*-
"""
"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

def normalize_image(raw_image:np.ndarray):
    '''
    Function that takes in a 2D image and normalises the image by streching the intensity between 0 and 1 and returns the normalised image
    
    Args: 
        raw_image : single channel 2D image
    '''
    image = cv.normalize(
        raw_image, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F
    )
    return image

def create_composite_brainbow_image( image:np.ndarray, 
                                     channel_map:dict, 
                                     normalize:bool = True):
    '''
    Function that takes in an image and normalises across the brainbow flurophores and creates and returns a composite RGB image that has a bit-depth of 8. 
    
    Args:  
        image_path: multichannel image of type CHW
        channel_map: a dictionary indicating the channel map specifying the channel position/index of mKate2, mOrange and eGFP
        normalize: (bool) instruct to normalise the each channel of the image
    '''
    
    #create composite image
    if (normalize == True): 
        comp = np.stack([normalize_image(image[channel_map['mKate2']]).astype('uint8'),
                         normalize_image(image[channel_map['mOrange']]).astype('uint8'),
                         normalize_image(image[channel_map['eGFP']]).astype('uint8')],
                            axis =2)
    else: 
         comp = np.stack([image[channel_map['mKate2']],
                          image[channel_map['mOrange']],
                          image[channel_map['eGFP']]],
                            axis =2)
    return comp

def create_cell_image_from_brainbow_FP(image:np.ndarray, 
                                         channel_map:dict):
    '''
    Functions that takes in a multichannel image containing the brainbow fluroscent protiens and retuns a cell image computed from adding the intensity of all three fluroscent protiens 
    
    Args:
        image:  multichannel image of type CHW
        channel_map: a dictionary indicating the channel map specifying the channel position/index of mKate2, mOrange and eGFP
    '''
    
    cell_ch = np.add(np.add(image[channel_map['mKate2']],image[channel_map['mOrange']]),
                     image[channel_map['eGFP']]).astype("uint16")
    
    return cell_ch


    
    
