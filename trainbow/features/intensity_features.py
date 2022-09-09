# -*- coding: utf-8 -*-


"""
Library for computing features that describe the intensity distribution
This module provides functions that one can use to obtain and describe the intensity distribution of a given image
Available Functions:
-intensity_histogram_measures: Computes descriptors of the intensity histogram
"""
# Import modules
import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
from skimage import measure



def histogram_feat(regionmask: np.ndarray, intensity: np.ndarray):
    """Computes Intensity Distribution features
    
    This functions computes features that describe the distribution characteristic of the intensity.
    
    Args:
        regionmask : binary background mask
        intensity  : intensity image
    """
    
    foreground_pixels = (intensity * regionmask).ravel()
    feat = {
        "int_sum": np.sum(foreground_pixels, 0),
        "int_min": np.percentile(foreground_pixels, 0),
        "int_d25": np.percentile(foreground_pixels, 25),
        "int_median": np.percentile(foreground_pixels, 50),
        "int_d75": np.percentile(foreground_pixels, 75),
        "int_max": np.percentile(foreground_pixels, 100),
        "int_mean": np.mean(foreground_pixels),
        "int_mode": stats.mode(foreground_pixels, axis=None)[0][0],
        "int_sd": np.std(foreground_pixels),
        "kurtosis": float(kurtosis(foreground_pixels)),
        "skewness": float(skew(foreground_pixels)),
        "entropy": shannon_entropy((intensity * regionmask)),

    }
    return feat




def brainbow_ratio_features(eGFP_im: np.ndarray,
                            mOrange_im: np.ndarray,
                            mKate_im: np.ndarray):
    """Computes expression ratios of the brainbow flurophores
        
    Args:
        eGFP_im : eGFP image
        mOrange_im  : mOrange image
        mKate_im : mKate image
    """
    
    feat = {
            "mKate_by_mOrange_mean" : np.mean(np.divide(mKate_im,mOrange_im).flatten()),
            "mKate_by_eGFP_mean" : np.mean(np.divide(mKate_im,eGFP_im).flatten()),
            "eGFP_by_mOrange_mean" : np.mean(np.divide(eGFP_im,mOrange_im).flatten()),
            "mKate_by_mOrange_sd" : np.std(np.divide(mKate_im,mOrange_im).flatten()),
            "mKate_by_eGFP_sd" : np.std(np.divide(mKate_im,eGFP_im).flatten()),
            "eGFP_by_mOrange_sd" : np.std(np.divide(eGFP_im,mOrange_im).flatten())   
            }
    
    return feat


def image_colorfulness(R: np.ndarray,
                       G: np.ndarray,
                       B: np.ndarray):
    '''
    Compute image colorfulness
    
    ref: Hasler, David, and Sabine E. Suesstrunk. "Measuring colorfulness in natural images." Human vision and electronic imaging VIII. Vol. 5007. SPIE, 2003.
    '''
     
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    
    # derive the "colorfulness" metric and return it
    colorfulness = stdRoot + (0.3 * meanRoot)
    
    return colorfulness


def brainbow_colorfulness(eGFP_im,mOrange_im,mKate_im):
    '''
    Compute colorfulness meteric for multiple combinations of brainbow flurophores
    '''
    
    feat = {
          "eom_colorfulness" : image_colorfulness(eGFP_im,mOrange_im,mKate_im),
          "emo_colorfulness" : image_colorfulness(eGFP_im,mKate_im,mOrange_im),
          "meo_colorfulness" : image_colorfulness(mKate_im,eGFP_im,mOrange_im),
          "moe_colorfulness" : image_colorfulness(mKate_im,mOrange_im,eGFP_im),
          "ome_colorfulness" : image_colorfulness(mOrange_im,mKate_im,eGFP_im),
          "oem_colorfulness" : image_colorfulness(mOrange_im,eGFP_im,mKate_im)
          }

    return feat

def brainbow_intensity_features (regionmask: np.ndarray, 
                                         DAPI_image: np.ndarray,
                                         eGFP_image: np.ndarray,
                                         mOrange_image: np.ndarray,
                                         mKate_image: np.ndarray,
                                         hsv_image: np.ndarray
                                        ):
    """
    Compute all intensity features
    
    """
    
    hue_img = hsv_image[:, :, 0]
    saturation_img = hsv_image[:, :, 0]
    value_img = hsv_image[:, :, 2]

    
    feat = pd.concat([pd.DataFrame([histogram_feat(regionmask,DAPI_image)]).add_prefix('DAPI_').reset_index(drop=True),
                      pd.DataFrame([histogram_feat(regionmask,eGFP_image)]).add_prefix('eGFP_').reset_index(drop=True),
                      pd.DataFrame([histogram_feat(regionmask,mOrange_image)]).add_prefix('mOrange_').reset_index(drop=True),
                      pd.DataFrame([histogram_feat(regionmask,mKate_image)]).add_prefix('mKate_').reset_index(drop=True),
#                       pd.DataFrame([brainbow_ratio_features(eGFP_image,
#                                                             mOrange_image,
#                                                             mKate_image)]).reset_index(drop=True),
                      pd.DataFrame([brainbow_colorfulness(eGFP_image,
                                                          mOrange_image,
                                                          mKate_image)]).reset_index(drop=True),
                      
                      pd.DataFrame([histogram_feat(regionmask,hue_img)]).add_prefix('hue_').reset_index(drop=True),
                      pd.DataFrame([histogram_feat(regionmask,saturation_img)]).add_prefix('saturation_').reset_index(drop=True),
                      pd.DataFrame([histogram_feat(regionmask,value_img)]).add_prefix('value_').reset_index(drop=True),
                      
                     ],axis=1)
    
    return feat


