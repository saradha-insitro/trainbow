# -*- coding: utf-8 -*-
"""
Library for computing features that describe the global boundary features
This module provides functions that one can use to obtain and describe the shape and size of an object
Available Functions:
-general_morphology: Complutes simple morphology features
"""

# import libraries
import numpy as np
import pandas as pd
from skimage import measure


def general_morphology(regionmask: np.ndarray):
    """ Compute image morphology features
    Args:
        regionmask : binary background mask
    """
    
    morphology_features = ['centroid','area','perimeter','bbox_area','convex_area',
                            'equivalent_diameter','major_axis_length','minor_axis_length',
                            'eccentricity','orientation']
    
    regionmask=regionmask.astype('uint8')

    feat = pd.DataFrame(measure.regionprops_table(regionmask,properties=morphology_features))
    feat["concavity"] = (feat["convex_area"] - feat["area"]) / feat["convex_area"]
    feat["solidity"] = feat["area"] / feat["convex_area"]
    feat["a_r"] = feat["minor_axis_length"] / feat["major_axis_length"]
    feat["shape_factor"] = (feat["perimeter"] ** 2) / (4 * np.pi * feat["area"])
    feat["area_bbarea"] = feat["area"] / feat["bbox_area"]
    
    return pd.DataFrame(feat)