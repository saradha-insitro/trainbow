# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import pandas as pd
from trainbow.features.intensity_features import brainbow_intensity_features
from trainbow.features.morphology_features import general_morphology
from trainbow.features.segmentation_metrics import evaluate_cell_nuclei_segmentation

from skimage.color import rgb2hsv
from trainbow.utils import image_utils,database_utils
from skimage import measure



def _obj_brainbow_features (image: np.ndarray,
                            labelled_image: np.ndarray,
                            channel_map: dict):
    """
    This function takes in a raw image and a instance segmentation mask containing segmented objects 
    and a channel map containing the channel index of the brainbow flurophores 
    and then returns a dataframe of the intensity and morphology features. 
    """
    #get hsv image
    rgb_img = image_utils.create_composite_brainbow_image(image,channel_map,True)
    hsv_img = rgb2hsv(rgb_img)
    
    #get the regionprops in the respective channels
    DAPI_props = measure.regionprops(labelled_image, image[channel_map['DAPI']])
    eGFP_props = measure.regionprops(labelled_image, image[channel_map['eGFP']])
    mOrange_props = measure.regionprops(labelled_image, image[channel_map['mOrange']])
    mKate_props = measure.regionprops(labelled_image, image[channel_map['mKate2']])
    hsv_props = measure.regionprops(labelled_image, hsv_img)

    
    features = pd.DataFrame()
    for flag in (range(len(DAPI_props))):
        feat = pd.concat(
                [pd.DataFrame([flag+1], columns=["object_label"]),
                 brainbow_intensity_features(regionmask = DAPI_props[flag].image, 
                                            DAPI_image =  DAPI_props[flag].intensity_image, 
                                            eGFP_image =  eGFP_props[flag].intensity_image, 
                                            mOrange_image =  mOrange_props[flag].intensity_image, 
                                            mKate_image =  mKate_props[flag].intensity_image,
                                            hsv_image = hsv_props[flag].intensity_image).reset_index(drop=True),
                 general_morphology(regionmask = DAPI_props[flag].image).reset_index(drop=True)
                ],

                axis=1)

        features = pd.concat([feat,features], ignore_index = True)
        
    return features

def _image_brainbow_features (image: np.ndarray,
                              channel_map: dict,
                              labelled_image: np.ndarray = None):
    """
    This function takes in a raw image and a sematic segmentation mask
    and a channel map containing the channel index of the brainbow flurophores 
    and then returns a dataframe of the intensity and morphology features at the image level
    
    """
    #get hsv image
    rgb_img = image_utils.create_composite_brainbow_image(image,channel_map,True)
    hsv_img = rgb2hsv(rgb_img)
    
    # if no labelled mask is provided create an empty ones
    if labelled_image is None:
        labelled_image = np.ones([image.shape[0], image.shape[1]])        

    # if the given label image is empty assume the whole image is an onject
    if (np.max(labelled_image) == 0):
        labelled_image = labelled_image +1

    features = brainbow_intensity_features(regionmask = labelled_image, 
                                           DAPI_image =  image[channel_map['DAPI']],
                                           eGFP_image =  image[channel_map['eGFP']], 
                                           mOrange_image =  image[channel_map['mOrange']], 
                                           mKate_image =  image[channel_map['mKate2']],
                                           hsv_image = hsv_img)
    return features


def extract_brainbow_features( uid: str,
                               input_image_path: dir,
                               channel_map: dict,
                               object_type:str,
                               input_labelled_image_path:dir = None
                              ):
    #read in the raw image
    image = database_utils.read_image(input_image_path)

    #read in the segmented image
    if input_labelled_image_path is not None:
        obj_seg_mask = database_utils.read_image(input_labelled_image_path)
        
    elif input_labelled_image_path is None:
        obj_seg_mask = np.ones([image[channel_map['DAPI']].shape[0], image[channel_map['DAPI']].shape[1]])        

    #compute the brainbow features
    if (object_type == 'object'):
        feat = _obj_brainbow_features(image = image,
                                      labelled_image = obj_seg_mask,
                                      channel_map = channel_map)
        feat.rename(columns={'object_label': 'cell_label'}, inplace=True)
    elif (object_type == 'image'):
        obj_sem_mask = obj_seg_mask > 0 #converting instance to segmantic segmentations
        feat = _image_brainbow_features(image = image,
                                        channel_map = channel_map,
                                        labelled_image = obj_seg_mask)
        feat.rename(columns={'object_label': 'image_label'}, inplace=True)
    else:
        raise ValueError("method must be one of {}".format(["object","image"]))

    #set uid as a variable 
    feat['uid'] = uid
    
    del(image, obj_seg_mask)
    
    return feat

def extract_cell_nuclei_segmetaion_metrics(uid: str,
                                           nuc_mask_path: dir,
                                           cell_mask_path: dir,
                                          ):
    """
    Calculate segmentation metrics for each image
    """
            
    #read in the images
    nuc_mask = database_utils.read_image(nuc_mask_path)
    cell_mask = database_utils.read_image(cell_mask_path)

    feat = evaluate_cell_nuclei_segmentation(cell_mask,nuc_mask)
    if feat is None:
        pass
    else:
        feat = pd.concat([pd.DataFrame([uid], columns=["uid"]),
                          feat.reset_index(drop=True)
                          ],axis=1)
        del(cell_mask, nuc_mask)


        return pd.DataFrame(feat)
