# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from cellpose import models

from scipy import ndimage
from skimage import filters, measure,exposure,segmentation

from trainbow.utils import image_utils, database_utils
from trainbow.segmentation import voronoi

def segment_nuclei_cellpose(images:list,
                            nuc_dia:int = 35,
                            cp_flowthresh:float = 0.4,
                            use_gpu:bool = False
                           ):
    '''
    Function to segment nuclei using the public cell pose model.
    
    Args:
        images: list of 2D grayscale images of nuclei
        nuc_dia: estimated diameter of the nuclei
        cp_flowthresh: cell pose's flow threshold. Increase this value if there is the objects are underdetected
    '''
    
    # model_type='cyto' or model_type='nuclei'
    model = models.Cellpose(gpu = use_gpu, model_type='nuclei')

    nuc_masks, _, _, _ = model.eval(images, 
                                    diameter = nuc_dia, 
                                    channels = [0,0],
                                    flow_threshold = cp_flowthresh, 
                                    do_3D = False,
                                    interp = False)
    return nuc_masks

def segment_cells_watershed(cell_image:np.ndarray,
                            nuc_seg:np.ndarray
                           ):
    
    #process and threshold the cell stain mask to generate the cell mask
    processed_cyto = filters.gaussian(exposure.equalize_adapthist(cell_image, clip_limit=0.1),sigma=3)
    thresholded_cyto = processed_cyto > filters.threshold_li(processed_cyto)
    
    #Generate the cell basins using a eucledian distance transform of the cell mask
    distance = ndimage.distance_transform_edt(thresholded_cyto)
    ref_cell_image = -1 * distance
    
    #Generate the labels using watershed with the nuclear segmented lables as seeds
    cell_seg = segmentation.watershed(ref_cell_image, 
                                          markers=nuc_seg, 
                                          mask=thresholded_cyto, compactness=0.1)
    
    return cell_seg




def segment_cells_voronoi( cell_image:np.ndarray,
                            nuc_seg:np.ndarray
                           ):
    
    #process and threshold the cell stain mask to generate the cell mask
    processed_cyto = filters.gaussian(exposure.equalize_adapthist(cell_image, clip_limit=0.1),sigma=3)
    thresholded_cyto = processed_cyto > filters.threshold_li(processed_cyto)
   
    #generate voronoi spaces from nuclear centroids
    features = measure.regionprops_table(nuc_seg,properties=('label','centroid'))
    
    if (len(features['centroid-0'])< 6):    
        print("Too few nuclei for voronoi segmentation using watershed instead")
        cell_seg = segment_cells_watershed(cell_image,nuc_seg)
        
    else:
        vor_image = voronoi.get_voronoi_map(centroids = np.stack((features['centroid-0'],features['centroid-1']),axis=1),
                                           labels = features['label'],
                                           img_height = nuc_seg.shape[0],
                                           img_width = nuc_seg.shape[1])

        #generate cell mask from voronoi image
        cell_seg = np.multiply(vor_image,thresholded_cyto.astype('uint16'))


    return cell_seg




def segment_cells_from_nuclei(image:np.ndarray,
                              channel_map:dict,
                              nuclear_mask:np.ndarray,
                              cyto_stain:int = None,
                              method: str = 'voronoi'):
    '''
    Function to segment cells using nuclear segmentation mask as reference. 
    
    Args:
        image: a multichannel image in CHW format
        channel_map: an annotation dictionary maping the flurophores to the channel index
        cyto_stain: the stain that should be used to generate the cell mask - if set to None, a psuedo stain is generated using all the brainbow flurosphores.
    '''
         
    # if no cytoplasmic stain is present, generate a pesudo stain using the brainbow colors
    if cyto_stain is None:
        #generate a psuedo cell stain image by adding the brainbow FP
        cell_ch = image_utils.create_cell_image_from_brainbow_FP(image,channel_map)
    else:
        cell_ch = image[channel_map[cyto_stain]]
   
    if (method == 'watershed'):
        cell_mask = segment_cells_watershed(cell_ch,nuclear_mask)
    elif (method == 'voronoi'):
        cell_mask = segment_cells_voronoi(cell_ch,nuclear_mask)
    else:
        raise ValueError("method must be one of {}".format(["voronoi","watershed"]))

    return cell_mask
    
    
    
    
    
    
    
    
    
    
    