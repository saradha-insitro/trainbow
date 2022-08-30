# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from cellpose import models

from skimage import filters, measure,exposure

from trainbow.utils import image_utils, database_utils,voronoi

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


def segment_cells_and_nuclei_cp_and_vor( image:np.ndarray,
                                         channel_map:dict,
                                         cyto_stain:int = None,
                                         estimated_nucleus_dia:int = 35,
                                         cp_nuc_flowthresh:float =0.4,
                                        cp_use_gpu:bool = False
                                       ):
    '''
    Function to segment cells and nuclei. This is achieved by first generating nuclei labelled masks using cellpose's default model, then the nuclear centroids are used to generate labelled voronoi spaces and the cell masks using cell paint or generating using the brainbow flurophores are then used generate cell mask. 
    
    Args:
        image: a multichannel image in CHW format
        channel_map: an annotation dictionary maping the flurophores to the channel index
        cyto_stain: the stain that should be used to generate the cell mask - if set to None, a psuedo stain is generated using all the brainbow flurosphores.
        nuc_dia: estimated diameter of the nuclei to be inputed to the cell pose model. Set to None if unknown. 
        cp_flowthresh: cell pose's flow threshold. Increase this value if there is the objects are underdetected
    '''
    
    #process and segment the nuclear image
    processed_nuc = filters.gaussian(image[channel_map['DAPI']],sigma=2) #smoothen the image using a gaussian filter 
    nuc_mask = segment_nuclei_cellpose([processed_nuc],estimated_nucleus_dia,cp_nuc_flowthresh,
                                       use_gpu = cp_use_gpu
                                      )[0] #obtain nuc labelled image

         
    # if no cytoplasmic stain is present, generate a pesudo stain using the brainbow colors
    if cyto_stain is None:
        #generate a psuedo cell stain image by adding the brainbow FP
        cell_ch = image_utils.create_cell_image_from_brainbow_FP(image,channel_map)
    else:
        cell_ch = image[channel_map[cyto_stain]]
   
    #process and threshold the pseudo cell stain mask to generate the cell mask
    processed_cyto = filters.gaussian(exposure.equalize_adapthist(cell_ch, clip_limit=0.1),sigma=3)
    thresholded_cyto = processed_cyto > filters.threshold_li(processed_cyto)
  

    #generate voronoi spaces from nuclear centroids
    features=measure.regionprops_table(nuc_mask,properties=('label','centroid'))
    
    vor_image=voronoi. get_voronoi_map(centroids = np.stack((features['centroid-0'],features['centroid-1']),axis=1),
                                       labels = features['label'],
                                       img_height = nuc_mask.shape[0],
                                       img_width = nuc_mask.shape[1])
    
    #generate cell mask from voronoi image
    cell_mask = np.multiply(vor_image,thresholded_cyto.astype('uint8'))

    
    return (nuc_mask,vor_image,cell_mask)
    
    
    
    
    
    
    
    
    
    
    