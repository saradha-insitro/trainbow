# -*- coding: utf-8 -*-
'''
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from trainbow.utils import image_utils, database_utils
import trainbow.segmentation.cell_segmentation as seg

def segment_cells(acquisition_df:pd.DataFrame,
                  channel_map:dict,
                  output_dir:str,
                  cytoplasm_stain:str = None,
                  est_nucleus_dia: int = 35,
                  nuc_flowthresh_cp: float = 0.4
                 ):
    '''
    Function that segments images in the acquisition dataframe and saves the segmentation masks in the output directory
    
    Args:
        acquisition_df: acquisition dataframe containing image file path and uid
        channel_map: an annotation dictionary maping the flurophores to the channel index
        output_dir: location where the segmentation masks need to stored
    '''
    
    #setup the paths 
    nuclear_seg_dir = os.path.join(output_dir,"nuc_seg")
    cellular_seg_dir = os.path.join(output_dir,"cell_seg")
   # vormap_seg_dir = os.path.join(output_dir,"voronoi")

    #get uids for all the images in the acquisition df
    image_uids = acquisition_df.uid.unique()

    #loop through all images
    for i in (range(len(image_uids))):
        #get file path
        image_path = acquisition_df[acquisition_df.uid == image_uids[i]].file_path.unique()[0]
   
        #open the image
        image = database_utils.read_image(image_path)
        #segment cells
       # nuc_mask,vor_map,cell_mask = seg.segment_cells_and_nuclei_cp_and_vor(image,channel_map)
        nuc_mask,_,cell_mask = seg.segment_cells_and_nuclei_cp_and_vor(image = image ,
                                                                       channel_map = channel_map,
                                                                       cyto_stain = cytoplasm_stain,
                                                                       estimated_nucleus_dia = est_nucleus_dia,
                                                                       cp_nuc_flowthresh = nuc_flowthresh_cp
                                                                      )
      
        #save the segmentation masks
        database_utils.save_object(nuc_mask, os.path.join(nuclear_seg_dir,(image_uids[i]+".npy")))
        database_utils.save_object(cell_mask, os.path.join(cellular_seg_dir,(image_uids[i]+".npy")))
        #database_utils.save_object(vor_map, os.path.join(vormap_seg_dir,(image_uids[i]+".npy")))
