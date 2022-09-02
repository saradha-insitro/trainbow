# -*- coding: utf-8 -*-
'''
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from trainbow.utils import image_utils, database_utils
import trainbow.segmentation.cell_segmentation as seg
from tqdm import tqdm 
from skimage import filters
import multiprocessing as mp



_dapi_index = 1



def segment_nuclei(args:list):
    
    #hacky path : to FIX
    input_image_path = args[0]
    output_image_path = args[1]
    
    #open the image
    image = database_utils.read_image(input_image_path)
    processed_nuc = filters.gaussian(image[_dapi_index],sigma=2) #smoothen the image using a gaussian filter 
    nuc_mask = seg.segment_nuclei_cellpose([processed_nuc])[0] #obtain nuc labelled image
    
    #save image
    database_utils.save_object(nuc_mask, output_image_path)

    return output_image_path
    
def segment_cells(args:list):
    
    #hacky path : to FIX
    input_image_path = args[0]
    input_nuc_mask_path = args[1]
    output_image_path = args[2]
    
    #open the image
    image = database_utils.read_image(input_image_path)
    nuc_mask = database_utils.read_image(input_nuc_mask_path)

    cell_mask = seg.segment_cells_from_nuclei(image,_channel_map,nuc_mask)
        
    #save image
    database_utils.save_object(cell_mask, output_image_path)

    return output_image_path


def segment_nuclei_batch(acquisition_df:pd.DataFrame,
                         channel_map:dict,
                         num_cpus:int,
                         output_dir:dir
                          ):
    
    #setup the paths 
    nuclear_seg_dir = os.path.join(output_dir,"nuc_seg")

    #set up DAPI index 
    global _dapi_index
    _dapi_index = channel_map['DAPI']

    #set up the parallel processing
    num_cpus = num_cpus if num_cpus <= mp.cpu_count() else mp.cpu_count()
    print("Using {} cpus".format(num_cpus))
    
    #set the paths to images
    image_uids = acquisition_df.uid.unique()
    file_paths = list(acquisition_df[acquisition_df.uid.isin(image_uids)].file_path)
    output_seg_imag_paths = [os.path.join(nuclear_seg_dir,(uid+".npy")) for uid in image_uids]
    
    
    #segment and save 
    input_output_paths= np.column_stack([file_paths,output_seg_imag_paths])
    pool = mp.Pool(num_cpus)
    nuc_mask_dir = pool.map(segment_nuclei, [paths for paths in input_output_paths])
     
    return nuc_mask_dir
    

def segment_cells_from_nuclei_batch(acquisition_df:pd.DataFrame,
                                     channel_map:dict,
                                     num_cpus:int,
                                     output_dir:dir
                                      ):
    
    #setup the paths 
    nuclear_seg_dir = os.path.join(output_dir,"nuc_seg")
    cell_seg_dir = os.path.join(output_dir,"cell_seg")

    #set up channel_map
    global _channel_map
    _channel_map = channel_map

    
    #set up the parallel processing
    num_cpus = num_cpus if num_cpus <= mp.cpu_count() else mp.cpu_count()
    print("Using {} cpus".format(num_cpus))
    
    #set the paths to images
    image_uids = acquisition_df.uid.unique()
    image_file_paths = list(acquisition_df[acquisition_df.uid.isin(image_uids)].file_path)
    nuc_mask_paths = [os.path.join(nuclear_seg_dir,(uid + ".npy")) for uid in image_uids]
    output_seg_image_paths = [os.path.join(cell_seg_dir,(uid+".npy")) for uid in image_uids]
    
    
    #segment and save 
    input_output_paths= np.column_stack([image_file_paths,nuc_mask_paths,output_seg_image_paths])
    pool = mp.Pool(num_cpus)
    cell_mask_dir = pool.map(segment_cells, [paths for paths in input_output_paths])
     
    return cell_mask_dir