# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import boto3
from skimage import measure
from tqdm import tqdm

import trainbow.utils as utils
from trainbow.features.extract_features import extract_brainbow_features,extract_cell_nuclei_segmetaion_metrics
import multiprocessing as mp

   
def _extract_brainbow_features(args:list):
    if _object_type == "object":
        feat = extract_brainbow_features( uid = args[0],
                                           input_image_path = args[1],
                                           channel_map = _channel_map,
                                           object_type = _object_type,
                                           input_labelled_image_path = args[2]
                                         )
        #save the feature dataset to the given output directory
        utils.database_utils.save_object(feat,args[3])

        return args[3]

    elif _object_type == "image":
        feat = extract_brainbow_features( uid = args[0],
                                           input_image_path = args[1],
                                           channel_map = _channel_map,
                                           object_type = _object_type
                                         )
        return feat

    elif _object_type == "masked_image":
        
        feat = extract_brainbow_features( uid = args[0],
                                           input_image_path = args[1],
                                           channel_map = _channel_map,
                                           object_type = _object_type,
                                           input_labelled_image_path = args[2]
                                         )
    

        return feat

def _extract_segmetation_features(args:list):
    
    #hacky path : to FIX
    feat = extract_cell_nuclei_segmetaion_metrics( uid = args[0],
                                           nuc_mask_path = args[1],
                                           cell_mask_path = args[2]
                                         )
    if feat is None:
        pass
    else:
        #save the feature dataset to the given output directory
        utils.database_utils.save_object(feat,args[3])

    return args[3]    
    
    
def extract_image_features_batch(acquisition_df:pd.DataFrame,
                                   channel_map:dict,
                                   num_cpus:int,
                                  ):
    
    
    #set up channel_map
    global _channel_map
    _channel_map = channel_map
    
    #set up the parallel processing
    num_cpus = num_cpus if num_cpus <= mp.cpu_count() else mp.cpu_count()
    print("Using {} cpus".format(num_cpus))

    #set the paths to images
    image_uids = list(acquisition_df.uid.unique())
    img_file_paths = list(acquisition_df[acquisition_df.uid.isin(image_uids)].file_path)
     
    #compute brainbow features at image level
    print("computing image level brainbow features")
    global _object_type 
    _object_type = "image"
    
    pool = mp.Pool(num_cpus)
    input_output_paths= np.column_stack([image_uids,img_file_paths])
    features = list(tqdm(pool.imap(_extract_brainbow_features, [paths for paths in input_output_paths]), total = len(image_uids)))
    pool.close()
    pool.join()
    
    return features

    
def extract_features_batch(acquisition_df:pd.DataFrame,
                           channel_map:dict,
                           num_cpus:int,
                           output_dir:dir
                          ):
    
    #setup the paths 
    nuclear_seg_dir = os.path.join(output_dir,"nuc_seg")
    cell_seg_dir = os.path.join(output_dir,"cell_seg")
    cell_feat_paths = os.path.join(output_dir,"brainbow_cell_features")
    image_feat_paths = os.path.join(output_dir,"brainbow_image_features")
    seg_metrics_paths = os.path.join(output_dir,"segmentation_metrics_features")

    #set up channel_map
    global _channel_map
    _channel_map = channel_map
    
    #set up the parallel processing
    num_cpus = num_cpus if num_cpus <= mp.cpu_count() else mp.cpu_count()
    print("Using {} cpus".format(num_cpus))

    #set the paths to images
    image_uids = list(acquisition_df.uid.unique())
    img_file_paths = list(acquisition_df[acquisition_df.uid.isin(image_uids)].file_path)
    cell_mask_paths = [os.path.join(cell_seg_dir,(uid + ".npy")) for uid in image_uids]
    nuc_mask_paths = [os.path.join(nuclear_seg_dir,(uid + ".npy")) for uid in image_uids]
   
    #compute segmentation features
#     print("computing segmentation features")
#     pool = mp.Pool(num_cpus)
    # output_paths = [os.path.join(seg_metrics_paths,(uid + ".pkl")) for uid in image_uids]
#     input_output_paths= np.column_stack([image_uids,nuc_mask_paths,cell_mask_paths,output_paths])
#     _ = list(tqdm(pool.imap(_extract_segmetation_features, [paths for paths in input_output_paths]), total = len(image_uids)))
#     pool.close()
#     pool.join()
    
        
#     #compute brainbow features at image level
#     print("computing image level brainbow features")
#     global _object_type 
#     _object_type = "image"
#     pool = mp.Pool(num_cpus)
#     output_paths = [os.path.join(image_feat_paths,(uid + ".pkl")) for uid in image_uids]
#     input_output_paths= np.column_stack([image_uids,img_file_paths,cell_mask_paths,output_paths])
#     feat = list(tqdm(pool.imap(_extract_brainbow_features, [paths for paths in input_output_paths]), total = len(image_uids)))
#     pool.close()
#     pool.join()

    #compute brainbow features at cell level
    print("computing cell level brainbow features")
    _object_type = "object"
    pool = mp.Pool(num_cpus)
    output_paths = [os.path.join(cell_feat_paths,(uid + ".pkl")) for uid in image_uids]
    input_output_paths= np.column_stack([image_uids,img_file_paths,cell_mask_paths,output_paths,])
    _ = list(tqdm(pool.imap(_extract_brainbow_features, [paths for paths in input_output_paths]), total = len(image_uids)))
    pool.close()
    pool.join()
    
    print("Done")