import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trainbow.utils as utils
import trainbow.visualizations.image_viewers as viz
import trainbow.batch.segmenters as seg
import os


plate_id = 'PB2454' #plate id
microscope_id = 6 # techdev scope or nikon 4
#Fluroscent map
channel_map ={
    'DAPI':1,
    'eGFP':2,
    'mOrange':0,
    'mKate2':3,
    'DPC_top':4,
    'DPC_bottom':5,
    'DPC_left':6,
    'DPC_right':7
}

well_map = {
            "Mix" : ['A01','A02'],
            "Control" : ['A03','A04'],
    
            "CA137_0ng" : ['B01'],
            "CA137_250ng" : ['B02'],
            "CA137_500ng" : ['B03'],
            "CA137_1000ng" : ['B04'],
            
            "CM137_0ng" : ['C01'],
            "CM137_250ng" : ['C02'],
            "CM137_500ng" : ['C03'],
            "CM137_1000ng" : ['C04'],
            
}

bucket = 's3://insitro-user/'
output_dir = os.path.join('saradha/',plate_id)

experiment_acquisition = utils.database_utils.create_acquistion_df(plate_id,microscope_id)
#drop duplicate file paths - keeping only the last row
experiment_acquisition = experiment_acquisition.drop_duplicates(subset='file_path', keep="last")

## instance segmentation 
nuc_mask_paths = seg.segment_nuclei_batch(acquisition_df = experiment_acquisition,
                                          channel_map = channel_map,
                                          num_cpus = 10,
                                          output_dir = os.path.join(bucket,output_dir)
                                          );

cell_mask_paths = seg.segment_cells_from_nuclei_batch(acquisition_df = experiment_acquisition,
                                                      channel_map = channel_map,
                                                      num_cpus = 10,
                                                      output_dir = os.path.join(bucket,output_dir))

#compute trainbow features at the cellular level
feat_paths = extract_features_batch(acquisition_df = experiment_acquisition,
                                  channel_map = channel_map,
                                  num_cpus = 10,
                                  output_dir = os.path.join(bucket,output_dir))
features = pd.concat([utils.database_utils.load_obj(os.path.join(bucket,path)) 
              for path in utils.database_utils.get_file_list(os.path.join(output_dir,"brainbow_features"))])
utils.database_utils.save_object(features,os.path.join(os.path.join(bucket,output_dir),"cellular_brainbow_features.pkl"))


#compute trainbow features at the level
image_features = extract_features_batch_FOV(acquisition_df = experiment_acquisition,
                                      channel_map = channel_map,
                                      num_cpus = 10,
                                      output_dir = os.path.join(bucket,output_dir))
utils.database_utils.save_object(image_features,os.path.join(os.path.join(bucket,output_dir),"image_brainbow_features.pkl"))
