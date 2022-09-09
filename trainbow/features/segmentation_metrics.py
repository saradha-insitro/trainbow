# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import pandas as pd
from skimage import measure


def _compute_label_abundance(instances_in_label: np.ndarray,
                                      obj_label:int):
    """ 
    Args:
        instances_in_label  : intensity image
        obj_label : cell label level
    """

    # get all the intances in the image
    intance_in_label_count = list(instances_in_label[instances_in_label>0].flatten())
    temp = {x:intance_in_label_count.count(x) for x in intance_in_label_count}
    
    #remove instances less than or equal to 5 pixels
    instances_in_label = list({k: v for k, v in temp.items() if v >= 10})
    instance_number_same_as_label = obj_label in instances_in_label

    #compute features
    feat = { "label": obj_label,
            "num_of_instances_in_label": len(instances_in_label),
             "instance_number_same_as_label": instance_number_same_as_label
           }
    
    return feat


def _compute_consistency_between_segmentation_masks(base_segmentation_mask: np.ndarray,
                                                    comp_segmentation_mask: np.ndarray):


    #get the regionprops of base mask with comparison(comp) masks
    props = measure.regionprops(base_segmentation_mask, comp_segmentation_mask)
    
    #compute the number of segmentatations instances in within the each object in base labeled image 
    feat = pd.DataFrame() # initialise empty dataframe
    for flag in (range(len(props))):
        features = pd.concat(
                    [pd.DataFrame([flag+1], columns=["object_label"]),
                     pd.DataFrame([_compute_label_abundance( 
                                                             instances_in_label = props[flag].intensity_image, 
                                                             obj_label = props[flag].label)]).reset_index(drop=True)
                    ],axis=1)
        feat = pd.concat([feat,features], ignore_index = True)
    
    return feat

def evaluate_cell_nuclei_segmentation(cell_mask: np.ndarray,
                                      nuc_mask: np.ndarray):
    
    
    if (np.max(cell_mask) == 0) | (np.max(nuc_mask) == 0):
        pass 
    else:
        nuclei_over_cells_feat = _compute_consistency_between_segmentation_masks(cell_mask,nuc_mask)

        cells_over_nuclei_feat = _compute_consistency_between_segmentation_masks(nuc_mask,cell_mask)

        #compute overlapping metrics
        num_of_cells = nuclei_over_cells_feat.shape[0]
        num_of_cells_with_no_nuc = np.sum(nuclei_over_cells_feat.num_of_instances_in_label==0)

        num_of_cells_with_mismatched_nuc_labels = np.sum(nuclei_over_cells_feat[
            nuclei_over_cells_feat.instance_number_same_as_label<1].num_of_instances_in_label>0)

        num_of_cells_with_more_than_one_nuc = np.sum(nuclei_over_cells_feat.num_of_instances_in_label>1)
        well_matched_cells_nuc = nuclei_over_cells_feat[(nuclei_over_cells_feat.instance_number_same_as_label == True) & 
                                                       (nuclei_over_cells_feat.num_of_instances_in_label ==1)].shape[0]

        num_of_nuclei = cells_over_nuclei_feat.shape[0]

        num_of_nuc_with_no_cells = np.sum(cells_over_nuclei_feat.num_of_instances_in_label==0)


        features = {"num_of_cells" : num_of_cells,
                    "num_of_nuclei" : num_of_nuclei,
                    "num_of_cells_with_no_nuc" : num_of_cells_with_no_nuc,
                    "num_of_nuc_with_no_cells" : num_of_nuc_with_no_cells,
                    "num_of_cells_with_mismatched_nuc_labels" : num_of_cells_with_mismatched_nuc_labels,
                    "num_of_cells_with_more_than_one_nuc" : num_of_cells_with_more_than_one_nuc,
                    "num_of_well_matched_cells_nuc" : well_matched_cells_nuc
                    }
        return pd.DataFrame([features])


        
        
        
        
        
        
        
        
        