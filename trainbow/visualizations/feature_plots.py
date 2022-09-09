# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trainbow.utils import image_utils, database_utils, plot_utils


_bb_color_features = ['eGFP_int_sum','eGFP_int_min','eGFP_int_d25','eGFP_int_median',
                      'eGFP_int_d75','eGFP_int_max','eGFP_int_mean','eGFP_int_mode',
                      'eGFP_int_sd','eGFP_kurtosis','eGFP_skewness','eGFP_entropy',
                      
                      'mOrange_int_sum', 'mOrange_int_min','mOrange_int_d25','mOrange_int_median',
                      'mOrange_int_d75','mOrange_int_max','mOrange_int_mean','mOrange_int_mode',
                      'mOrange_int_sd','mOrange_kurtosis','mOrange_skewness','mOrange_entropy',
                      
                      'mKate_int_sum','mKate_int_min','mKate_int_d25','mKate_int_median',
                      'mKate_int_d75','mKate_int_max','mKate_int_mean','mKate_int_mode',
                      'mKate_int_sd','mKate_kurtosis','mKate_skewness','mKate_entropy',
                      
                      'mKate_by_mOrange_mean','mKate_by_eGFP_mean','eGFP_by_mOrange_mean',
                      'mKate_by_mOrange_sd','mKate_by_eGFP_sd','eGFP_by_mOrange_sd',
                      
                      'eom_colorfulness','emo_colorfulness','meo_colorfulness',
                      'moe_colorfulness','ome_colorfulness','oem_colorfulness',
                      
                      'hue_int_sum','hue_int_min','hue_int_d25','hue_int_median',
                      'hue_int_d75','hue_int_max','hue_int_mean','hue_int_mode',
                      'hue_int_sd','hue_kurtosis','hue_skewness','hue_entropy',
                      
                      'saturation_int_sum','saturation_int_min','saturation_int_d25','saturation_int_median',
                      'saturation_int_d75','saturation_int_max','saturation_int_mean','saturation_int_mode',
                      'saturation_int_sd','saturation_kurtosis','saturation_skewness','saturation_entropy',
                      
                      'value_int_sum','value_int_min','value_int_d25','value_int_median',
                      'value_int_d75','value_int_max','value_int_mean','value_int_mode',
                      'value_int_sd','value_kurtosis','value_skewness','value_entropy']


_morphology_features = ['area','perimeter','bbox_area','convex_area','equivalent_diameter',
                        'major_axis_length','minor_axis_length','eccentricity','concavity','solidity',
                        'a_r','shape_factor','area_bbarea']

_cre_conc = ['0ng','250ng','500ng','1000ng']
_NF_protocol = ['CA137','CM137','CD118','CM130']


def plt_pca_NF_optimization():
    pass


def plot_heatmap_NF_optimization(feature: str,
                                 dataset:pd.DataFrame,
                                 well_anno: dict,
                                 norm: bool = True
                                ):
    

    agg_data = dataset.groupby(['well_loc'],as_index=False)[feature].median()

    population = pd.DataFrame()
    for sample in list(well_anno.keys()):
        population  = pd.concat([population,
                                pd.DataFrame([{'sample' : sample,
                                              'feature' : np.mean(agg_data[agg_data.well_loc.isin(well_anno[sample])][feature])
                                             }]).reset_index(drop=True)])

    mat = np.array(list(population.iloc[1:17]['feature'])).reshape(-1, 4)
    matrix = pd.DataFrame(mat, columns=_cre_conc, index= _NF_protocol)
    if norm:
        normalised_matrix = (matrix.T/list(np.sum(matrix,axis = 1))).T
        plot_utils.plot_heatmap(normalised_matrix.round(2), feature)
    else:
        plot_utils.plot_heatmap(matrix.round(2), feature)

    