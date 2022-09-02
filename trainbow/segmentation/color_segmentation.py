# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure,exposure, color
from skimage import segmentation as skseg
from skimage.util import img_as_float
from skimage.future import graph
import cv2 as cv

from trainbow.utils import image_utils, database_utils


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])

def color_seg_slic_rag(image_path,
                       channel_map:dict,
                       slic_compactness = 30,
                       slic_n_seg = 1000,
                       rag_thresh = 12,
                      ):
    
    #read in the image
    image = database_utils.read_image(image_path)
    
    image = image_utils.create_composite_brainbow_image(image, channel_map)
    #Increase the image contrast
    corrected_image = exposure.adjust_gamma(image, gamma =1, gain = 10)
    segments = skseg.slic(corrected_image, compactness=slic_compactness, 
                        n_segments=slic_n_seg, start_label=1)

    #generate RAG and segment 
    g = graph.rag_mean_color(corrected_image, segments)
    seg_rag = graph.merge_hierarchical(segments, g, 
                                       thresh=rag_thresh, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)

    number_of_objects  = np.unique(seg_rag)

    #color the labels
    out = color.label2rgb(seg_rag, corrected_image, kind='avg', bg_label=0)
    out = skseg.mark_boundaries(out, seg_rag, (1,1, 1))

    return (segments,seg_rag,seg_rag,out)