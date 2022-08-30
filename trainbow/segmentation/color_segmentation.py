
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure,exposure, color
from skimage import segmentation as skseg
from skimage.util import img_as_float
from skimage.future import graph
import cv2 as cv



def image_colorfulness(image):
    # split the image into its respective RGB components
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    
    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))
    
    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
    
    # derive the "colorfulness" metric and return it
    colorfulness = stdRoot + (0.3 * meanRoot)
    
    return colorfulness



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
                       slic_compactness = 30,
                       slic_n_seg = 1000,
                       rag_thresh = 12,
                       plot_image = False
                      ):
    
    #read in the image
    image = open_as_composite_image(image_path)
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

    if plot_image:
        fig = plt.figure(figsize=(8, 2),dpi=300)
        fig.subplots_adjust(hspace = 0, wspace= 0)
        ax1 = fig.add_subplot(141)
        ax2 = fig.add_subplot(142)
        ax3 = fig.add_subplot(143)
        ax4 = fig.add_subplot(144)

        ax1.imshow(corrected_image,alpha=0.9)
        ax1.axis('off')
        ax1.set_title('Composite')

        ax2.imshow(skseg.mark_boundaries(corrected_image, segments,color=[1,1,1]))
        ax2.axis('off')
        ax2.set_title('SLIC')

        ax3.imshow(skseg.mark_boundaries(corrected_image, seg_rag,color=[1,1,1]))
        ax3.axis('off')
        ax3.set_title('RAG')

        ax4.imshow(out)
        ax4.axis('off')
        ax4.set_title('RAGSEg')

        fig.show()