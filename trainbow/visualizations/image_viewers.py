# -*- coding: utf-8 -*-
'''


'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trainbow.utils import image_utils, database_utils
from skimage import exposure
from skimage.color import rgb2hsv


def plot_random_images_from_sample(sample: str,
                                         acquisition_df:pd.DataFrame,
                                         well_anno: dict, 
                                         channel_map: dict,
                                         number_of_img:int = 10,
                                         img_size:int = 2,
                                         img_dpi:int = 300,
                                         normalise_comp_image:bool = True,
                                         equalise_img:bool = False,
                                         comp_img_gain:int = 10,
                                         comp_img_alpha:float = 0.9,
                                         comp_img_gamma: int = 8,
                                         max_num_of_images_per_row:int = 5
                                   
                                        ):
    '''
    Function to visualise n brainbow images randomly sampled (with replacement) for a given sample
    
    Args:
        sample: the sample to be sampled from
        acquisition_df: pandas aquisition dataframe containing the image paths and the well locations
        well_anno: an annotation dictionary maping the samples to the wells        
        channel_map: an annotation dictionary maping the flurophores to the channel index
        number_of_img: number of images to randomly choose per sample
        img_size: the size of one image in a plot
        img_dpi: resolution (dpi) of the plot
        normalise_comp_image: should the channels of each image be normalised- note that this will give weird images if set to true and there is no signal in the image
        comp_img_gain: digital brightness gain -> see sklearn's exposure.adjust gamma for more details
        comp_img_alpha: blending/transparency of channels -> see matplotlib's imshow for more details
        comp_img_gamma: digital contrast -> see sklearn's exposure.adjust gamma for more details
        max_num_of_images_per_row: a number specifying the maximum number of columns in the image matrix


    '''
    
    if sample not in well_anno:
        raise Exception("Error: sample not in given annotated well map") 
    
    #set the layout of the image
    nrow = int(np.ceil(number_of_img/max_num_of_images_per_row))
    ncol = number_of_img if nrow <= 1 else max_num_of_images_per_row
    
    fig, axs = plt.subplots(nrow,ncol, dpi = img_dpi, figsize=(img_size*ncol,img_size*nrow))
    fig.subplots_adjust(hspace = 0.1, wspace= 0.1)

    for ax in axs.ravel():
        #select a random image in a random well for a given sample
        well = np.random.choice(well_anno[sample])
        image_path = np.random.choice(acquisition_df[acquisition_df.well_loc == well].file_path)

        # read in the image
        image = database_utils.read_image(image_path)

        #create composite image
        bb_comp = image_utils.create_composite_brainbow_image(image, channel_map, normalise_comp_image)
        
        # visualise image
        if equalise_img:
            img_t = np.stack([exposure.equalize_adapthist(bb_comp[:,:,0], clip_limit=0.3),
                              exposure.equalize_adapthist(bb_comp[:,:,1], clip_limit=0.3),
                              exposure.equalize_adapthist(bb_comp[:,:,2], clip_limit=0.3)],axis = 2)
            ax.imshow(exposure.adjust_gamma(img_t,gamma = comp_img_gamma))
        else:
            ax.imshow(exposure.adjust_gamma(bb_comp, gamma =1, gain = comp_img_gain),alpha = comp_img_alpha)


        ax.axis('off')
    
    fig.suptitle(sample,fontweight='bold')
    fig.show()

def plot_random_hsv_images_from_sample(sample: str,
                                         acquisition_df:pd.DataFrame,
                                         well_anno: dict, 
                                         channel_map: dict,
                                         number_of_img:int = 10,
                                         img_size:int = 2,
                                         img_dpi:int = 300,
                                         normalise_comp_image:bool = True,
                                         max_num_of_images_per_row:int = 5
                                   
                                        ):
    '''
    Function to visualise n brainbow images randomly sampled (with replacement) for a given sample
    
    Args:
        sample: the sample to be sampled from
        acquisition_df: pandas aquisition dataframe containing the image paths and the well locations
        well_anno: an annotation dictionary maping the samples to the wells        
        channel_map: an annotation dictionary maping the flurophores to the channel index
        number_of_img: number of images to randomly choose per sample
        img_size: the size of one image in a plot
        img_dpi: resolution (dpi) of the plot
        normalise_comp_image: should the channels of each image be normalised- note that this will give weird images if set to true and there is no signal in the image
        max_num_of_images_per_row: a number specifying the maximum number of columns in the image matrix


    '''
    
    if sample not in well_anno:
        raise Exception("Error: sample not in given annotated well map") 
    
    #set the layout of the image
    nrow = int(np.ceil(number_of_img/max_num_of_images_per_row))
    ncol = number_of_img if nrow <= 1 else max_num_of_images_per_row
    
    fig, axs = plt.subplots(nrow,ncol, dpi = img_dpi, figsize=(img_size*ncol,img_size*nrow))
    fig.subplots_adjust(hspace = 0.1, wspace= 0.1)

    for ax in axs.ravel():
        #select a random image in a random well for a given sample
        well = np.random.choice(well_anno[sample])
        image_path = np.random.choice(acquisition_df[acquisition_df.well_loc == well].file_path)

        # read in the image
        image = database_utils.read_image(image_path)

        #create composite image
        bb_comp = image_utils.create_composite_brainbow_image(image, channel_map, normalise_comp_image)
        
        hsv_image = rgb2hsv(bb_comp) # convert rgb to hsv

        # visualise image
        ax.imshow(hsv_image)
        ax.axis('off')
    
    fig.suptitle(sample,fontweight='bold')
    fig.show()


def plot_random_images_from_all_samples(acquisition_df:pd.DataFrame,
                                        well_anno: dict,
                                        channel_map: dict,
                                        number_of_img:int = 10,
                                        img_size:int = 2,
                                        img_dpi:int = 300,
                                        normalise_comp_image:bool = True,
                                        comp_img_gain:int = 10,
                                        comp_img_alpha:float = 0.9,
                                        max_num_of_images_per_row:int = 5
                                             ):
    '''
    Function to visualise n brainbow images randomly sampled (with replacement) for all samples. 
    
    Args:
        acquisition_df: pandas aquisition dataframe containing the image paths and the well locations
        well_anno: an annotation dictionary maping the samples to the wells        
        channel_map: an annotation dictionary maping the flurophores to the channel index
        number_of_img: number of images to randomly choose per sample
        img_size: the size of one image in a plot
        img_dpi: resolution (dpi) of the plot
        normalise_comp_image: should the channels of each image be normalised- note that this will give weird images if set to true and there is no signal in the image
        comp_img_gain: digital brightness gain -> see sklearn's exposure.adjust gamma for more details
        comp_img_alpha: blending/transparency of channels -> see matplotlib's imshow for more details
        max_num_of_images_per_row: a number specifying the maximum number of columns in the image matrix

       
    '''
    
    for sample in list(well_anno.keys()):
        plot_random_images_from_sample(sample = sample,
                                             acquisition_df = acquisition_df,
                                             well_anno = well_anno,
                                             channel_map = channel_map,
                                             number_of_img = number_of_img,
                                             img_size = img_size,
                                             img_dpi = img_dpi,
                                             normalise_comp_image = normalise_comp_image,
                                             comp_img_gain = comp_img_gain,
                                             comp_img_alpha = comp_img_alpha
                                            )

        
        
def plot_brainbow_FP_expression_across_samples(acquisition_df:pd.DataFrame,
                                               well_anno: dict,
                                               channel_map: dict,
                                               img_size:int = 2,
                                               img_dpi:int = 300,
                                               max_num_of_images_per_row:int = 6):
    '''
    Plot the expression of brainbow fluroscent protiens (FP) across all conditions ensuring that the LUTs are consistent across image. 
    
    Args:
        acquisition_df: pandas aquisition dataframe containing the image paths and the well locations
        well_anno: an annotation dictionary maping the samples to the wells
        channel_map: an annotation dictionary maping the flurophores to the channel index
        img_size: the size of one image in a plot
        max_num_of_images_per_row: a number specifying the maximum number of columns in the image matrix
    
    '''
    #Maximum instensity values
    rmax = 0
    gmax = 0
    bmax = 0

    images = {} # initialise an empty dict

    for sample in list(well_anno.keys()):
        #select a random image in a random well for a given sample
        image_path = np.random.choice(acquisition_df[acquisition_df.well_loc == np.random.choice(well_anno[sample])].file_path)
        # read in the image
        image = database_utils.read_image(image_path)
        #create brainbow composite image
        comp = image_utils.create_composite_brainbow_image(image,channel_map,normalize = False)
        #asess the maximum intensity values for the channels
        rmax = max(rmax,np.max(comp[:,:,0]))
        gmax = max(gmax,np.max(comp[:,:,1]))
        bmax = max(bmax,np.max(comp[:,:,2]))
        images[sample] = comp


    #set the layout of the image
    nrow = int(np.ceil(len(list(well_anno.keys()))/max_num_of_images_per_row))
    ncol = len(list(well_anno.keys())) if nrow <= 1 else max_num_of_images_per_row
    print(nrow,ncol)


    # visualise the expression of each flurophore for each sample
    #mKate2
    fig, axs = plt.subplots(nrow,ncol, dpi = img_dpi, figsize=(img_size*ncol,img_size*nrow))
    fig.subplots_adjust(hspace = 0.2, wspace= 0.1)
    for ax,sample in zip(axs.ravel(),list(well_anno.keys())):
        ax.imshow(images[sample][:,:,0],vmin =0, vmax = int(0.7*rmax),cmap ="Reds_r")
        ax.axis('off')
        ax.set_title(sample, fontsize=8)
    #fig.suptitle("mKate2",fontweight='bold')
    fig.show()

    #mOrange
    fig, axs = plt.subplots(nrow,ncol, dpi = img_dpi, figsize=(img_size*ncol,img_size*nrow))
    fig.subplots_adjust(hspace = 0.2, wspace= 0.1)
    for ax,sample in zip(axs.ravel(),list(well_anno.keys())):
        ax.imshow(images[sample][:,:,1],vmin =0, vmax = int(0.7*gmax),cmap ="Greens_r")
        ax.axis('off')
        ax.set_title(sample, fontsize=8)
    #fig.suptitle("mOrange",fontweight='bold')
    fig.show()

    #eGFP
    fig, axs = plt.subplots(nrow,ncol, dpi = img_dpi, figsize=(img_size*ncol,img_size*nrow))
    fig.subplots_adjust(hspace = 0.2, wspace= 0.1)
    for ax,sample in zip(axs.ravel(),list(well_anno.keys())):
        ax.imshow(images[sample][:,:,2],vmin =0, vmax = int(0.7*bmax),cmap ="Blues_r")
        ax.axis('off')
        ax.set_title(sample, fontsize=8)
    #fig.suptitle("eGFP",fontweight='bold')
    fig.show()

      
        
        
def plot_brainbow_composite_image_channelwise(image:np.ndarray,
                                             exp_gain:int = 10
                                            ):
    '''
    Plot the composite and individual channels of a given brainbow composite image
    Args:
        image: composite brainbow image
        exp_gain: digital gain to apply to alter the brightness of an image- ref to skimage.exposure.adjust_gamma
    
    '''

    fig = plt.figure(figsize=(8, 2),dpi=300)
    fig.subplots_adjust(hspace = 0, wspace= 0)
    ax1 = fig.add_subplot(141)
    ax2 = fig.add_subplot(142)
    ax3 = fig.add_subplot(143)
    ax4 = fig.add_subplot(144)

    ax1.imshow(exposure.adjust_gamma(image, gamma =1, gain = exp_gain),alpha=0.9)
    ax1.axis('off')
    ax1.set_title('Composite')

    ax2.imshow(exposure.adjust_gamma(image[:,:,2], gamma =1, gain = exp_gain),cmap='Blues_r')
    ax2.axis('off')
    ax2.set_title('eGFP')

    ax3.imshow(exposure.adjust_gamma(image[:,:,1], gamma =1, gain = exp_gain),cmap='Greens_r')
    ax3.axis('off')
    ax3.set_title('mOrange')

    ax4.imshow(exposure.adjust_gamma(image[:,:,0], gamma =1, gain = exp_gain),cmap='Reds_r')
    ax4.axis('off')
    ax4.set_title('mKate2')

    fig.show()
    