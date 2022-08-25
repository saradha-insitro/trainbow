# -*- coding: utf-8 -*-
"""
Library containing functions that aid in extracting data from insitro's database. Note that these functions are dependent on the core repository. 

"""
from ml_projects.posh.utils import data
from insitro_data.utils import db_utils
import os
import pandas as pd


def read_image(path:str, storage_format:str = "CHW"):
    '''
    Function that reads and returns an image given its path
    
    Args:
        path: path to image
    '''
    
    file_type = os.path.splitext(path)[1]
    
    if (file_type == ".tiff"):
        image = data.read_image(path)
    elif (file_type == ".npy"):
        image = data.read_numpy_image(path,image_storage_format = storage_format)
    else:
        raise Exception("Error reading the image: path does not contain tiff or npy file")
    return image


def create_acquistion_df(plate_id: str,microscope_id: int,measurement_ids:list = None):
    '''
    Function that returns an acquisition dataframe for given plate id and microscope id or measurement id. It queries the cannoical sql database to obtain the measurement ids for a given plate id and microscope id. Alternatively, the user can input the measurement id directly. 
    
    Args:
        plate_id: plate id or barcode
        microscope_id: microcsope number, it is typically 6 for the techdev scope and 2 for the nikon 3 confocal 
        measurement_ids: list containing the measurement uuids. 
    '''
    
    #If the masurement ids are not specified, obtain them. 
    if measurement_ids is None:
        #connect to the sql database 
        engine = db_utils.get_readonly_research_db_engine()
        con = engine.connect()
        # set up sql query 
        query = (f"select * from canonical_image where plate_barcode = "+f"'{plate_id}'"+" AND microscope_id = "+str(microscope_id))
        # extract the data associated with the plate id for that microscope
        experiment = pd.read_sql(query, con)
        #obtain the measurement ids
        measurement_ids = list(experiment.measurement_id.unique())
        #print the number of unique measurement ids found 
        print("For the plate id {pid}, {l} unique measurements were found".format(pid = plate_id, l = len(measurement_ids)))
    
    #if the measurement ids are provided obtain the acquisition dataframe
    if(microscope_id == 6): #tech_dev_scope
        acquisition_df = data.create_multi_acquisition_canonical_dataframe(measurement_ids)
    elif(microscope_id == 2): #nikon 3 confocal
        acquisition_df = data.create_multi_acquisition_nikon_dataframe(measurement_ids)
    else:
        raise Exception("Error obtaining the acquisition dataframe- check microscope id")

    return acquisition_df
                            
