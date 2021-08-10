import os
import cv2
import numpy as np
from PIL import Image
from GPSPhoto import gpsphoto


#Metadata Globals
DJI_PREFIX = "drone-dji"
TIFF_PREFIX = "tiff"
MARKER_URI = "http://ns.adobe.com/xap/1.0/"

DJI_DICT = {"RelativeAltitude": "relative altitude", 
            "GimbalRollDegree": "gimbal roll", 
            "GimbalYawDegree": "gimbal yaw", 
            "GimbalPitchDegree": "gimbal pitch", 
            "FlightRollDegree": "flight roll",
            "FlightYawDegree": "flight yaw", 
            "FlightPitchDegree": "flight pitch"
}

TIFF_DICT = {"Model": "model"}

METADATA_LIST = ["latitude", "longitude", "relative altitude", "flight yaw", "flight pitch", "flight roll", "gimbal yaw", "gimbal pitch", "gimbal roll", "model", "name"]

"""
Inputs:
- filepath: The filepath of the metadata file.
            Each line of the file represents an image.
            The metadata includes data listed in the order
            specified in METADATA_LIST seperated by spaces.

Outputs:
- md_dict: A dictionary with metadata string as keys and
           lists as vaues in which the ith index of the list
           is the value of metadata string for the ith image.
"""
def get_metadata(img_dir_path):
    md_dict = {md:[] for md in METADATA_LIST}
    for img_name in os.listdir(img_dir_path):
        if ".JPG" in img_name or ".jpg" in img_name: 
            img_path = os.path.join(img_dir_path, img_name)
            
            #Get name
            md_dict["name"].append(img_name)
            
            #Get GPS Data 
            data = gpsphoto.getGPSData(img_path)

            lat, lng = data["Latitude"], data["Longitude"]
            md_dict["latitude"].append(lat)
            md_dict["longitude"].append(lng)

            #Iterate through XMP data 
            with Image.open(img_path) as im:
                for segment, content in im.applist:
                    marker, body = content.split('\x00'.encode(), 1)
                    if segment == 'APP1' and marker == MARKER_URI.encode():
                        lines = body.decode().split("\n")
                        for line in lines:
                            strip_line = line.strip()
                            
                            #Get drone orientation 
                            if strip_line.startswith(DJI_PREFIX):
                                rev_strip_line = strip_line[len(DJI_PREFIX) + 1:]
                                key, val = tuple(rev_strip_line.split("="))
                                val = float(val.strip('"'))
                                if key in DJI_DICT.keys():
                                    rev_key = DJI_DICT[key]
                                    md_dict[rev_key].append(val)
                            
                            #Get drone model 
                            if strip_line.startswith(TIFF_PREFIX):
                                rev_strip_line = strip_line[len(TIFF_PREFIX) + 1:]
                                key, val = tuple(rev_strip_line.split("="))
                                val = val.strip('"')
                                if key in TIFF_DICT.keys():
                                    rev_key = TIFF_DICT[key]
                                    md_dict[rev_key].append(val)
        
    return md_dict

"""
Inputs:
- Model: The model of the drone.

Outputs:
- sensor_dim: Tuple representing the width and height of the camera sensor in mm.
- focal length: The focal length of the camera in mm.
"""
def get_camera_specs(model):
    #Mavic Mini
    if model == "FC7203":
        return (.63, .47), .449
    elif model == "FC330":
        return (1.32, .88), .88
    else:
        raise ValueError("Unrecognized model number")

"""
Inputs:
- base_dir: The root directory of the images.
- name_list: A list of the files of the images in the directory.

Outputs:
- img_list: List of images ordered by name list
"""
def get_img_list(base_dir, name_list):

    path_list = [os.path.join(base_dir, name) for name in name_list]

    img_list = []
    for path in path_list:
        img = cv2.imread(path)
        img_list.append(img)

    return img_list