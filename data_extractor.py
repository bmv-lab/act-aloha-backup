'''
File name: data_extractor.py
Description: Extracts image data out of hdf5 file for a specific camera and episode.
'''

import h5py
import os
import cv2
# Open the HDF5 file in read mode
output_directory = 'Low_img_crab'
file_path = "/home/bmv/aloha/src/dataset/Point_Crab_Master/episode_0.hdf5"  # Replace with the path to your HDF5 file
name = 'cam_low'
with h5py.File(file_path, 'r') as hf:
    if 'observations/images' in hf:
        image_group = hf['observations/images']

        # List of camera names
        camera_names = list(image_group.keys())

        for cam_name in camera_names:
            if cam_name == name:
            # Assuming image data is stored as uint8
                images_data = image_group[cam_name][:]

                # Iterate over the images
                for index, image_data in enumerate(images_data):

                    output_path = os.path.join(output_directory, f"Camera_{cam_name}_Image_{index}.png")
                    cv2.imwrite(output_path, image_data)

    else:
        print("Group 'observations/images' not found in the HDF5 file.")