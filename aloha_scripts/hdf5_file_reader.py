import h5py
import cv2
import numpy as np
# Open the HDF5 file in read mode
file_path = '/home/bmv/aloha/src/dataset/Point_Crab_v2/episode_0.hdf5'  # Replace with the path to your HDF5 file
with h5py.File(file_path, 'r') as file:
    # List the top-level groups in the HDF5 file
    top_level_groups = list(file.keys())
    print(f'Top-level groups: {top_level_groups}')

    # Access a specific group and its datasets
    group_name = 'observation'  # Replace with the name of the group you want to access
    if group_name in file and group_name == 'action':
        group = file[group_name]
        print(group)
        data = group[()]
        print("Action data: \n",data)

        # Access a specific dataset and read its data
    else:
        group_obs = file['observations']
        dataset_name =  list(group_obs.keys()) # Replace with the name of the dataset you want to read
        print(dataset_name)
        for i in dataset_name:
            if i != 'images':
                dataset = group_obs[i]
                data = dataset[()]  # Read the entire dataset into a NumPy array
                print('{} \n'.format(i))
                print(data)
            else:
                for img in range(300):
                    image_dict = dict()
                    for cam_name in ['cam_high', 'cam_low', 'cam_wrist']:
                        image_dict[cam_name] = file[f'/observations/images/{cam_name}'][img]
                    stacked = np.hstack((image_dict['cam_high'],image_dict['cam_low'],image_dict['cam_wrist']))
                    # stacked = image_dict['cam_wrist']
                    while True:
                        cv2.imshow('img',stacked)
                        if cv2.waitKey(0) == ord('q'):
                            break
                    cv2.destroyAllWindows()
                

# The HDF5 file is automatically closed when exiting the 'with' block
