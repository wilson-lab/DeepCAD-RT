from deepcad.test_collection import testing_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename,download_demo
import pdb
import os

# %% First setup some parameters for testing
datasets_path = '/n/data1/hms/neurobio/wilson/DeepCAD_datasets/20230811-3_EK021_7f_shade_cone_bright/time_slices_ch1_trial_001/'  # folder containing tif files for testing
pth_dir = '/n/data1/hms/neurobio/wilson/DeepCAD_datasets/20230811-1_EK021_7f_infinite_bar_methyl_salicylate/rt/pth/time_slices_ch1_trial_001_202308260929/' #'/n/data1/hms/neurobio/wilson/DeepCAD_datasets/20230811-1_EK021_7f_infinite_bar_methyl_salicylate/rt/pth/'
denoise_model = 'best_model'  # A folder containing pth models to be tested
test_datasize = 10000000              # the number of frames to be tested (test all frames if the number exceeds the total number of frames in a .tif file)
GPU = '0'                             # the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')
patch_xy = 15                         # the width and height of 3D patches
patch_t = 15                          # the time dimension of 3D patches
overlap_factor = 0.6                  # the overlap factor between two adjacent patches. 
                                      # Since the receptive field of 3D-Unet is ~90, seamless stitching requires an overlap (patch_xyt*overlap_factor）of at least 90 pixels.
num_workers = 1                       # if you use Windows system, set this to 0.

# %% Make the subfolders we will need
output_dir = os.path.join(datasets_path,'rt','test')
os.makedirs(output_dir)

# %% Setup some parameters for result visualization during testing period (optional)
visualize_images_per_epoch = False  # choose whether to display inference performance after each epoch
save_test_images_per_epoch = True  # choose whether to save inference image after each epoch in pth path

# %% Play the demo noise movie (optional)
# playing the first noise movie using opencv.
display_images = False

if display_images:
    display_filename = get_first_filename(datasets_path)
    print('\033[1;31mDisplaying the first raw file -----> \033[0m')
    print(display_filename)
    display_length = 500  # the frames number of the noise movie
    # normalize the image and display
    display(display_filename, display_length=display_length, norm_min_percent=0.5, norm_max_percent=99.8)

test_dict = {
    # dataset dependent parameters
    'patch_x': patch_xy,
    'patch_y': patch_xy,
    'patch_t': patch_t,
    'overlap_factor':overlap_factor,
    'scale_factor': 1,                   # the factor for image intensity scaling
    'test_datasize': test_datasize,
    'datasets_path': datasets_path,
    'pth_dir': pth_dir,                 # pth file root path
    'denoise_model' : denoise_model,
    'output_dir' : output_dir,     # result file root path
    # network related parameters
    'fmap': 16,                          # the number of feature maps
    'GPU': GPU,
    'num_workers': num_workers,
    'visualize_images_per_epoch': visualize_images_per_epoch,
    'save_test_images_per_epoch': save_test_images_per_epoch,
    'select_img_num': 1000000
}
# %%% Testing preparation
# first we create a testing class object with the specified parameters
tc = testing_class(test_dict)
# start the testing process
tc.run()
