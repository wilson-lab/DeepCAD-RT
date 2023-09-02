from deepcad.test_collection import testing_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename,download_demo
import argparse
import pdb
import os

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation")
parser.add_argument('--patch_xy', type=int, default=150, help="the width and height of image sequence")
parser.add_argument('--patch_t', type=int, default=150, help="the slices of image sequence")
parser.add_argument('--overlap_factor', type=int, default=0.6, help="overlap between image sequences")
parser.add_argument('--num_workers', type=int, default=1, help="number of CPU workers, rec 4 if you can")
parser.add_argument('--datasets_path', type=str, default='/n/data1/hms/neurobio/wilson/DeepCAD_datasets/20230811-3_EK021_7f_shade_cone_bright/', help="dataset root path")
parser.add_argument('--pth_dir', type=str, default='/n/data1/hms/neurobio/wilson/DeepCAD_datasets/pth/', help="pth file root path")
parser.add_argument('--denoise_model', type=str, default='20230811-1_EK021_7f_infinite_bar_methyl_salicylate', help='A folder containing models to be tested')
parser.add_argument('--test_datasize', type=int, default=10000000, help='dataset size to be tested')
opt = parser.parse_args()
print('the parameters of your training ----->')
print(opt)
def find_folders_with_partial_name(directory_path, partial_name):
    matching_folders = []
    for folder_name in os.listdir(directory_path):
        if partial_name in folder_name and os.path.isdir(os.path.join(directory_path, folder_name)):
            matching_folders.append(os.path.join(directory_path, folder_name))
    return matching_folders
########################################################################################################################

# %% First setup some parameters for testing
datasets_path = opt.datasets_path # folder containing tif files for testing
pth_dir = opt.pth_dir # standard model location + specific model
denoise_model = opt.denoise_model     # A folder containing pth models to be tested
test_datasize = opt.test_datasize     # the number of frames to be tested (test all frames if the number exceeds the total number of frames in a .tif file)
GPU = opt.GPU                         # the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')
patch_xy = opt.patch_xy               # the width and height of 3D patches
patch_t = opt.patch_t                 # the time dimension of 3D patches
overlap_factor = opt.overlap_factor   # the overlap factor between two adjacent patches. 
                                      # Since the receptive field of 3D-Unet is ~90, seamless stitching requires an overlap (patch_xyt*overlap_factor）of at least 90 pixels.
num_workers = opt.num_workers         # if you use Windows system, set this to 0.

# %% Make the subfolders we will need
partial_name = "time_slices_"
matching_folders = find_folders_with_partial_name(datasets_path, partial_name)

# %% iterate over each folder
for folder in matching_folders:
  output_dir = os.path.join(folder,'rt','test')
  os.makedirs(output_dir, exist_ok=True)
  
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
    'datasets_path': folder,
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
