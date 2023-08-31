from deepcad.train_collection import training_class
from deepcad.movie_display import display
from deepcad.utils import get_first_filename,download_demo
import argparse
import pdb
import os
import shutil

#############################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--GPU', type=str, default='0', help="the index of GPU you will use for computation")
parser.add_argument('--patch_xy', type=int, default=10, help="the width and height of image sequence")
parser.add_argument('--patch_t', type=int, default=10, help="the slices of image sequence")
parser.add_argument('--overlap_factor', type=int, default=0.25, help="overlap between image sequences")
parser.add_argument('--num_workers', type=int, default=1, help="number of CPU workers, rec 4 if you can")
parser.add_argument('--datasets_path', type=str, default='/n/data1/hms/neurobio/wilson/DeepCAD_datasets/20230811-3_EK021_7f_shade_cone_bright/', help="dataset root path")
parser.add_argument('--train_datasets_size', type=int, default=20630, help='dataset size to be tested')
parser.add_argument('--n_epochs', type=int, default=10, help='dataset size to be tested')
parser.add_argument('--chosen_plane', type=int, default=0, help='dataset size to be tested')
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
train_datasets_size = opt.train_datasets_size    # dataset size for training (the number of patches)
GPU = opt.GPU                         # the index of GPU you will use for computation (e.g. '0', '0,1', '0,1,2')
patch_xy = opt.patch_xy               # the width and height of 3D patches
patch_t = opt.patch_t                 # the time dimension of 3D patches
overlap_factor = opt.overlap_factor   # the overlap factor between two adjacent patches. 
                                      # Since the receptive field of 3D-Unet is ~90, seamless stitching requires an overlap (patch_xyt*overlap_factor）of at least 90 pixels.
num_workers = opt.num_workers         # if you use Windows system, set this to 0.
n_epochs = opt.n_epochs               # the number of training epochs
chosen_plane = opt.chosen_plane       # Which plane in a multi-plane z-stack should we train with?

# %% Make the subfolders we will need
partial_name = "time_slices_"
matching_folders = find_folders_with_partial_name(datasets_path, partial_name)

# %% iterate over each folder
for folder in matching_folders:
  output_dir = os.path.join(folder,'rt','train')
  os.makedirs(output_dir, exist_ok=True)
  pth_dir = os.path.join(folder,'rt','pth') # pth file and visualization result file path
  os.makedirs(pth_dir, exist_ok=True)
  
  #%% use the chosen plane
  tif_files = [file for file in os.listdir(folder) if file.endswith('.tif')]
  if chosen_plane==0:
    num_tif_files = len(tif_files)
    chosen_plane = int(num_tif_files/2)
  train_folder = os.path.join(folder, str(chosen_plane))
  tif_path=os.path.join(folder,tif_files[chosen_plane])
  os.makedirs(train_folder, exist_ok=True)
  shutil.copy(tif_path, train_folder)

  # %% Setup some parameters for result visualization during testing period (optional)
  visualize_images_per_epoch = False  # choose whether to display inference performance after each epoch
  save_test_images_per_epoch = True  # choose whether to save inference image after each epoch in pth path
  
  # %% Play the demo noise movie (optional)
  # playing the first noise movie using opencv.
  display_images = False
  
  if display_images:
    display_filename = get_first_filename(train_folder)
    print('\033[1;31mDisplaying the first raw file -----> \033[0m')
    print(display_filename)
    display_length = 500  # the frames number of the noise movie
    # normalize the image and display
    display(display_filename, display_length=display_length, norm_min_percent=0.5, norm_max_percent=99.8)
  
  train_dict = {
    # dataset dependent parameters
    'patch_x': patch_xy,
    'patch_y': patch_xy,
    'patch_t': patch_t,
    'overlap_factor':overlap_factor,
    'scale_factor': 1,                  # the factor for image intensity scaling
    'select_img_num': 100000,           # select the number of images used for training (use all frames by default)
    'train_datasets_size': train_datasets_size,
    'datasets_path': train_folder,
    'pth_dir': pth_dir,
    # network related parameters
    'n_epochs': n_epochs,
    'lr': 0.00005,                       # initial learning rate
    'b1': 0.5,                           # Adam: bata1
    'b2': 0.999,                         # Adam: bata2
    'fmap': 16,                          # the number of feature maps
    'GPU': GPU,
    'num_workers': num_workers,
    'visualize_images_per_epoch': visualize_images_per_epoch,
    'save_test_images_per_epoch': save_test_images_per_epoch,
    'output_dir': output_dir
  }
  # %%% Training preparation
  # first we create a training class object with the specified parameters
  tc = training_class(train_dict)
  # start the training process
  tc.run()
