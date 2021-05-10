# %%
import os
from numba import jit, prange
from pprint import pprint
import cv2
from img_manipolation import *
from tqdm import tqdm
import logging
logging.basicConfig(filename='parse.log', encoding='utf-8', level=logging.INFO)
# %%
# initialize the generator for the respective folders
# ==============================================================================
training_path           = os.walk(os.path.join("dataset", "training"), topdown=False)
validation_gallery_path = os.walk(os.path.join("dataset", "validation", "gallery"), topdown=False)
validation_query_path   = os.walk(os.path.join("dataset", "validation", "query"), topdown=False)


# %%
# Create the class for the Training and Validation instances
# ==============================================================================
class Dataset:
    """
    The class Dataset is standardize all the tasks easily across The Training
    and Test set. 
    The methods are:
        print_dirs()    -> perform pprint of the directory saved in self.list_dirs
        get_dirs()      -> return (lst) self.list_dirs
        len_dirs()      -> return (int) len(self.list_dirs)
        print_files()   -> perform pprint of the files saved in self.list_files
        get_files()     -> return (lst) self.list_files
        len_files()     -> return (int) len(self.list_files)
        parse_image()   -> return a generator for looping through the dataset
    """

    # On instance creation parse the directory and add them to a list
    # Although we have nested loops and Big-OH Notation of N^2 the average time 
    # for the given dataset is pretty reasonable, as follows:
    # training_path:            277 ns ± 12.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # validation_gallery_path:  274 ns ± 9.12 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # validation_query_path:    269 ns ± 10.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
    # --------------------------------------------------------------------------
    def __init__(self, mygenerator):
        self.list_dirs = []
        self.list_files = []
        self.mygenerator = mygenerator
        for root, dirs, files in self.mygenerator:
            for d in prange(len(dirs)):
                self.list_dirs.append(os.path.join(root, dirs[d]))
            for f in  prange(len(files)):
                if not files[f].endswith('.DS_Store'):
                    self.list_files.append(os.path.join(root, files[f]))
    # --------------------------------------------------------------------------


    # Utilities functions on the directory list
    # --------------------------------------------------------------------------
    def print_dirs(self):
        pprint(self.list_dirs)

    def get_dirs(self):
        return self.list_dirs

    def len_dirs(self):
        return len(self.list_dirs)
    # --------------------------------------------------------------------------
    
    
    # Utilities functions on the file list
    # --------------------------------------------------------------------------
    def print_files(self):
        pprint(self.list_files)

    def get_files(self):
        return self.list_files

    def len_files(self):
        return len(self.list_files)
    # --------------------------------------------------------------------------


    # creating a generator to use later on with all the image loaded
    # --------------------------------------------------------------------------
    def parse_image(self, color=False):
        for i in prange(len(self.list_files)):
            img = cv2.imread(self.list_files[i], cv2.IMREAD_COLOR)
            yield img
    # --------------------------------------------------------------------------



# %%

# initialize the instance of the Dataset Class
# ==============================================================================
Training            = Dataset(training_path)
Validation_Gallery   = Dataset(validation_gallery_path)
Validation_Query    = Dataset(validation_query_path)
# ==============================================================================



# %%
# Debug pprint all files
# ==============================================================================
# Training.print_files()
# Validation_Gallery.print_files()
# Validation_Query.print_files()
# ==============================================================================


# Debug pprint all directories
# ==============================================================================
# Training.print_dirs()
# Validation_Gallery.print_dirs()
# Validation_Query.print_dirs()
# ==============================================================================



# %%
# Initial folder setup, create the folder structure to save the altered images
# ==============================================================================
def create_dir_structure(basedir, subdir=None, sub_subdir=None):
    if not os.path.isdir(basedir):
        os.mkdir(basedir)
    if subdir != None:
        if not os.path.isdir(os.path.join(basedir, subdir)):
            os.mkdir(os.path.join(basedir, subdir))
    if subdir != None and sub_subdir != None:
        if not os.path.isdir(os.path.join(basedir, subdir, sub_subdir)):
            os.mkdir(os.path.join(basedir, subdir, sub_subdir))
# ==============================================================================


# %%
# Create basic structure for storing the altered images for debug purpose
# ==============================================================================
def create_processed():
    create_dir_structure("processed")
    create_dir_structure("processed", "training")
    create_dir_structure("processed", "validation")
    create_dir_structure("processed", "validation", "gallery")
    create_dir_structure("processed", "validation", "query")
# ==============================================================================


# example usage of the generator in file saving mode
# ==============================================================================
def save_all_images(myinstance):
    counter = 0
    failed = 0
    all_files = myinstance.get_files()
    for img in tqdm(myinstance.parse_image(color=False), total=myinstance.len_files()):
        
        # extrapolate from the filename the new path replacing the folder name
        # --------------------------------------------------------------------------
        fname = all_files[counter].replace("dataset", "processed")
        # --------------------------------------------------------------------------
        # if the file exist skips incrementing the loop
        # --------------------------------------------------------------------------
        if os.path.isfile(fname):
            counter += 1
            continue
        # --------------------------------------------------------------------------


        # The algorithm tries to save the image, if no subfolder is found it creates
        # one and then try again to save the image
        # --------------------------------------------------------------------------
        try:
            global_visual_debugger(img, savefig=True, fname=fname)
            logging.info(f'created visual debug for {fname}')
        except:
            sep_index = fname.rfind(os.path.sep)
            splitted_fname = fname[sep_index:]
            fpath = fname[:sep_index]
            os.mkdir(fpath)
            try:
                global_visual_debugger(img, savefig=True, fname=os.path.join(fpath,splitted_fname))
                logging.info(f'created folder {fpath}')
            except:
                logging.warning(f'Failed to save {splitted_fname} in {fpath}')
                failed +=1
        # --------------------------------------------------------------------------

        counter += 1
    print(f"Process finished with {failed} operations, you can run again or take a look at parse.log file")
# ==============================================================================



# %%
# example usage of the generator in file viewing mode
# ==============================================================================
def visualize_all_images():
    counter = 0
    all_files = Training.get_files()
    for img in tqdm(Training.parse_image(color=False), total=Training.len_files()):
        # pick_color_channel(img, "r")
        # noise_over_image(img, prob=0.015)
        # fakehdr(img, alpha=-100, beta=355, preset=None)
        # visual_fakehdr_debug(img, preset="dark")
        # visual_fakehdr_debug(img, alpha=-100, beta=355)
        # global_visual_debugger(img)

        # global_visual_debugger(img)

        pass
# ==============================================================================


# %%
