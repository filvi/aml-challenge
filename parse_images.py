# %%
import os
from numba import jit, prange
from pprint import pprint
import cv2
from img_manipolation import *
from tqdm import tqdm
import logging
from orb_processing import *
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
    The class Dataset standardizes all the tasks easily across The Training\n
    and Test set. \n
    The methods are:\n
        print_dirs()    -> perform pprint of the directory saved in self.list_dirs\n
        get_dirs()      -> return (lst) self.list_dirs\n
        len_dirs()      -> return (int) len(self.list_dirs)\n
        print_files()   -> perform pprint of the files saved in self.list_files\n
        get_files()     -> return (lst) self.list_files\n
        len_files()     -> return (int) len(self.list_files)\n
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
    
    # creating a generator with path to images
    # --------------------------------------------------------------------------
    def parse_image_path(self, color=False):
        for i in prange(len(self.list_files)):
            yield self.list_files[i]
            
    # --------------------------------------------------------------------------

    def __getitem__(self, idx):
        '''
        Makes the dataset iterable.
        :param idx: the integer index of the element to retrieve
        :return: cv2.imread() object at position [idx].
        '''
        #JM
        img = cv2.imread(self.list_files[idx], cv2.IMREAD_COLOR)
        return img



# %%

# initialize the instance of the Dataset Class
# JM needs an os.walk object as input
# ==============================================================================
Training            = Dataset(training_path)  
Validation_Gallery  = Dataset(validation_gallery_path)
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



        # # %%
        # # Initial folder setup, create the folder structure to save the altered images
        # # ==============================================================================
        # def create_dir_structure(basedir, subdir=None, sub_subdir=None):
        #     if not os.path.isdir(basedir):
        #         os.mkdir(basedir)
        #     if subdir != None:
        #         if not os.path.isdir(os.path.join(basedir, subdir)):
        #             os.mkdir(os.path.join(basedir, subdir))
        #     if subdir != None and sub_subdir != None:
        #         if not os.path.isdir(os.path.join(basedir, subdir, sub_subdir)):
        #             os.mkdir(os.path.join(basedir, subdir, sub_subdir))
        # # ==============================================================================


        # # %%
        # # Create basic structure for storing the altered images for debug purpose
        # # ==============================================================================
        # def create_processed():
        #     create_dir_structure("processed")
        #     create_dir_structure("processed", "training")
        #     create_dir_structure("processed", "validation")
        #     create_dir_structure("processed", "validation", "gallery")
        #     create_dir_structure("processed", "validation", "query")
        # # ==============================================================================
        # create_processed()

        # # example usage of the generator in file saving mode
        # # ==============================================================================
        # def save_all_images(myinstance):
        #     counter = 0
        #     failed = 0
        #     all_files = myinstance.get_files()
        #     for img in tqdm(myinstance.parse_image(color=False), total=myinstance.len_files(), desc="Save all images"):
                
        #         # extrapolate from the filename the new path replacing the folder name
        #         # --------------------------------------------------------------------------
        #         fname = all_files[counter].replace("dataset", "processed")
        #         # --------------------------------------------------------------------------
        #         # if the file exist skips incrementing the loop
        #         # --------------------------------------------------------------------------
        #         if os.path.isfile(fname):
        #             counter += 1
        #             continue
        #         # --------------------------------------------------------------------------


        #         # The algorithm tries to save the image, if no subfolder is found it creates
        #         # one and then try again to save the image
        #         # --------------------------------------------------------------------------
        #         try:
        #             global_visual_debugger(img, savefig=True, fname=fname)
        #             # logging.info(f'created visual debug for {fname}')
        #         except:
        #             sep_index = fname.rfind(os.path.sep)
        #             splitted_fname = fname[sep_index:]
        #             fpath = fname[:sep_index]
        #             os.mkdir(fpath)
        #             try:
        #                 global_visual_debugger(img, savefig=True, fname=os.path.join(fpath,splitted_fname))
        #                 # logging.info(f'created folder {fpath}')
        #             except:
        #                 logging.warning(f'Failed to save {splitted_fname} in {fpath}')
        #                 failed +=1
        #         # --------------------------------------------------------------------------

        #         counter += 1
        #     print(f"Process finished with {failed} operations, you can run again or take a look at parse.log file")
        # # ==============================================================================



            # save_all_images(Training)
            # save_all_images(Validation_Gallery)
            # save_all_images(Validation_Query)



            # # %%
            # # example usage of the generator in file viewing mode
            # # ==============================================================================
            # def visualize_all_images():
            #     counter = 0
            #     all_files = Training.get_files()
            #     for img in tqdm(Training.parse_image_path(color=False), total=Training.len_files(),  desc="Display all images"):
            #         # pick_color_channel(img, "r")
            #         # noise_over_image(img, prob=0.015)
            #         # fakehdr(img, alpha=-100, beta=355, preset=None)
            #         # visual_fakehdr_debug(img, preset="dark")
            #         # visual_fakehdr_debug(img, alpha=-100, beta=355)
            #         # global_visual_debugger(img)

            #         img = cv2.imread(img)
            #         # # global_visual_debugger(img)
            #         # cv2.imshow('image',enhance_features(img, 5, 7, False) + enhance_features(img, 2, 4, True))
            #         # cv2.waitKey(0)
            #         # cv2.destroyAllWindows()

            #         im = cv2.imread( os.path.join("dataset", "training", "1", "ec50k_00010001.jpg"))
            #         visual_debug_orb(img, im)
            #         pass
            # # ==============================================================================


            # %%
            # counter = 0
            # all_files = Training.get_files()
            # matches_list = []
            # for img in tqdm(Training.parse_image_path(color=False), total=Training.len_files(),  desc="Display all images"):
            #     im1 = cv2.imread( os.path.join("dataset", "training", "1", "ec50k_00010003.jpg"))
            #     im2 = cv2.imread(img)
            #     temp_dict = {}
                
            #     a = return_distance(im1, im2)
            #     print(a[0].distance)
            #     print(a[1].distance)
            #     print(a[50].distance)
            #     print(a[-1].distance)
            #     counter += 1
            #     break
                