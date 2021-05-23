# %%

import numpy as np
import cv2
import random
from numba import prange, jit, parfors
import matplotlib.pyplot as plt
import os

#  TODO aggiungere formato classe
# %%
# creating the generator for each image subset
# ==================================================================
def pick_color_channel(image, channel):

    """
    This function allows you to manually pick a color channel for the image `img`
    The choice for `channels` are (case insensitive)
    - [R]ed
    - [G]reen
    - [B]lue
    """
    img = image.copy()
    
    
    # Check for the correct channel for the image to process
    # --------------------------------------------------------------------------
    if channel.lower() == "r":
        return np.stack((img[:,:,0],)*np.uint8(3), axis=2)

    elif channel.lower() == "g":
        return np.stack((img[:,:,1],)*np.uint8(3), axis=2)

    elif channel.lower() == "b":
        return np.stack((img[:,:,2],)*np.uint8(3), axis=2)
    else:
        raise ValueError("The channel specified in `pick_color_channel(image, channel)` is incorrect.\nPlease use one of the following: `r` `g` `b`")
# ==============================================================================




# %%
# ==============================================================================
# The jit function decorator of the packet Numba allows to compile the code to 
# make it run faster. 
# 
# The nogil=True allow us to exit the single threaded mode in which me use to
# work in vanilla python. GIL is an acronym for Global Interpreter Lock, and 
# setting this switch we can enhance performance drammatically.
#
# The parallel option allow us to parallelize the operation across the CPU
# threads
#
# The nopython arguments is to tell python try not to infere the type of data
# resulting in an even better timing.
#
# In fact the results measured with %time magic methods are the following:
#
# Normal -> Wall time: 128 ms
# Numba compile run ->  Wall time: 1.2 s
# Numba compiled ->  Wall time: 2.97 ms
# ==============================================================================
@jit(nogil=True, parallel=True, nopython=True)
def noise_over_image(image, prob=0.01):
    """
    Add salt and pepper noise to the image.\n
    The arguments for the function are:
    img -> the image we want to degrade
    prob -> value between 0 and 1 that denote the intensity of the degradation
    """ 
    img = image.copy()


    # initialize the matrix which will be returned
    # --------------------------------------------------------------------------
    output = np.zeros(img.shape, np.uint8)
    # --------------------------------------------------------------------------
    
    # set the max treshold for white value
    # --------------------------------------------------------------------------
    treshold = 1 - prob        
    # --------------------------------------------------------------------------

    # for each row and column of the matrix check if the condition applies to
    # change the value to white or black pixel to create a salt and pepper noise
    # --------------------------------------------------------------------------
    for i in prange(img.shape[0]):
        for j in prange(img.shape[1]):
            rnd = random.random()
            if rnd < prob:
                output[i][j] = 0
            elif rnd > treshold:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    # --------------------------------------------------------------------------

    return output
# ==============================================================================


# %%
# Fake HDR is a function that normalize the pixel from the alpha value to the beta
# one. We have some handy preset if we don't want to specify these value separately
# ==============================================================================
def fakehdr(image, alpha=-100, beta=355, preset=None):
    img = image.copy()
    
    # Setting the alpha and beta value for the presets
    # --------------------------------------------------------------------------
    if preset == "dark":
        alpha = -100
        beta = 150
    elif preset == "light":
        alpha = 500
        beta = -100
    norm_img = cv2.normalize(img,None, alpha=alpha, beta=beta, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    # --------------------------------------------------------------------------

    return norm_img
# ==============================================================================



# %%
# A simple visual debugger to test the correct value or preset with an image
# ==============================================================================
def visual_fakehdr_debug(img):
    # img = cv2.edgePreservingFilter(img, flags=2, sigma_s=200, sigma_r=0.1)
    img = cv2.cv2.detailEnhance(img, sigma_s=60, sigma_r=0.15)
    cv2.imshow('image',img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
# ==============================================================================


# %%

def enhance_features(img, val1, val2, inverse=True):
    kernel1 = np.ones((val1,val1),np.float32)/val1**2
    dst1 = cv2.filter2D(img,-1,kernel1)
    kernel2 = np.ones((val2,val2),np.float32)/val2**2
    dst2 = cv2.filter2D(img,-1,kernel2)
    if inverse:
        return  dst1 - dst2 - img
    else:
        return  dst1 - dst2 + img


def draw_orb(img, enhanced=True):
    if enhanced:
        training_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_gray  = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
        training_image = enhance_features(training_image, 5, 7, True)
        training_gray  = enhance_features(training_gray, 5, 7, True)
    else:
        training_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        training_gray = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
    test_image = cv2.pyrDown(training_image)
    test_image = cv2.pyrDown(test_image)
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()

    train_keypoints, train_descriptor = orb.detectAndCompute(training_gray, None)
    test_keypoints, test_descriptor = orb.detectAndCompute(test_gray, None)

    keypoints_without_size = np.copy(training_image)
    keypoints_with_size = np.copy(training_image)

    cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

    cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return (keypoints_with_size, keypoints_without_size)







# %%
# This function allows us to visually debug the changes in the images
# ==============================================================================
def global_visual_debugger(image, savefig=False, fname=0):
    """
    VD stands for visual debug, with this function we can have visual
    information about the changes made in the images while applying the various
    functions to degrade/alterate the image
    """
    img = image.copy()

    
    # subplot settings
    # --------------------------------------------------------------------------
    column = 4
    row = 4
    fig=plt.figure(dpi=300)
    fig.subplots_adjust(wspace = 0.5, hspace = 0.5)
    # --------------------------------------------------------------------------
    
    
    # Original
    # --------------------------------------------------------------------------
    img_original = img
    im1 = fig.add_subplot(row,column,1)
    im1.title.set_text("original image")
    im1.title.set_size(5)
    plt.axis('off')
    plt.imshow(img)
    # --------------------------------------------------------------------------


    # Original + Noise
    # --------------------------------------------------------------------------
    noise_original = noise_over_image(img)
    im2 = fig.add_subplot(row,column,5)
    im2.title.set_text("original image + noise")
    im2.title.set_size(5)
    plt.axis('off')
    plt.imshow(noise_original)
    # --------------------------------------------------------------------------


    # Red 
    # --------------------------------------------------------------------------
    img_red = pick_color_channel(img, "r")
    im3 = fig.add_subplot(row,column,2)
    im3.title.set_text("red channel")
    im3.title.set_size(5)
    plt.axis('off')
    plt.imshow(img_red)
    # --------------------------------------------------------------------------

    # Red + Noise
    # --------------------------------------------------------------------------
    noise_red = noise_over_image(img_red)
    im4 = fig.add_subplot(row,column,6)
    im4.title.set_text("red + noise")
    im4.title.set_size(5)
    plt.axis('off')
    plt.imshow(noise_red)
    # --------------------------------------------------------------------------

    # Green
    # --------------------------------------------------------------------------
    img_green = pick_color_channel(img, "g")
    im5 = fig.add_subplot(row,column,3)
    im5.title.set_text("green channel")
    im5.title.set_size(5)
    plt.axis('off')
    plt.imshow(img_green)
    # --------------------------------------------------------------------------

    # Green + Noise
    # --------------------------------------------------------------------------
    noise_green = noise_over_image(img_green, 0.01)
    im6 = fig.add_subplot(row,column,7)
    im6.title.set_text("green channel + noise")
    im6.title.set_size(5)
    plt.axis('off')
    plt.imshow(noise_green)
    # --------------------------------------------------------------------------

    # Blue
    # --------------------------------------------------------------------------
    img_blue = pick_color_channel(img, "g")
    im7 = fig.add_subplot(row,column,4)
    im7.title.set_text("blue channel")
    im7.title.set_size(5)
    plt.axis('off')
    plt.imshow(img_blue)
    # --------------------------------------------------------------------------

    # Blue + Noise
    # --------------------------------------------------------------------------
    noise_blue = noise_over_image(img_blue, 0.013)
    im8 = fig.add_subplot(row,column,8)
    im8.title.set_text("blue channel + noise")
    im8.title.set_size(5)
    plt.axis('off')
    plt.imshow(noise_blue)
    # --------------------------------------------------------------------------
    
    # Fake HDR preset DARK
    # --------------------------------------------------------------------------
    img_hdr_dark = fakehdr(img, preset="dark")
    im9 = fig.add_subplot(row,column,9)
    im9.title.set_text("Fake HDR dark")
    im9.title.set_size(5)
    plt.axis('off')
    plt.imshow(img_hdr_dark)
    # --------------------------------------------------------------------------

    # Fake HDR preset DARK + Noise
    # --------------------------------------------------------------------------
    noise_hdr_dark = noise_over_image(img_hdr_dark, 0.013)
    im10 = fig.add_subplot(row,column,10)
    im10.title.set_text("Fake HDR dark + noise")
    im10.title.set_size(5)
    plt.axis('off')
    plt.imshow(noise_hdr_dark)
    # --------------------------------------------------------------------------
    
    
    # Fake HDR preset LIGHT
    # --------------------------------------------------------------------------
    img_hdr_light = fakehdr(img, preset="light")
    im11 = fig.add_subplot(row,column,11)
    im11.title.set_text("Fake HDR light")
    im11.title.set_size(5)
    plt.axis('off')
    plt.imshow(img_hdr_light)
    # --------------------------------------------------------------------------


    # Fake HDR preset LIGHT + Noise
    # --------------------------------------------------------------------------
    noise_hdr_light = noise_over_image(img_hdr_light, 0.013)
    im12 = fig.add_subplot(row,column,12)
    im12.title.set_text("Fake HDR Light + noise")
    im12.title.set_size(5)
    plt.axis('off')
    plt.imshow(noise_hdr_light)
    # --------------------------------------------------------------------------
    
    # ORB points
    # --------------------------------------------------------------------------
    orb_point_distance = draw_orb(img,enhanced=False)[0]
    im9 = fig.add_subplot(row,column,13)
    im9.title.set_text("ORB points with distance")
    im9.title.set_size(5)
    plt.axis('off')
    plt.imshow(orb_point_distance)
    # --------------------------------------------------------------------------

    # Fake HDR preset DARK + Noise
    # --------------------------------------------------------------------------
    orb_point = draw_orb(img,enhanced=False)[1]
    im10 = fig.add_subplot(row,column,14)
    im10.title.set_text("ORB points")
    im10.title.set_size(5)
    plt.axis('off')
    plt.imshow(orb_point)
    # --------------------------------------------------------------------------
    
    
    # Fake HDR preset LIGHT
    # --------------------------------------------------------------------------
    enhanced_distance = draw_orb(img,enhanced=True)[0]
    im11 = fig.add_subplot(row,column,15)
    im11.title.set_text("Enhanced image with ORB distance")
    im11.title.set_size(5)
    plt.axis('off')
    plt.imshow(enhanced_distance)
    # --------------------------------------------------------------------------


    # Fake HDR preset LIGHT + Noise
    # --------------------------------------------------------------------------
    enhanced = draw_orb(img,enhanced=True)[1]
    im12 = fig.add_subplot(row,column,16)
    im12.title.set_text("Enhanced image with ORB points")
    im12.title.set_size(5)
    plt.axis('off')
    plt.imshow(enhanced)
    # --------------------------------------------------------------------------
    if not savefig:
        plt.show()
    else:
        plt.savefig(fname)
    plt.close("all")
# ==============================================================================
