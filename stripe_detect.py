from __future__ import print_function
import nipy
from scipy import signal, ndimage
import time
import numpy as np
from matplotlib import pyplot as plt
from librosa.feature import spectral_centroid
from skimage import segmentation, morphology

#########
# Previous Attempts
#load image
img = nipy.io.files.as_image('/Users/Jonny/Documents/brainhaq/rucyb2_4d.nii.gz')
img = np.rot90(img,axes=(0,2))
plt.imshow(img[:,50,:,48])

# Visualize some slices
for i in range(img.shape[1]):
    plt.imshow(img[:,i,:,48])
    plt.pause(0.01)

# Animate the autocorrelation
plt.figure()
for i in range(40,60):
    autocor = signal.correlate2d(img[:,50,:,i],img[:,50,:,i],mode="same")
    plt.subplot(121)
    plt.imshow(autocor)
    plt.subplot(122)
    plt.imshow(img[:,50,:,i])
    plt.pause(0.1)

# User a filter kernel to ... maximize striping
imgmax = np.max(img)
kern = np.array([[imgmax],[0.],[imgmax]])
conv = signal.convolve2d(img[:,50,:,47],kern)
plt.imshow(conv)

#######
# The basic function
def find_stripes(img, coronal_slice):
    # Use spectral density estimations to find stripes
    # Stripes are essentially high-frequency artefacts

    # First convert image to floats if we haven't already
    if 'int' in img.dtype.name:
        img = img.astype('float32')

    # Then rescale to 0-1 if we haven't already
    if np.max(img) > 1.:
        vmin = np.min(img)
        vmax = np.max(img)
        img = (img-vmin)/(vmax-vmin)

    # Segment the image to get the brain
    # Done shittily here, but check out scipy's segmentation module
    # Also computing for the volume even though we only do it on a slice for speed
    brain = np.zeros(img.shape,dtype=np.int)
    for t in range(brain.shape[3]):
        print("Segmenting Image, t = {}/{}".format(t,brain.shape[3]),end="\r")
        # Get a seed region for stuff that is probably brain.
        probably_brain = np.zeros(img.shape[0:3], dtype=np.uint8)
        probably_brain[img[:,:,:,t] > .2] = 1
        probably_brain = ndimage.morphology.binary_fill_holes(probably_brain)
        open_brain = morphology.opening(probably_brain,morphology.ball(3))
        brain[:,:,:,t] = morphology.closing(open_brain, morphology.ball(3))

    # Take a slice to save memory
    brain = brain[:,coronal_slice,:,:]
    img = img[:,coronal_slice,:,:]

    idx = np.where(brain)
    # Get a square box around the brain
    minx = np.min(idx[1])
    maxx = np.max(idx[1])
    miny = np.min(idx[0])
    maxy = np.max(idx[0])


    # Get shape w/ test array cuz i'm lazy
    wl = signal.welch(img[miny:maxy, minx:maxx, 0], axis=0)
    welches = np.zeros((wl[1].shape[0],wl[1].shape[1],img.shape[2]))

    for t in range(img.shape[2]):
        print("Computing Spectral Density, t = {}/{}".format(t+1,img.shape[2]),end="\r")
        wl = signal.welch(img[miny:maxy,minx:maxx,t],axis=0)
        welches[:,:,t] = np.array(wl[1])

    # Lazy fix to us not finding banding when it's only in part of the image
    welches_split = np.split(welches,2,axis=1)
    welches_1 = np.mean(welches_split[0],axis=1)
    welches_2 = np.mean(welches_split[1], axis=1)

    # Plot a few freqs for left and right
    plt.figure()
    plt.subplot(121)
    plt.plot(range(welches.shape[2]), welches_1[30, :])
    plt.plot(range(welches.shape[2]), welches_1[29, :])
    plt.plot(range(welches.shape[2]), welches_1[28, :])
    plt.subplot(122)
    plt.plot(range(welches.shape[2]), welches_2[30, :])
    plt.plot(range(welches.shape[2]), welches_2[29, :])
    plt.plot(range(welches.shape[2]), welches_2[28, :])





