import numpy as np
from skimage.measure import label
from skimage import morphology
image = np.load('stars.npy')
plus_struct = np.array([
    [0,0,1,0,0],
    [0,0,1,0,0],
    [1,1,1,1,1],
    [0,0,1,0,0],
    [0,0,1,0,0]])
x_struct = np.array([
    [1,0,0,0,1],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,1,0,1,0],
    [1,0,0,0,1]])
plus_image = morphology.opening(image, footprint=plus_struct)
x_image = morphology.opening(image, footprint=x_struct)
plus_labeled = label(plus_image)
x_labeled = label(x_image)
num = x_labeled.max() + plus_labeled.max()
print(num)