import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label
from object_properties import area
image=np.load("coins.npy")
labeled=label(image)
areas=np.array([], dtype=int)
for i in range(1, np.max(labeled)+1):
    areas=np.append(areas, area(labeled==i))
area_mean = np.unique(areas)
res=[1, 2, 5, 10]
for i in range(4):
    res[i] = res[i]*np.sum(areas==area_mean[i])
print(sum(res))


