import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import opening, dilation, closing, erosion

image = np.load("wires5.npy")
struct = np.ones((3,1))
labeled = label(image)
processed = opening(image, footprint=struct)
print(f"{labeled.max()}")


for n in range (1, labeled.max()+1):
    wire = label(opening(labeled==n, footprint=struct))
    parts = wire.max()
    print(f"Wire = {n}, parts = {parts}")


plt.subplot(121)
plt.imshow(image)
plt.subplot(122)
plt.imshow(processed)
plt.show()
