import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
from skimage.io import imread
from skimage.color import rgb2hsv

image = imread('balls_and_rects.png')
hsv = rgb2hsv(image)
h = hsv[:, :, 0]

colors_c = []
colors_r = []
for color in np.unique(h):
    if color == 0:
        continue
    binary = h == color
    labeled = label(binary)
    props = regionprops(labeled)

    for prop in props:
        norm_area = prop.area_bbox / prop.area
        if norm_area == 1:
            colors_r.append(color)
        elif norm_area > 0.75:
            colors_c.append(color)
groups_r = [[colors_r[0]]]
groups_c = [[colors_c[0]]]
delta = 0.05
for i in range(1, len(colors_r)):
    if abs(colors_r[i - 1] - colors_r[i]) < delta:
        groups_r[-1].append(colors_r[i])
    else:
        groups_r.append([])

for i in range(1, len(colors_c)):
    if abs(colors_c[i - 1] - colors_c[i]) < delta:
        groups_c[-1].append(colors_c[i])
    else:
        groups_c.append([])

print(f"Всего фигур: {len(colors_c) + len(colors_r)}")

print(f"Круги: {len(colors_c)}")
for grp in groups_c:
    avg_color = np.mean(grp)
    print(avg_color, len(grp))

print(f"Прямоугольники: {len(colors_r)}")
for grp in groups_r:
    avg_color = np.mean(grp)
    print(avg_color, len(grp))

plt.subplot(121)
plt.imshow(h, cmap="gray")
plt.subplot(122)
plt.plot(np.unique(h), 'o')
plt.show()