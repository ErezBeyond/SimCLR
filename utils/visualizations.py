import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.axes import Axes
# plt.ion()
# plt.interactive(False)
import matplotlib.patches as patches
import matplotlib.colors as mpl_colors
import cv2
import shapely.geometry as geometry
import numpy as np

distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c',
                   '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
                   '#000075']
distinct_float_rgb = np.array([mpl_colors.to_rgb(x) for x in distinct_colors])
distinct_int_bgr = (distinct_float_rgb[:, ::-1] * 255).astype(int)
n_distinct = len(distinct_colors)


def imshow_and_boxes(I, boxes, color=None):
    return imshow_and_rects(I, np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2])), color)

def imshow_and_rects(I, rects, color=None):
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(I)
    if color is None:
        color = distinct_colors[0]
    elif isinstance(color, (int, float)):
        color = distinct_colors[int(color) % n_distinct]
    for rect in rects:
        # Create a Rectangle patch
        rect_obj = patches.Rectangle((rect[0], rect[1]), rect[2], rect[3], linewidth=3, edgecolor=color,
                                     facecolor=color, alpha=0.1)
        # Add the patch to the Axes
        ax.add_patch(rect_obj)
    plt.show()
    return ax
