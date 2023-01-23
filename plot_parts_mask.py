import numpy as np
from Extracted_features_framework_part_based import Extracted_features_framework
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2

extracted_features_framework = Extracted_features_framework("D:/Pair-reID_data/posenet_features/", ["extracted_features_q.pickle", "extracted_features_g.pickle"], is_distances_matrix_computed = True, is_confidence_matrix_computed=True)

print(extracted_features_framework)

fig, axs = plt.subplots(3,3,figsize=(10,10))

print(type(axs[0,0]))

img1 = cv2.imread("D:\\Pair-reID_data\\reference_repertory\\" + str(200) + ".jpg")
axs[0,0].title.set_text("person")
axs[0,0].imshow(img1)
axs[0,0].set_xticks([])
axs[0,0].set_yticks([])


print(extracted_features_framework.parts_masks.shape)
names = ["foreground", "head", "torso", "right arm","left arm","right leg","left leg","right foot","left foot"]
row = 0
col = 1
for i in range(1,9):

    im = axs[row, col].imshow(extracted_features_framework.parts_masks[200,i],vmin=0,vmax=1)
    axs[row, col].title.set_text(names[i])
    axs[row, col].set_xticks([])
    axs[row, col].set_yticks([])

    divider = make_axes_locatable(axs[row, col])

    ax_cb = divider.append_axes("right", size="5%", pad=0.05)
    fig = axs[row, col].get_figure()
    fig.add_axes(ax_cb)

    plt.colorbar(im, cax=ax_cb)
    ax_cb.yaxis.tick_right()
    ax_cb.yaxis.set_tick_params(labelright=True)

    col += 1
    if col == 3:
        col = 0
        row += 1
plt.savefig("parts_mask.jpg")
plt.show()