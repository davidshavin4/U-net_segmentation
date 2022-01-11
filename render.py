
import matplotlib.pyplot as plt
import numpy as np

def show_images(lst_imgs, lst_titles=[], img_size=(5,10), ncols=5, cmap=None, font_size=8, keep_ticks=False):
    if not isinstance(lst_imgs, list):
        lst_imgs = [lst_imgs]
    if not len(lst_titles):
        lst_titles = [f"image {i+1}" for i in range(len(lst_imgs))]
    nrows = len(lst_imgs) // ncols + int(len(lst_imgs) % ncols > 0)
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*img_size[0],
                                                  nrows*img_size[1]))
    if type(ax) == np.ndarray:
        ax = ax.flatten()
    else:
        ax = np.array([ax])
    for i, [img, title] in enumerate(zip(lst_imgs, lst_titles)):
        if cmap is None and (len(img.shape)<3 or img.shape[2]==1):
            cmap = 'gray'
        ax[i].imshow(img, cmap=cmap)
        ax[i].set_title(title, fontdict={'fontsize': font_size})
        if not keep_ticks:
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()



# path = "dataA/dataA/CameraRGB/02_00_000.png"
# paths = glob("dataA/dataA/CameraRGB/*")[:20]
# images = [plt.imread(path) for path in paths]
# #img = plt.imread(path)
# show_images(images, img_size=(10,10))
# plt.show()