from model import UNetModel
from preprocess import show_batch, create_dataloader
from device import DeviceDataLoader
from glob import glob



def test():
    x = torch.randn((3, 3, 256, 256))
    print('x.shape: ', x.shape)
    model = UNetModel()

    pred = model(x)
    print('pred.shape: ', pred.shape)

if __name__=="__main__":
    images_path = "dataA/dataA"
    images_paths = glob(images_path+"/CameraRGB/*")
    images = []
    titles = []
    for i, img_path in enumerate(images_paths[:50]):
        images.append(plt.imread(img_path))
        titles.append("title_"+str(i))

    #show_images(images, titles, ncols=10)

    train_dl = create_dataloader(images_path)
    #print('train_dl: ', train_dl)

    for img, seg in train_dl:
        #print('img.shape: ', img.shape)
        #print('seg.shape: ', seg.shape)
        show_batch(img, ncols = 16)
        #show_batch(seg)
        #print('seg: ', seg)
        break

