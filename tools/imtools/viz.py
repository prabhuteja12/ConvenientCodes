import numpy as np
import torch
from PIL import Image


def imcascade(*imgs, savepath=None):
    # imgs -- multiple pytorch tensors or numpy tensors
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    assert(isinstance(imgs[0], (np.ndarray, torch.Tensor)))
    if torch.is_tensor(imgs[0]):
        imgs = list(map(lambda x: x.cpu().numpy(), imgs))
    if isinstance(imgs[0], np.ndarray):
        imgs = [Image.fromarray(x.astype(np.uint8)) for x in imgs]
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    # print(min_shape)
    imgs_comb = np.hstack((np.asarray(i.resize(min_shape).convert('L')) for i in imgs))

    imgs_comb = Image.fromarray(imgs_comb.astype(np.uint8))
    if savepath:
        imgs_comb.save(savepath)
    return imgs_comb


if __name__ == "__main__":
    dir = '/idiap/home/prabhuteja/Downloads/eye_test/img/'
    im1 = Image.open(dir + '5.jpg')
    im2 = Image.open(dir + '10.jpg')
    im3 = Image.open(dir + '25.jpg')
    
    out = imcascade(np.array(im1), np.array(im2), np.array(im3))
    out.show()
