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

    imgs_comb = np.hstack((np.asarray(i.resize(min_shape).convert('L')) for i in imgs))
    imgs_comb = Image.fromarray(imgs_comb.astype(np.uint8))
    
    if savepath:
        imgs_comb.save(savepath)
    return imgs_comb
