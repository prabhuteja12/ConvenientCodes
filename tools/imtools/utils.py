import numpy as np
import torch


def to_categorical(y, num_classes, dtype='float32'):
    typ = 'np'
    if isinstance(y, torch.Tensor):
        typ = 'torch'
        dev = y.device
        y = y.cpu().numpy()

    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    
    y = y.ravel().astype(np.int)
    y[y>=num_classes] = 0

    n = y.shape[0]
    
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.moveaxis(np.reshape(categorical, output_shape), -1, 0)
    if typ == 'torch':
        return torch.from_numpy(categorical).to(dev)
    return categorical


if __name__ == "__main__":
    a = [[1, 2], [0, 2]]
    a = np.array(a)
    res = to_categorical(a, 3, dtype=np.int)
    print(res.shape)
    print(res)