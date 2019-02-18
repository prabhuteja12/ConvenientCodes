import random
import os


def seed_everything(seed=1212):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    try: 
        import torch
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

    try:
        import numpy as np
        np.random.seed(seed)
    except:
        pass
