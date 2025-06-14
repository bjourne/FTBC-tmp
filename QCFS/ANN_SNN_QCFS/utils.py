import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import random
import os
import logging
from Models import IF

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger



def val(model, test_loader, T, device, sample_iter=None):
    print("Validation with", device)


    if sample_iter is None:
        sample_iter = len(test_loader)
        print(f"total number of batches: {sample_iter}")

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for bi, (xs, ys) in enumerate(test_loader):
            if bi > 0 and bi % 10 == 0:
                print(f"bi: {bi}")
                print(f"acc. until now: {100 * correct / total}")

            xs, ys = xs.to(device), ys.to(device)
            if T > 0:
                outputs = model(xs)
                outputs = outputs.mean(0)
            else:
                outputs = model(xs)
            _, predicted = outputs.max(1)
            total += float(ys.size(0))
            correct += float(predicted.eq(ys).sum().item())
            if bi == sample_iter:
                break
        final_acc = 100 * correct / total

    return final_acc
