import torch
import torch.nn.functional as F
import clip
from PIL import Image, ImageFile, UnidentifiedImageError
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from timeit import default_timer as timer
import argparse
import pickle
import json
import numpy as np

class Dset(data.Dataset):
    def __init__(self, params, preprocess):
        self.params = params
        self.preprocess = preprocess
        self.idx2imgpath = params['idx2imgpath']
        self.numDataPoints = len(self.idx2imgpath)

    def __len__(self):
        return self.numDataPoints

    def __getitem__(self, index):
        image_path = self.idx2imgpath[str(index)]
        try:
            image = Image.open(image_path)
            image = self.preprocess(image) 
        except UnidentifiedImageError as e:
            return None
        except OSError as e:
            return None    
        
        item = {}
        item['image_data'] = image
        if self.params['data_name'] == 'CC12M':
            image_id = int(image_path.split('/')[-1]) 
            item['image_id'] = torch.LongTensor([image_id])
        elif self.params['data_name'] == 'VD':
            image_id = int(image_path.split('/')[-1][-16:-4])
            item['image_id'] = torch.LongTensor([image_id])
        return item

def batch_iter(dataloader, params):
    for epochId in range(params['num_epochs']):
        for idx, batch in enumerate(tqdm(dataloader)):
            yield epochId, idx, batch

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def cov_mean(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Thanks :
    - https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
    - https://github.com/numpy/numpy/blob/master/numpy/lib/function_base.py#L2276-L2494

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m.type(torch.double)  
    fact = 1.0 / (m.size(1) - 1)

    mean = torch.mean(m, dim=1, keepdim=True)
    m = m - mean
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze(), mean.squeeze()


if __name__ == '__main__':
    # argument 
    parser = argparse.ArgumentParser(description='Arguments for in-domain image data filtering')
    parser.add_argument('-gpu_ids', nargs="+", type=int, default=[0, 1, 2, 3, 4, 5], help="List of ids of GPUs to use.")
    parser.add_argument('-imgpath', type=str, default='visdial_idx2imgpath.json')
    parser.add_argument('-num_epochs', default=1, type=int, help='total number of epochs')
    parser.add_argument('-step', default='build', type=str, choices=['build', 'inference'])
    parser.add_argument('-feats', default='vd_image_feats.pt', type=str)
    parser.add_argument('-batch_size', default=1440, type=int)
    parser.add_argument('-threshold', default=50, type=int, help="criterion to determine in-domain data")
    params = vars(parser.parse_args(args=None))

    # device setting
    if isinstance(params["gpu_ids"], int):
        params["gpu_ids"] = [params["gpu_ids"]]
        
    device = (
        torch.device("cuda", params["gpu_ids"][0]) 
        if params["gpu_ids"][0] >= 0
        else torch.device("cpu")
    )
    params['device'] = device


    if params['step'] == 'build':
        # Out-of-distribution detection model building step
        # We build multivariate normal distribution by computing covariance matrix and mean 

        # load image to path json from visual dialog data
        idx2imgpath = json.load(open(params['imgpath'], "r"))
        params['idx2imgpath'] = idx2imgpath
        params['data_name'] = 'VD'

        model, preprocess = clip.load("ViT-B/32", device=device)
        model = torch.nn.DataParallel(model.visual, params["gpu_ids"])

        # dataloader
        dataset = Dset(params, preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=8,
            pin_memory=False,
            collate_fn=collate_fn
        )
        print("num_data points: ", dataset.numDataPoints)
        image_feats = []

        for epoch_id, idx, batch in batch_iter(dataloader, params):  
            with torch.no_grad():
                image_feat = batch['image_data'].to(device)
                image_feat = image_feat.type(model.module.conv1.weight.dtype)
                image_feat = model(image_feat)
                image_feats.append(image_feat)

        image_feats = torch.cat(image_feats, 0)
        print("img feats size: ", image_feats.size())
        torch.save(image_feats, "./vd_image_feats.pt")

    else:
        # compute the likelihood of unknown data point with multivariate normal distribution
        # load cc12m idx to image path json
        idx2imgpath = json.load(open(params['imgpath'], "r"))
        params['idx2imgpath'] = idx2imgpath
        params['data_name'] = 'CC12M'

        # clip model load
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = torch.nn.DataParallel(model.visual, params["gpu_ids"])

        # dataloader
        dataset = Dset(params, preprocess)
        dataloader = DataLoader(
            dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=16,
            pin_memory=False,
            collate_fn=collate_fn
        )
        print("\nnum_data points: ", dataset.numDataPoints)
        
        # load pre-extracted image features to build multivariate normal distribution 
        image_feats = torch.load(params['feats'])
        print("img feats size: ", image_feats.size())

        # compute covariance matrix & mean vector
        # define OOD detector with multivariate gaussian
        cov, mean = cov_mean(image_feats.t(), rowvar=True)
        OOD_detector = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=cov)
        json_score = []

        for epoch_id, idx, batch in batch_iter(dataloader, params):  
            with torch.no_grad():
                image_feat = batch['image_data'].to(device)
                image_feat = image_feat.type(model.module.conv1.weight.dtype)
                image_feat = model(image_feat)

                log_probs = OOD_detector.log_prob(image_feat)
                for i in range(log_probs.size(0)):
                    json_score.append(
                        {
                            "image_id": batch['image_id'][i].item(),
                            "log_prob": log_probs[i].item()
                        }                         
                    )

        json.dump(json_score, open(params['data_name'] + "_logprobs.json", "w"))
