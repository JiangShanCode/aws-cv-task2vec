import os
import torch
import torch.nn as nn
import logging
import numpy as np
import task_similarity

from basicsr.archs.nas_arch import NAS
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms 
from natsort import natsorted
from basicsr.utils import  scandir

class mydata(Dataset): 
    def __init__(self, gt, lr, base_transform=None):
        super(mydata, self).__init__()
        self.gt = gt
        self.base_transform = base_transform 
        self.imgs_path_gt = natsorted(list(scandir(self.gt, recursive=True, full_path=False)))
        self.imgs_gt = []
        for path in self.imgs_path_gt:
            with open(os.path.join(self.gt,path), 'rb') as f:
                # print(os.path.join(root,path))
                img = Image.open(f).convert('RGB')
                self.imgs_gt.append(img)

        self.lr = lr
        self.base_transform = base_transform 
        self.imgs_path_lr = natsorted(list(scandir(self.lr, recursive=True, full_path=False)))
        self.imgs_lr = []
        for path in self.imgs_path_lr:
            with open(os.path.join(self.lr,path), 'rb') as f:
                # print(os.path.join(root,path))
                img = Image.open(f).convert('RGB')
                self.imgs_lr.append(img)
        print("dataloder done")

    def __len__(self):
        return len(self.imgs_lr)


    def __getitem__(self, index):
        imgs_lr = self.imgs_lr[index]
        imgs_gt = self.imgs_gt[index]
    
        if self.base_transform is not None:
            imgs_lr = self.base_transform(imgs_lr)
            imgs_gt = self.base_transform(imgs_gt)

        return imgs_lr,imgs_gt

class Embedding:
    def __init__(self, hessian, scale, meta=None):
        self.hessian = np.array(hessian)
        self.scale = np.array(scale)
        self.meta = meta

class Task2Vec:
    def __init__(self, model):
        self.model = model
        # Fix batch norm running statistics (i.e., put batch_norm layers in eval mode)
        self.model.train()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = nn.L1Loss() 
        self.loss_fn = self.loss_fn.to(self.device)

    def montecarlo_fisher(self, dataset: Dataset, epochs: int = 1):
            logging.info("Using montecarlo Fisher")
            blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            data_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=True,
                              num_workers=6,
                              pin_memory=True,
                              drop_last=True)
            
            logging.info("Computing Fisher...")

            for p in self.model.parameters():
                p.grad2_acc = torch.zeros_like(p.data)
                p.grad_counter = 0
            for k in range(epochs):
                logging.info(f"\tepoch {k + 1}/{epochs}")
                for i, (data, target) in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
                    data = data.to(self.device)
                    output = self.model(data,8)

                    # The gradients used to compute the FIM needs to be for y sampled from
                    # the model distribution y ~ p_w(y|x), not for y from the dataset
                    # if self.bernoulli:
                    #     target = torch.bernoulli(F.sigmoid(output)).detach()
                    # else:
                    #     target = torch.multinomial(F.softmax(output, dim=-1), 1).detach().view(-1)
                    
                    # if torch.randn(1) > 0:
                    #     target = target.to(self.device)
                    # else:
                    #     target = blur(target).to(self.device)
                    
                    target = 0.999 * output + 0.001 * target.to(self.device)
                    loss = self.loss_fn(output, target)

                    self.model.zero_grad()
                    loss.backward()

                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad2_acc += p.grad.data ** 2
                            p.grad_counter += 1
            for p in self.model.parameters():
                if p.grad_counter == 0:
                    del p.grad2_acc
                else:
                    p.grad2_acc /= p.grad_counter
            logging.info("done")

    def extract_embedding(self, model):
        """
        Reads the values stored by `compute_fisher` and returns them in a common format that describes the diagonal of the
        Fisher Information Matrix for each layer.

        :param model:
        :return:
        """
        hess, scale = [], []
        for name, module in model.named_modules():
            # if module is model.classifier:
            #     continue
            # The variational Fisher approximation estimates the variance of noise that can be added to the weights
            # without increasing the error more than a threshold. The inverse of this is proportional to an
            # approximation of the hessian in the local minimum.
            if hasattr(module, 'logvar0') and hasattr(module, 'loglambda2'):
                logvar = module.logvar0.view(-1).detach().cpu().numpy()
                hess.append(np.exp(-logvar))
                loglambda2 = module.loglambda2.detach().cpu().numpy()
                scale.append(np.exp(-loglambda2).repeat(logvar.size))
            # The other Fisher approximation methods directly approximate the hessian at the minimum
            elif hasattr(module, 'weight') and hasattr(module.weight, 'grad2_acc'):
                grad2 = module.weight.grad2_acc.cpu().detach().numpy()
                filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
                hess.append(filterwise_hess)
                scale.append(np.ones_like(filterwise_hess))
        return Embedding(hessian=np.concatenate(hess), scale=np.concatenate(scale), meta=None)
    
    def embed(self, dataset: Dataset):
        # # Cache the last layer features (needed to train the classifier) and (if needed) the intermediate layer features
        # # so that we can skip the initial layers when computing the embedding
        # if self.skip_layers > 0:
        #     self._cache_features(dataset, indexes=(self.skip_layers, -1), loader_opts=self.loader_opts,
        #                          max_samples=self.max_samples)
        # else:
        #     self._cache_features(dataset, max_samples=self.max_samples)
        # # Fits the last layer classifier using cached features
        # self._fit_classifier(**self.classifier_opts)

        # if self.skip_layers > 0:
        #     dataset = torch.utils.data.TensorDataset(self.model.layers[self.skip_layers].input_features,
        #                                              self.model.layers[-1].targets)
        self.montecarlo_fisher(dataset)
        embedding = self.extract_embedding(self.model)
        return embedding

def define_model(model_path):
    model = NAS(
            upscale= 4,
            block = 8,
            feature = 48,
            output_filter = 2)
    loadnet = torch.load(model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    # print(keyname)
    model.load_state_dict(loadnet[keyname], strict=True)
    return model

def main():
    dataset_names = ["LoLTF","LoL","FIFA17","CSGO","CSGO_2","PU"]
    dataset_paths =[{"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_LoLTF_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_LoLTF_sub"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_LoL_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_LoL_sub"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_FIFA17_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_FIFA17_sub"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_CSGO_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_CSGO_sub"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_CSGO_2_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_CSGO_2_sub"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_PU_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_PU_sub"}]
    
    model_path = "/root/workspace/LiveSR/experiments/train_NAS_Ultra_SRx4_Game/models/net_g_latest.pth"
    tr_transforms = transforms.Compose([
        transforms.ToTensor()])
    
    dataset_list = []
    for dataset_path in dataset_paths:
        dataset_list.append( mydata(dataset_path["gt"],dataset_path["lr"],tr_transforms))

    embeddings = []
    for name, dataset in zip(dataset_names, dataset_list):
        print(f"Embedding {name}")
        probe_network = define_model(model_path)
        probe_network = probe_network.to("cuda:0")
        embeddings.append( Task2Vec(probe_network).embed(dataset) )
    task_similarity.plot_distance_matrix(embeddings, dataset_names)
    

if __name__ == '__main__':
    main()