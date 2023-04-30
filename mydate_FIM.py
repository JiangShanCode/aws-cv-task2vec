import os,shutil
import torch
import torch.nn as nn
import logging
import numpy as np
import task_similarity
import variational
import copy
from basicsr.archs.nas_arch import NAS
from basicsr.archs.edsr_arch import EDSR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms 
from natsort import natsorted
from basicsr.utils import  scandir
from functorch import make_functional_with_buffers, vmap, grad

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

class Task2Vec_My:
    def __init__(self, model):
        # self.fmodel, self.params, self.buffers = make_functional_with_buffers(model)
        self.model = model
        # print(self.model )
        
        self.perturbed_model = self.transform_weights(self.model)
        # print(self.perturbed_model)
        # for name, param in self.perturbed_model.named_parameters():
        #     if name == "networks.tail.0.weight":
        #         print(name,param)
        
        # for name, param in self.model.named_parameters():
        #     if name == "networks.tail.0.weight":
        #         print(name,param)
        # Fix batch norm running statistics (i.e., put batch_norm layers in eval mode)
        # self.model.train()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss_fn = nn.L1Loss() 
        self.loss_fn = self.loss_fn.to(self.device)
        # self.ft_compute_grad = grad(self.compute_loss_stateless_model)

    # def compute_loss_stateless_model (self, params, buffers, sample):
    #     predictions = self.fmodel(params, buffers, sample) 
    #     targets = 0.999 * predictions
    #     loss = self.loss_fn(predictions, targets)
    #     return loss
    
    # def compute_grad(self, sample):

    #     prediction = self.model(sample)
    #     target = 0.999 * prediction
    #     loss = self.loss_fn(prediction, target)

    #     return torch.autograd.grad(loss, list(self.model.parameters()))
    
    # def compute_sample_grads(self, data):
    #     """ manually process each sample with per sample gradient """
    #     sample_grads = [self.compute_grad(data[i]) for i in range(32)]
    #     sample_grads = zip(*sample_grads)
    #     sample_grads = [torch.stack(shards) for shards in sample_grads]
    #     return sample_grads

    def transform_weights(self,model):
        model_copy = copy.deepcopy(model)
        
        for name, param in model_copy.named_parameters():
            if "weight" in name:
                # create a new random tensor
                # mask = torch.FloatTensor(param.size()).uniform_() > 0.5
                mask = torch.FloatTensor([0.999])
                param.data.mul_(mask.cuda())
        return model_copy
    
    def montecarlo_fisher(self, dataset: Dataset, epochs: int = 1):
        
        logging.info("Using montecarlo Fisher")
        data_loader = DataLoader(dataset=dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=6,
                            pin_memory=True,
                            drop_last=True)
        
        
        # ft_compute_sample_grad = vmap(self.ft_compute_grad, in_dims=(None,None,0))
        

        logging.info("Computing Fisher...")

        for p in self.model.parameters():
            p.grad2_acc = torch.zeros_like(p.data)
            p.grad_counter = 0

        for k in range(epochs):
            logging.info(f"\tepoch {k + 1}/{epochs}")

            for id, (data, target) in enumerate(tqdm(data_loader, leave=False, desc="Computing Fisher")):
                # print(id)
                data = data.to(self.device)
                # target = target.to(self.device)
                # print(data.size())

                # per_sample_grads = self.compute_sample_grads(data)
                # # for i in per_sample_grads:
                # #     print(i.shape)
                '''
                ft_per_sample_grads = ft_compute_sample_grad(self.params, self.buffers, data)
                for (name,p),grad in zip(self.model.named_parameters(),ft_per_sample_grads):
                    if "weight" in name:
                            # grad (B, grad) 
                            grad2 = grad.detach() ** 2
                            grad2 = grad2.mean(dim=0).squeeze()
                            p.grad2_acc += grad2
                            p.grad_counter += 1

                '''
                # for per_sample_grad, ft_per_sample_grad in zip(per_sample_grads, ft_per_sample_grads):
                #     assert torch.allclose(per_sample_grad, ft_per_sample_grad, atol=3e-3, rtol=1e-5)
                # [print(i.shape) for i in ft_per_sample_grads]

                output = self.model(data)
                # print("output",output)

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
                
                target = 0.5 * output 
                # target = self.perturbed_model(data)
                # print("target",target)

                loss = self.loss_fn(output, target)

                self.model.zero_grad()
                loss.backward()

                for p in self.model.parameters():
                    # print(p.grad2_acc)
                    # return
                    if p.grad is not None:
                        p.grad2_acc += p.grad.data ** 2
                        p.grad_counter += 1
                
        for p in self.model.parameters():
            if p.grad_counter == 0:
                del p.grad2_acc
            else:
                # print(p.grad_counter)
                p.grad2_acc /= p.grad_counter
        
        logging.info("done")

    def extract_embedding(self, is_model, model):
        """
        Reads the values stored by `compute_fisher` and returns them in a common format that describes the diagonal of the
        Fisher Information Matrix for each layer.

        :param model:
        :return:
        """

        if is_model:
            for name,p in model.named_parameters():
                # print(name)
                if p.grad_counter == 0:
                    del p.grad2_acc
                else:
                    p.grad2_acc /= p.grad_counter

        hess, scale = [], []
        # print(type(model))
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
                # print(grad2.shape)
                filterwise_hess = grad2.reshape(grad2.shape[0], -1).mean(axis=1)
                # print(filterwise_hess)
                # print(filterwise_hess.shape)
                hess.append(filterwise_hess)
                scale.append(np.ones_like(filterwise_hess))
                # print(np.ones_like(filterwise_hess).shape)
            
        # print(np.concatenate(hess).shape)
        return Embedding(hessian=np.concatenate(hess), scale=np.concatenate(scale), meta=None)
    
    def embed(self,is_model,dataset: Dataset = None):
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
        if not is_model:
            # print("is not model")
            self.montecarlo_fisher(dataset)
        # self.variational_fisher(dataset)
        embedding = self.extract_embedding(is_model,self.model)
        return embedding

def define_model(model_path):
    model = NAS(
            upscale= 4,
            block = 8,
            feature = 48,
            output_filter = 2)

    # model = EDSR(
    #     num_in_ch= 3,
    #     num_out_ch= 3,
    #     num_feat= 256,
    #     num_block= 32,
    #     upscale= 4,
    #     res_scale= 0.1,
    #     img_range= 255.,
    #     rgb_mean= [0.4488, 0.4371, 0.4040] 
    # )

    loadnet = torch.load(model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    # print(keyname)
    model.load_state_dict(loadnet[keyname], strict=True)

    if loadnet.get("FIM_dict"):
        for name,p in model.named_parameters():
            if loadnet["FIM_dict"].get("{}.grad_counter".format(name)):
                p.grad_counter = loadnet["FIM_dict"]["{}.grad_counter".format(name)]
                p.grad2_acc = loadnet["FIM_dict"]["{}.grad2_acc".format(name)]
            else:
                p.grad_counter = 0
                p.grad2_acc = torch.zeros_like(p.data)
    
    return model

def main():
    # dataset_names = ["PU_Model","LoLTF","LoL","FIFA17","CSGO","CSGO_2","PU"]
    # dataset_paths =[{"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_LoLTF_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_LoLTF_sub"},
    #                {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_LoL_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_LoL_sub"},
    #                {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_FIFA17_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_FIFA17_sub"},
    #                {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_CSGO_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_CSGO_sub"},
    #                {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_CSGO_2_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_CSGO_2_sub"},
    #                {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR_PU_sub","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4_PU_sub"}]
    # model_path = "/root/workspace/LiveSR/experiments/train_NAS_Ultra_SRx4_Game_Overfit_PU_FIM/models/net_g_latest.pth"

    dataset_names = ["LoLTF","LoL","FIFA17","CSGO","CSGO_2","PU"]
    dataset_paths =[{"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR/LoLTF","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4/LoLTF"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR/LoL","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4/LoL"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR/FIFA17","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4/FIFA17"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR/CSGO","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4/CSGO"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR/CSGO_2","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4/CSGO_2"},
                   {"gt":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/HR/PU","lr":"/root/workspace/LiveSR/datasets/GamingDataSet/Val/Bicubic/x4/PU"}]
    model_path = "/root/workspace/LiveSR/experiments/pretrained_models/train_NAS_Ultra_SRx4_Game_Frame/models/net_g_latest.pth"

    # dataset_names = [i for i in range(1,11)]
    # dataset_paths = [{"gt":"/root/workspace/LiveSR/datasets/OST300/datasets/OutdoorSceneTest300/HR_dir/{}".format("%03d" % id),"lr":"/root/workspace/LiveSR/datasets/OST300/datasets/OutdoorSceneTest300/Bicubic/x4_dir/{}".format("%03d" % id)} \
    #                  for id in range(1,11)]
    # model_path = "/root/workspace/LiveSR/experiments/pretrained_models/EDSR/206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb/models/net_g_300000.pth"
    
    tr_transforms = transforms.Compose([
        transforms.ToTensor()])
    
    dataset_list = []
    for dataset_path in dataset_paths:
        dataset_list.append( mydata(dataset_path["gt"],dataset_path["lr"],tr_transforms))

    embeddings = []
    # probe_network = define_model(model_path)
    # probe_network = probe_network.to("cuda:0")
    # embeddings.append( Task2Vec_My(probe_network).embed(is_model = True) )
    # print(Task2Vec_My(probe_network).embed(is_model = True).hessian)
    probe_network = define_model(model_path)
    probe_network = probe_network.to("cuda:0")
    task2vec = Task2Vec_My(probe_network)
    for name, dataset in zip(dataset_names[:], dataset_list):
        print(f"Embedding {name}")
        
        embeddings.append( task2vec.embed(is_model=False,dataset=dataset) )
        # print(task2vec.embed(is_model=False,dataset=dataset).hessian)
    task_similarity.plot_distance_matrix(embeddings, dataset_names)
    

if __name__ == '__main__':
    main()
    # imgs_path_gt = natsorted(list(scandir("/root/workspace/LiveSR/datasets/OST300/datasets/OutdoorSceneTest300/HR", recursive=True, full_path=False)))
    # i = 1
    # for img_path in imgs_path_gt:
    #     img_name = os.path.basename(img_path)
    #     print(img_name)
    #     os.makedirs(os.path.join("/root/workspace/LiveSR/datasets/OST300/datasets/OutdoorSceneTest300/HR_dir","%03d" % i),exist_ok=True)
    #     shutil.copy(os.path.join("/root/workspace/LiveSR/datasets/OST300/datasets/OutdoorSceneTest300/HR",img_path),os.path.join("/root/workspace/LiveSR/datasets/OST300/datasets/OutdoorSceneTest300/HR_dir","%03d" % i,img_name))
    #     i += 1