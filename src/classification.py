import os
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.datasets as dset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import open_clip
import torch.nn.functional as F
import pickle
from open_clip.transform import image_transform

def torch_save(classifer, save_path="./"):
    if os.path.dirname(save_path) != '':
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(classifer.cpu(), f)
        
def torch_load(save_path, device=None):
    with open(save_path, 'rb') as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier

def merge_yes_no_feature(dataset, model, device):
    txt = []
    N = len(dataset.classes)
    model.eval()
    if N:
        with open("./prompt/prompt.txt") as f:
            prompt_lis = f.readlines()
        num_prom = len(prompt_lis)
    for idx in range(num_prom):
        for name in dataset.classes:
            txt.append(open_clip.tokenize(prompt_lis[idx].replace("\n", "").format(name), 77).unsqueeze(0))
    txt = torch.cat(txt, dim=0)
    txt = txt.reshape(num_prom, len(dataset.classes), -1)
    text_inputs = txt.to(device)
    
    text_yes_ttl = torch.zeros(len(dataset.classes), 512).to(device)
    text_no_ttl = torch.zeros(len(dataset.classes), 512).to(device)
    
    with torch.no_grad():
        for i in range(num_prom):
            text_yes_i = model.encode_text(text_inputs[i])
            text_yes_i = F.normalize(text_yes_i, dim=-1)
            text_no_i = model.encode_text(text_inputs[i], "no")
            text_no_i = F.normalize(text_no_i, dim=-1)
            
            text_yes_ttl += text_yes_i
            text_no_ttl += text_no_i
            
    return F.normalize(text_yes_ttl, dim=-1), F.normalize(text_no_ttl, dim=-1)
    

class ViT_Classifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head_yes, classification_head_no):
        super().__init__()
        self.image_encoder = image_encoder
        flag = True
        self.fc_yes = nn.Parameter(classification_head_yes, requires_grad=flag)    # num_classes  num_feat_dimension
        self.fc_no = nn.Parameter(classification_head_no, requires_grad=flag)      # num_classes  num_feat_dimension
        self.scale = 100. # this is from the parameter of logit scale in CLIPN
        
    def set_frozen(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = False
    def set_learnable(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = True
            
    def forward(self, x):
        inputs = self.image_encoder(x)
        inputs_norm = F.normalize(inputs, dim=-1)
        fc_yes = F.normalize(self.fc_yes, dim=-1)
        fc_no = F.normalize(self.fc_no, dim=-1)
        
        logits_yes = self.scale * inputs_norm @ fc_yes.T 
        logits_no = self.scale * inputs_norm @ fc_no.T
        return logits_yes, logits_no, inputs
    
    def save(self, path = "./"):
        torch_save(self, path)
        
    @classmethod
    def load(cls, filename = "./", device=None):
        return torch_load(filename, device)
    
        
def load_model(model_type='ViT-B-16', pre_train="./", dataset = None, device=None):
    model, process_train, process_test = open_clip.create_model_and_transforms(model_type, pretrained=pre_train, device=device, freeze = False)
    weight_yes, weight_no = merge_yes_no_feature(dataset, model, device)
    vit_classifier =  ViT_Classifier(model.visual, weight_yes, weight_no)
    return vit_classifier, process_train, process_test
