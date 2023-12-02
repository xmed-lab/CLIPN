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

def merge_yes_no_feature(dataset, model, device, model_type):
    txt = []
    dimension = 512 if "ViT-B" in model_type else 768
    N = len(dataset.classes)
    if N:
        with open("/home/hwangfd/open_clip/src/prompt.txt") as f:
            yes_lines = f.readlines()
        num_prom = len(yes_lines)
    else:
        num_prom = 8

    for prom in range(num_prom):
        for name in dataset.classes:
            txt.append(open_clip.tokenize(name, 77, "test", prom).unsqueeze(0))
    txt = torch.cat(txt, dim=0)
    txt = txt.reshape(num_prom, len(dataset.classes), 2, -1)
    text_inputs = txt.to(device)
    text_yes_ttl = torch.zeros(len(dataset.classes), dimension).to(device)
    text_no_ttl = torch.zeros(len(dataset.classes), dimension).to(device)
    with torch.no_grad():
        for i in range(num_prom):
            text_yes_i = model.encode_text(text_inputs[i,:,0,:])
            text_yes_i = F.normalize(text_yes_i, dim=-1)
            text_yes_ttl += text_yes_i
            text_no_i = model.encode_text(text_inputs[i,:,1,:], "no")
            text_no_i = F.normalize(text_no_i, dim=-1)
            text_no_ttl += text_no_i        
    return F.normalize(text_yes_ttl, dim=-1), F.normalize(text_no_ttl, dim=-1)

class ViT_Classifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head_yes, classification_head_no):
        super().__init__()
        self.image_encoder = image_encoder
        flag = True
        self.fc_yes = nn.Parameter(classification_head_yes, requires_grad=flag)    # num_classes  num_feat_dimension
        self.fc_no = nn.Parameter(classification_head_no, requires_grad=flag)      # num_classes  num_feat_dimension
        self.scale = 100
    def set_frozen(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = False
    def set_learnable(self, module):
        for module_name in module.named_parameters():
            module_name[1].requires_grad = True
            
    def forward(self, x):
        inputs = self.image_encoder(x)
        return inputs
    
    def save(self, path = "./"):
        torch_save(self, path)
        
    @classmethod
    def load(cls, filename = "./", device=None):
        return torch_load(filename, device)
    
        
def load_model(model_type='ViT-B-16', pre_train="./", dataset = None, device=None, is_fine_tune = False, filename = "./fine_tune/test.pt", multi = False):
    if is_fine_tune:
        vit_classifier = ViT_Classifier.load(filename = filename)
        
        image_mean = getattr(vit_classifier.image_encoder, 'image_mean', None)
        image_std = getattr(vit_classifier.image_encoder, 'image_std', None)
        preprocess_train = image_transform(vit_classifier.image_encoder.image_size, is_train=True, mean=image_mean, std=image_std)
    
        scale = 1
        image_size0, image_size1 = vit_classifier.image_encoder.image_size
        image_size0, image_size1 = image_size0 * scale, image_size1 * scale
        preprocess_val = image_transform((image_size0, image_size1), is_train=False, mean=image_mean, std=image_std)

        return vit_classifier, preprocess_train, preprocess_val
    else:
        if not multi:
            model, process_train, process_test = open_clip.create_model_and_transforms(model_type, pretrained=pre_train, device=device, freeze = False)
            weight_yes, weight_no = merge_yes_no_feature(dataset, model, device, model_type)
        #weight_yes, weight_no = torch.randn(len(dataset.classes), 512).to(device), torch.randn(len(dataset.classes), 512).to(device)
            vit_classifier =  ViT_Classifier(model.visual, weight_yes, weight_no)
    return vit_classifier, process_train, process_test
#pre_train="./logs/16-400M-0.1-0.5_1.5/checkpoints/epoch_1.pt"
#pre_train="laion400m_e32"
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
cifar10 = CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False)
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
vit_class, _, process = load_model(pre_train="./logs/16-400M-0.1-0.5_1.5/checkpoints/epoch_1.pt", dataset=cifar10)

'''