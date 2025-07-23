import os
import sys
from collections import OrderedDict
import logging
import pickle

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils import data
import clip

logger = logging.getLogger(__name__)

# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

prompt = "This is an indoor scene image of a {}."
# prompt = "This is an scene image of a {}."

class DataLoader:
    def __init__(self, fea_dir='', img_dir='', device='cpu', batch_size=32, num_workers=1):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = img_dir
        self.fea_dir = fea_dir
        self.data = OrderedDict()
        self.read_feature()

    def read_feature(self):
        if self.fea_dir == '':
            logger.info('feature file not found, generating feature file...')
            self.read_image()

        if os.path.exists(self.fea_dir):
            with open(self.fea_dir, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.read_image()

    def read_image(self):
        if self.img_dir == '':
            logger.error('image directory not found!')
            sys.exit()

        # model, preprocess = clip.load('RN101', device=self.device)
        model, preprocess = clip.load('ViT-L/14', device=self.device)
        model.eval()
        model.float()

        # resnet101 = models.resnet101(pretrained=True).to(self.device)
        # resnet101 = nn.Sequential(*list(resnet101.children())[:-1]).eval()

        # data_transforms = transforms.Compose([transforms.Resize(448), transforms.CenterCrop(448), transforms.ToTensor(),
        #                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        dataset = CustomDataset(self.img_dir, preprocess)
        self.data['label_dic'] = dataset.label_dict
        self.data['label_list'] = dataset.label_list
        text_descriptions = [prompt.format(label) for label in dataset.label_list]
        text_tokens = clip.tokenize(text_descriptions).to(self.device)
        text_features = model.encode_text(text_tokens).float()
        # self.data['text_features'] = text_features / text_features.norm(dim=-1, keepdim=True)
        self.data['text_features'] = text_features.detach().cpu().numpy()

        self.data['label_num'] = dataset.label_num
        datasetloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                                     num_workers=self.num_workers)
        with torch.no_grad():
            all_features = []
            for _, imgs in enumerate(datasetloader):
                imgs = imgs.to(self.device)
                features = model.encode_image(imgs)
                all_features.append(features.cpu().numpy())
            all_features = np.concatenate(all_features, axis=0)
            self.data['feature'] = all_features
        pickle.dump(self.data, open(self.fea_dir, 'wb'))


class CustomDataset(data.Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.label_dict = OrderedDict()
        self.label_num = OrderedDict()
        self.label_list = []
        self.path_list = []
        i = 0
        for root, dirs, files in os.walk(img_dir):
            dirs.sort()
            files.sort()
            if len(files) > 0:
                label = root.split('/')[-1]
                self.label_num[label] = len(files)
                for file in files:
                    self.label_dict[i] = label
                    self.path_list.append(self.img_dir + '/' + label + '/' + file)
                    i += 1
            else:
                self.label_list = dirs

    def __getitem__(self, index):
        image = Image.open(self.path_list[index])
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.label_dict.__len__()
