import json
import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .utils import my_pre_caption, mri_pre_catpion

import os
import pandas as pd

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]



class generation_mri_train(Dataset):
    def __init__(self, transform, ann_root, max_words=256, dataset='mimic_cxr', args=None):
        
        # self.annotation = json.load(open(os.path.join(ann_root),'r'))
        # self.ann = self.annotation['train']

        self.ann = pd.read_csv(ann_root, dtype=object)
        if args.sample_n != 0:      
            self.ann = self.ann.loc[:args.sample_n]

        self.ann = self.ann.reset_index(drop=True)
        self.transform = transform
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args

        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        def custom_sort(filename):
            numeric_part = ""
            for char in filename:
                if char.isdigit():
                    numeric_part += char
                else:
                    break

            if numeric_part:
                return int(numeric_part)
            else:
                return filename

        def resize_thickness(desired_thickness,x):
            current_thickness = x.shape[0]
            padding_thickness = desired_thickness - current_thickness

            # 创建填充参数
            # 注意：在这里，我们在每个维度的前后都填充相同数量的元素
            # 这样可以将厚度设置为24
            padding = (0,0,0,0,0,0,0, padding_thickness)

            # 使用 torch.pad 进行填充
            padded_image_data = torch.nn.functional.pad(x, padding, "constant", value=0)
            return padded_image_data

        if self.args.enable_3seq:
            while True:
                ann = self.ann.loc[index]
                image_id = '0' + str(ann['image']) if len(str(ann['image']))<14 else str(ann['image'])
                image_dir = '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)
                image_dir_list = os.listdir(image_dir)
                if ('1' in image_dir_list) and ('2' in image_dir_list) and ('3' in image_dir_list):
                    pic_new_list = []
                    for i in [1, 2, 3]:
                        pic_list = []
                        pic_dir =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(i)+'/'
                        pic_dir_list = os.listdir(pic_dir)
                        pic_dir_list.sort(key=custom_sort)
                        for j in pic_dir_list:
                            pic_path = '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(i)+'/'+j
                            pic = Image.open(pic_path).convert('RGB')
                            pic = self.transform(pic)
                            pic_list.append(pic)
                        pic_list = torch.stack(pic_list, 0)
                        pic_list = resize_thickness(24, pic_list)
                        pic_new_list.append(pic_list)
                    image = torch.stack(pic_new_list, dim=0)
     
                    cls_labels = eval(ann['encode_label'])[:self.args.num_label]
                    prompt = [SCORES[l] for l in cls_labels]
                    prompt = ' '.join(prompt)+' '
                    if self.args.ddp:
                        caption = prompt + ' '.join(self.tokenizer.tokenize(mri_pre_catpion(ann['诊断结论'], self.max_words)))
                    else:
                        caption = ' '.join(self.tokenizer.tokenize(mri_pre_catpion(ann['诊断结论'], self.max_words)))
                    cls_labels = torch.from_numpy(np.array(cls_labels)).long()
                    if self.args.cfe:
                        clip_indices =  eval(ann['clip_indices'])[:self.args.clip_k]
                        clip_memory = self.clip_features[clip_indices]
                    else:
                        clip_memory = torch.zeros([self.args.clip_k,768]).float()
                    return image, caption, cls_labels, clip_memory
                else:
                    index = random.randrange(len(self.ann))
                    continue

        else:
            ann = self.ann.loc[index] 
            image_id = ann['image']
            image_path =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(3)+'/'
            if not os.path.exists(image_path):
                # print(f'{image_path} not exists')
                # logging.info(f'{image_path} not exists')
                image_id = '0' + image_id
                image_path =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(3)+'/'
            if not os.path.exists(image_path):
                image_id = '00' + image_id
                image_path =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(3)+'/'
            pic_real_path = os.listdir(image_path)
            pic_real_path.sort(key=custom_sort) 
            pics = [Image.open(image_path+pic_real_path).convert('RGB') for pic_real_path in pic_real_path]
            pic_list_temp = []
            for pic in pics:
                transform_pic = self.transform(pic)
                pic_list_temp.append(transform_pic)
            pic_list_temp = torch.stack(pic_list_temp,0)
            image = resize_thickness(24,pic_list_temp)
           
            caption = mri_pre_catpion(ann['诊断结论'], self.max_words)#+self.tokenizer.eos_token

            cls_labels = eval(ann['encode_label'])[:self.args.num_label]
            cls_labels = torch.from_numpy(np.array(cls_labels)).long()

            # if self.args.condition:
            #     diagnosis_label = ''
            #     for idx, item in enumerate(cls_labels):
            #         if item==1:
            #             diagnosis_label = diagnosis_label + self.diagnosis_text[idx] + ';'
            #     prompt = '<|im_start|>' + diagnosis_label + '请根据以上信息生成膝关节诊断报告。'
            #     prompt_len = torch.tensor(len(self.tokenizer(prompt).input_ids))
            #     caption = prompt + caption
            #     return image, caption, prompt_len, cls_labels


            return image, caption, cls_labels

class generation_mri_eval(Dataset):
    def __init__(self, transform, ann_root, max_words=256, dataset='mimic_cxr', args=None):
        
        # self.annotation = json.load(open(os.path.join(ann_root),'r'))
        # self.ann = self.annotation['train']

        self.ann = pd.read_csv(ann_root, dtype=object)
        if args.sample_n != 0:      
            self.ann = self.ann.loc[:args.sample_n]        
        self.ann = self.ann.reset_index(drop=True)
        self.transform = transform
        self.max_words = max_words      
        self.dataset = dataset
        self.args = args
        train_local_text_features = torch.load('/mnt/nfs_share/XXX/data/train_local_text_features.pt')
        self.clip_features = torch.cat(train_local_text_features, dim=0)
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        def custom_sort(filename):
            numeric_part = ""
            for char in filename:
                if char.isdigit():
                    numeric_part += char
                else:
                    break

            if numeric_part:
                return int(numeric_part)
            else:
                return filename
        def resize_thickness(desired_thickness,x):
            current_thickness = x.shape[0]
            padding_thickness = desired_thickness - current_thickness

            # 创建填充参数
            # 注意：在这里，我们在每个维度的前后都填充相同数量的元素
            # 这样可以将厚度设置为24
            padding = (0,0,0,0,0,0,0, padding_thickness)

            # 使用 torch.pad 进行填充
            padded_image_data = torch.nn.functional.pad(x, padding, "constant", value=0)
            return padded_image_data

        if self.args.enable_3seq:
            ann = self.ann.loc[index]
            image_id = '0' + str(ann['image']) if len(str(ann['image']))<14 else str(ann['image'])
            image_dir = '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)
            image_dir_list = os.listdir(image_dir)
            #if ('1' in image_dir_list) and ('2' in image_dir_list) and ('3' in image_dir_list):
            pic_new_list = []
            for i in [1, 2, 3]:
                pic_list = []
                pic_dir =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(i)+'/'
                pic_dir_list = os.listdir(pic_dir)
                pic_dir_list.sort(key=custom_sort)
                for j in pic_dir_list:
                    pic_path = '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(i)+'/'+j
                    pic = Image.open(pic_path).convert('RGB')
                    pic = self.transform(pic)
                    pic_list.append(pic)
                pic_list = torch.stack(pic_list, 0)
                pic_list = resize_thickness(24, pic_list)
                pic_new_list.append(pic_list)
            image = torch.stack(pic_new_list, dim=0)

            cls_labels = eval(ann['encode_label'])[:self.args.num_label]
            prompt = [SCORES[l] for l in cls_labels]
            prompt = ' '.join(prompt)+' '
            if self.args.ddp:
                caption = prompt + ' '.join(self.tokenizer.tokenize(mri_pre_catpion(ann['诊断结论'], self.max_words)))
            else:
                caption = ' '.join(self.tokenizer.tokenize(mri_pre_catpion(ann['诊断结论'], self.max_words)))
            cls_labels = torch.from_numpy(np.array(cls_labels)).long()
            if self.args.cfe:
                clip_indices =  eval(ann['clip_indices'])[:self.args.clip_k]
                clip_memory = self.clip_features[clip_indices]
            else:
                clip_memory = torch.zeros([self.args.clip_k,768]).float()
            return image, caption, cls_labels, clip_memory
              

        ann = self.ann.loc[index] 
        image_id = ann['image']
        image_path =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(3)+'/'
        if not os.path.exists(image_path):
            # print(f'{image_path} not exists')
            # logging.info(f'{image_path} not exists')
            image_id = '0' + image_id
            image_path =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(3)+'/'
        if not os.path.exists(image_path):
            image_id = '00' + image_id
            image_path =  '/mnt/nvme_share/XXX/mri_pic/'+str(image_id)+ '/'+str(3)+'/'
        pic_real_path = os.listdir(image_path)
        pic_real_path.sort(key=custom_sort) 
        pics = [Image.open(image_path+pic_real_path).convert('RGB') for pic_real_path in pic_real_path]
        pic_list_temp = []

        for pic in pics:
            transform_pic = self.transform(pic)
            pic_list_temp.append(transform_pic)
        pic_list_temp = torch.stack(pic_list_temp,0)
        image = resize_thickness(24,pic_list_temp)
        caption = mri_pre_catpion(ann['诊断结论'], self.max_words)

        cls_labels = eval(ann['encode_label'])[:self.args.num_label] 
        cls_labels = torch.from_numpy(np.array(cls_labels)).long()

        return image, caption, cls_labels