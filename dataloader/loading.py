import os, torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  
import numpy as np
import dataloader.transforms as trans
import pickle
import pandas as pd
from typing import Optional, Tuple

class BUDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                #trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])


    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size


class APTOSDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        #print(self.size)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                #trans.adjust_light(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label

    def __len__(self):
        return self.size



class ISICDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (224,224)
        self.train = train
        with open(data_list, "rb") as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)

        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        #self.depths_transform = transforms.Compose([transforms.Resize((self.trainsize, self.trainsize)),transforms.ToTensor()])

    def __getitem__(self, index):
        data_pac = self.data_list[index]
        img_path = data_pac['img_root']
        #cl_img, cr_img, ml_img, mr_img = None
        img = Image.open(img_path).convert('RGB')

        img_torch = self.transform_center(img)

        label = int(data_pac['label'])

        
        return img_torch, label


    def __len__(self):
        return self.size
    
class MyDataset(Dataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True, val: bool = False, fold_n: int = -1):
        self.trainsize = (224,224)
        self.train = train
        self.val = val
        self.root = root
        self.fold_n = fold_n

        # Defining .csv file to be used
        if self.train or self.val:
            csv_name = csv_train
        else:
            csv_name = csv_test
            
        # Opening dataframe:
        self.df = pd.read_csv(csv_name, header=0)

    def __len__(self):
        return len(self.df)
    
    @property
    def labels_balance(self):
        y_series = self.df[self.y]
        v_counts = y_series.value_counts(normalize=True)
        sorted_v_counts = v_counts.sort_index()
        return np.asarray(sorted_v_counts)
    
class PadUfes20(MyDataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True, val: bool = False, fold_n: int = -1):
        super(PadUfes20, self).__init__(root, csv_train, csv_test, train, val, fold_n)
        if self.train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomVerticalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.x = "img_id"
        self.y = "diagnostic_number"
        
        if self.val:
            val_indexes = self.df["folder"] == fold_n + 1
            self.df = self.df[val_indexes].reset_index()
        elif self.train:
            train_indexes = self.df["folder"] != fold_n + 1
            self.df = self.df[train_indexes].reset_index()
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        img = self.transform_center(img)
        return img, int(self.df.loc[index][self.y])

class PNdbUfes(MyDataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True, val: bool = False, fold_n: int = -1):
        super(PadUfes20, self).__init__(root, csv_train, csv_test, train, val, fold_n)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.x = "path"
        self.y = "label_number"
        
        if self.val:
            val_indexes = self.df["folder"] == fold_n + 1
            self.df = self.df[val_indexes].reset_index()
        elif self.train:
            train_indexes = self.df["folder"] != fold_n + 1
            self.df = self.df[train_indexes].reset_index()
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        img = self.transform_center(img)
        return img, int(self.df.loc[index][self.y])
    
class HIBADataset(MyDataset):
    def __init__(self, root: str, csv_train: str, csv_test: str, train: bool = True):
        super(HIBADataset, self).__init__(root, csv_train, csv_test, train)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.x = "image"
        self.y = "diagnosis_label"
        
        # Adding image names
        self.df[self.x] = self.df["isic_id"] + ".JPG"
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform_center(img)
            
        return img, int(self.df.loc[index][self.y])
    
class HIBABinaryDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.trainsize = (224,224)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.train = train
        self.root = root
        
        # Defining .csv file to be used
        csv_name = "simplified_data.csv"
            
        # Opening dataframe:
        self.df = pd.read_csv(os.path.join(self.root, csv_name), header = 0)
        self.x = "image"
        self.y = "diagnosis_id"
        
        if self.train:
            self.df = self.df[self.df["fold"] != 0].reset_index()
        else:
            self.df = self.df[self.df["fold"] == 0].reset_index()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform_center(img)
            
        return img, int(self.df.loc[index][self.y])
    
class HIBASixDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.trainsize = (224,224)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.train = train
        self.root = root
        
        # Defining .csv file to be used
        csv_name = "simplified_data.csv"
            
        # Defining .csv file to be used
        if self.train:
            csv_name = "hiba6_parsed_train.csv"
        else:
            csv_name = "hiba6_parsed_test.csv"
            
        # Opening dataframe:
        self.df = pd.read_csv(os.path.join(self.root, csv_name), header = 0)
        self.x = "image"
        self.y = "diagnosis_label"
        
        # Adding image names
        self.df[self.x] = self.df["isic_id"] + ".JPG"

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform_center(img)
            
        return img, int(self.df.loc[index][self.y])
    

class HIBASixClinicalDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.trainsize = (224,224)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.train = train
        self.root = root
        
        # Defining .csv file to be used
        csv_name = "simplified_data.csv"
            
        # Defining .csv file to be used
        if self.train:
            csv_name = "hiba6_clinical_train.csv"
        else:
            csv_name = "hiba6_clinical_test.csv"
            
        # Opening dataframe:
        self.df = pd.read_csv(os.path.join(self.root, csv_name), header = 0)
        self.x = "image"
        self.y = "diagnosis_label"
        
        # Adding image names
        self.df[self.x] = self.df["isic_id"] + ".JPG"

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform_center(img)
            
        return img, int(self.df.loc[index][self.y])    

class LIPAIDataset(Dataset):
    def __init__(self, root: str, train: bool = True):
        self.trainsize = (224,224)
        if train:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                trans.RandomHorizontalFlip(),
                trans.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform_center = transforms.Compose([
                trans.CropCenterSquare(),
                transforms.Resize(self.trainsize),
                #trans.CenterCrop(self.trainsize),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
        self.train = train
        self.root = root
        
        # Defining .csv file to be used
        csv_name = "data.csv"
            
        # Opening dataframe:
        self.df = pd.read_csv(os.path.join(self.root, csv_name), header = 0)
        self.df["path"] = self.df["diagnosis"] + "/" + self.df["id"] + ".tif"
        self.x = "path"
        self.y = "discretized_diagnosis"
        
        if self.train:
            self.df = self.df[self.df["train"]].reset_index()
        else:
            self.df = self.df[~self.df["train"]].reset_index()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(index):
            index = index.tolist()
                
        img_path = os.path.join(self.root, "images", self.df.loc[index][self.x])
        img = Image.open(img_path).convert('RGB')
        
        img = self.transform_center(img)
            
        return img, int(self.df.loc[index][self.y])