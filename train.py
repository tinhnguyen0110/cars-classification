import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import glob 
import os
import cv2
from PIL import Image
import random
from dataset import get_data_loader
from tqdm import tqdm
import time
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import albumentations as A
from utils import plot_acc,plot_loss
import shutil
from model import build_model
import argparse
import pandas as pd
 #set seed
seed = 42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-e', '--epochs', type=int, default=70,
    help='Number of epochs to train our network for'
)
parser.add_argument(
    '-bs_train', '--batch-size-train', type=int,
    dest='batch_size_train', default=32,
    help='batch size data train for training the model'
)
parser.add_argument(
    '-bs_test', '--batch-size-test', type=int,
    dest='batch_size_test', default=16,
    help='batch size data test for training the model'
)

parser.add_argument(
    '-lr_backbone', '--learning-rate-backbone', type=float,
    dest='learning_rate_backbone', default=3e-5,
    help='Learning rate for training the model'
)
parser.add_argument(
    '-lr_head', '--learning-rate-head', type=float,
    dest='learning_rate_head', default=3e-4,
    help='Learning rate for training the model'
)

parser.add_argument(
    '-path_train', '--path-train', type=str,
    default='/home/a4000/Data/tinhnv/classifier_car/Data_training/Data_test/',
    help='Path Data for training the model'
)

parser.add_argument(
    '-name_label', '--name-label',
    default= None,
    help='list name label'
)

parser.add_argument(
    '-save_img_training', '--save-img-training', type=bool,
    default= False,
    help='check img train false class for training model'
)


args = vars(parser.parse_args())



def train_model(model, criterion, optimizer, num_epochs=3,epoch_start=0,save_img_false=False):
    loss_train = []
    loss_val = []
    acc_train = []
    acc_val = []
    for epoch in range(epoch_start,num_epochs):
        best_acc_val = 0
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            count = 1 
            tepoch =  tqdm(Dataloader[phase])
            for inputs, labels, path in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() 
                running_corrects += (torch.sum(preds == labels.data)/ len(inputs)).item() 
                loss_training = running_loss/count
                acc_training = running_corrects/count

                if save_img_false and epoch >= 67:
                    path_save = 'output/training/epoch_' + str(epoch)
                    isExist = os.path.exists(path_save)
                    if not isExist:
                        os.makedirs(path_save)
                    check_false = preds == labels.data
                    for idx_check,i in enumerate(check_false):
                        if i == False:
                            output_sigmoid = torch.softmax(outputs[idx_check],dim=0)
                            out_sort, indices = torch.sort(output_sigmoid,descending=True)
                            shutil.copy(path[idx_check],path_save + "/" + name_label[labels.data[idx_check]] +
                            "-{}_{:.2f}_{}_{:.2f}_{}_{:.2f}-".format(
                                name_label[indices[0]], out_sort[0], name_label[indices[1]], 
                                out_sort[1],name_label[indices[2]], out_sort[2]) + path[idx_check].split('/')[-1])
                            
                if phase == 'train':
                    tepoch.set_postfix(Loss_train=f" {loss_training:.4f}",acc_train=f"{acc_training:.4f}")

                else:
                    tepoch.set_postfix(Loss_val=f" {loss_training:.4f}",acc_val=f"{acc_training:.4f}")

                time.sleep(0.1)
                count +=1 

            if acc_training > best_acc_val and phase == 'validation':
                
                if not os.path.exists('output/checkpoint'):
                        os.makedirs(path_save)
                
                best_acc_val = acc_training
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, 'output/checkpoint/resnet152_best.pt')
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     }, f'output/checkpoint/resnet152_epoch_{epoch}.pt')
            if phase == 'train':
                loss_train.append(round(loss_training,4))
                acc_train.append(round(acc_training,4))
            else:
                loss_val.append(round(loss_training,4))
                acc_val.append(round(acc_training,4))    
        scheduler.step()
        plot_acc(acc_train,acc_val)
        plot_loss(loss_train,loss_val)
        print('---'*20)
    print('TRAINING COMPLETE')
    return model



if __name__ == '__main__':
    
    path_data = args['path_train']
    
    name_label =  args['name_label']
    num_epochs = args['epochs']
    
    batch_size_train = args['batch_size_train']
    batch_size_test = args['batch_size_test']
    
    
    
    save_img_false = args['save_img_training']
    
    Dataloader = get_data_loader(path_data,batch_size_train,batch_size_test,name_label)

    f = open("class_names.py", "w")
    f.write("class_names = "+str(Dataloader['train'].dataset.name_label))
    f.close()

    name2lenght = {}
    name2lenght['Version Data'] = path_data.split('/')[-2]

    if name_label == None:
        name_label= Dataloader['train'].dataset.name_label
    for i in glob.glob(path_data + '/*'):
        for j in glob.glob(i + '/*'):
            name2lenght[j.split('/')[-1]] = len(glob.glob(j + "/*"))
        df = pd.DataFrame([name2lenght])
        df = df.reindex(columns= ['Version Data'])
        df.to_csv('output/report_data_' + i.split('/')[-1] + '.csv',index=False)

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lr_1 = args['learning_rate_backbone']
    lr_2 = args['learning_rate_head']
    
    num_classes = Dataloader['train'].dataset.classes
    print('Classes: ', Dataloader['train'].dataset.name_label)
    print('Number of classes: ', num_classes)
    print(f"Epochs to train for: {num_epochs}")
    
    model = build_model(num_classes=num_classes)
    model.to(device)

    parameters = []
    for name,param in model.named_parameters():
        if 'classifier' in name:
            parameters += [{'params': [param],'lr':lr_2}]
        else:
            parameters += [{'params': [param],'lr':lr_1}]

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(parameters)

    # scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)
    lambda1 = lambda epoch: 0.9 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    model_trained = train_model(model, criterion, optimizer, num_epochs=num_epochs,save_img_false=save_img_false)

# python train.py -path_train ../../Data/Data_training_0604/ -bs_train 32 -bs_test 16 -save_img_training True 
# python train.py -path_train ../../Data_training/Data_test/ -save_img_training True -bs_train 48 -bs_test 16 