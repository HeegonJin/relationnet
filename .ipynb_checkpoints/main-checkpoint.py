'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, Subset, SubsetRandomSampler, SequentialSampler

from torchvision import *
import torchvision
import torchvision.transforms as transforms
import torchvision.models as pool_models

import random
import numpy as np
import pandas as pd
import os
import argparse

# from models import *
from models.transformer_relation_net import tf_relation_net
from models.lstm_relation_net import lstm_relation_net

from models.cifar10_models import densenet, resnet, mobilenetv2, googlenet, inception, vgg
from efficientnet_pytorch import EfficientNet
from models.custom_resnext import CustomResNext
from models.custom_densenet import CustomDensenet

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from cf_matrix import make_confusion_matrix
from utils import progress_bar, copy_to_experiment_dir, save_config_file

from relation_dataset import RelationData
from data.NEU.dataset import NEUdataset
from data.PCAM.dataset import PCAMDataset
from data.CUB_200_2011.cub2011 import Cub2011
# from data.cars.dataset import CarsDataset

from losses import DistillationLoss
import yaml
import json

import sys

sys.path.append('/notebooks/relationnet/AutoAugment')
# from AutoAugment import *
from AutoAugment.autoaugment import CIFAR10Policy, ImageNetPolicy
from AutoAugment.cutout import Cutout
from AutoAugment.ops import *
from sklearn.metrics import confusion_matrix


SEED_NUM = 1995
torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)
torch.cuda.manual_seed(SEED_NUM)
torch.cuda.manual_seed_all(SEED_NUM) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)

###custom dataset trained models###
celeba_efficientnet = EfficientNet.from_name('efficientnet-b4', num_classes=36)
celeba_efficientnet._fc = torch.nn.Linear(in_features=celeba_efficientnet._fc.in_features, out_features=36, bias=False) 
celeba_efficientnet.load_state_dict(torch.load('./models/other_models/celeba_efficientnet.pth')) #https://www.kaggle.com/nikperi/pytorch-efficientnet #celebA dataset
leaf_resnext = CustomResNext(model_name='resnext50_32x4d', pretrained=False)
leaf_resnext.load_state_dict(torch.load('./models/other_models/leaf_resnext.pth')['model']) #https://www.kaggle.com/yasufuminakama/cassava-resnext50-32x4d-starter-training/output?select=resnext50_32x4d_fold0_best.pth #Cassava leaf disease dataset
flower_efficientnet = EfficientNet.from_name('efficientnet-b0', num_classes=17)
flower_efficientnet.load_state_dict(torch.load('./models/other_models/flower_effcientnet'))
dtd_efficientnet = EfficientNet.from_name('efficientnet-b0', num_classes=47)
dtd_efficientnet.load_state_dict(torch.load('./models/other_models/dtd_effcientnet'))
chestct_resnet = pool_models.resnet18(pretrained=False)
chestct_resnet.fc = torch.nn.Linear(in_features=512, out_features=14, bias=True)
chestct_resnet.load_state_dict(torch.load('./models/other_models/chestct_resnet.pth')) # https://www.kaggle.com/orkatz2/pulmonary-embolism-pytorch-train
flower104_densenet= CustomDensenet() #https://www.kaggle.com/dhananjay3/pytorch-xla-for-tpu-with-multiprocessing#Model
flower104_densenet.load_state_dict(torch.load('./models/other_models/flower104_densenet.pt'))
for param in celeba_efficientnet.parameters():
    celeba_efficientnet.requires_grad = False
for param in leaf_resnext.parameters():
    leaf_resnext.requires_grad = False
for param in flower_efficientnet.parameters():
    flower_efficientnet.requires_grad = False
for param in dtd_efficientnet.parameters():
    dtd_efficientnet.requires_grad = False
for param in chestct_resnet.parameters():
    chestct_resnet.requires_grad = False
for param in flower104_densenet.parameters():
    flower104_densenet.requires_grad = False
    
###custom dataset trained models###

model_pool = []

if config['model_pool']['other_pretrains']:
    model_pool.append(celeba_efficientnet)
    model_pool.append(leaf_resnext)
    model_pool.append(flower_efficientnet)
    model_pool.append(dtd_efficientnet)
    model_pool.append(chestct_resnet)
    model_pool.append(flower104_densenet)

if config['model_pool']['cifar10_pretrains']:
#     model_pool.append(vgg.vgg11_bn(pretrained=True))
# #     model_pool.append(vgg.vgg13_bn(pretrained=True))
    model_pool.append(vgg.vgg16_bn(pretrained=True))
# #     model_pool.append(vgg.vgg19_bn(pretrained=True))
    # model_pool.append(resnet.resnet18(pretrained=True))
    model_pool.append(resnet.resnet50(pretrained=True))
    # model_pool.append(densenet.densenet121(pretrained=True))
    model_pool.append(densenet.densenet161(pretrained=True))
    model_pool.append(mobilenetv2.mobilenet_v2(pretrained=True))
    model_pool.append(googlenet.googlenet(pretrained=True))
    model_pool.append(inception.inception_v3(pretrained=True))
    
if config['model_pool']['imagenet_pretrains']:
    model_pool.append(pool_models.resnet50(pretrained=True))
    model_pool.append(pool_models.squeezenet1_0(pretrained=True))
    model_pool.append(pool_models.vgg16_bn(pretrained=True))
    model_pool.append(pool_models.densenet161(pretrained=True))
    model_pool.append(pool_models.googlenet(pretrained=True))
    model_pool.append(pool_models.shufflenet_v2_x1_0(pretrained=True))
    model_pool.append(pool_models.mobilenet_v2(pretrained=True))
    model_pool.append(pool_models.inception_v3(pretrained=True))
    model_pool.append(pool_models.mobilenet_v3_large(pretrained=True))
    model_pool.append(pool_models.mobilenet_v3_small(pretrained=True))
    model_pool.append(pool_models.resnext50_32x4d(pretrained=True))
    model_pool.append(pool_models.wide_resnet50_2(pretrained=True))
    model_pool.append(pool_models.mnasnet1_0(pretrained=True))

print(f'modelpool num_models:{len(model_pool)}')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train_target = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=28),
    CIFAR10Policy(), 
	transforms.ToTensor(), 
    Cutout(n_holes=1, length=28), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# transform_train_target = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.RandomCrop(224, padding=32),
#     transforms.RandomRotation(10),
#     transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])


transform_train_relation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=28),
    CIFAR10Policy(), 
	transforms.ToTensor(), 
    Cutout(n_holes=1, length=28), # (https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py)
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


if config['dataset'] == 'cifar10':
####CIFAR10####
    target_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train_target)
    modelpool_trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train_relation)

    indices = torch.randperm(len(target_trainset))[:int(len(target_trainset)*config['data_percent'])]
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    target_trainloader = torch.utils.data.DataLoader(
        target_trainset, batch_size=config['batch_size']['train'], sampler=sampler, num_workers=4, pin_memory=True,drop_last=True)     
    modelpool_trainloader = torch.utils.data.DataLoader(
        modelpool_trainset, batch_size=config['batch_size']['train'], sampler=sampler, num_workers=4, pin_memory=True,drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size']['test'], shuffle=False, num_workers=4, pin_memory=True,drop_last=True)
####CIFAR10####
elif config['dataset'] == 'cifar100':
####CIFAR10####
    target_trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train_target)
    modelpool_trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train_relation)

    indices = torch.randperm(len(target_trainset))[:int(len(target_trainset)*config['data_percent'])]
    sampler = torch.utils.data.SubsetRandomSampler(indices)

    target_trainloader = torch.utils.data.DataLoader(
        target_trainset, batch_size=config['batch_size']['train'], sampler=sampler, num_workers=4, pin_memory=True,drop_last=True)     
    modelpool_trainloader = torch.utils.data.DataLoader(
        modelpool_trainset, batch_size=config['batch_size']['train'], sampler=sampler, num_workers=4, pin_memory=True,drop_last=True)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size']['test'], shuffle=False, num_workers=4, pin_memory=True,drop_last=True)
####CIFAR10####

elif config['dataset'] == 'cub':
####CUB####
    target_trainset = Cub2011(root='./data', train=True, transform=transform_train_relation, download=False)
    modelpool_trainset = Cub2011(root='./data', train=True, transform=transform_train_relation, download=False)
    testset = Cub2011(root='./data', train=False, transform=transform_test, download=False)
    target_trainloader = torch.utils.data.DataLoader(
            target_trainset, batch_size=config['batch_size']['train'],shuffle=True, num_workers=4, pin_memory=True,drop_last=False)     
    modelpool_trainloader = torch.utils.data.DataLoader(
        modelpool_trainset, batch_size=config['batch_size']['train'], shuffle=True,num_workers=4, pin_memory=True,drop_last=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size']['test'], shuffle=False, num_workers=4, pin_memory=True,drop_last=False)
####CUB####

elif config['dataset'] == 'flower':
    data_dir = './data/flower_data'
    data_transforms = {
    'train': transform_train_relation,
    'valid': transform_test,
    }
    data_transforms['train'] = transform_train_relation
    data_transforms['test'] = transform_test
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'valid']}
    target_trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batch_size']['train'],
                                             shuffle=True, num_workers=4, pin_memory=True,drop_last=False)
    modelpool_trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batch_size']['train'],
                                             shuffle=True, num_workers=4, pin_memory=True,drop_last=False)
    testloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=config['batch_size']['test'],
                                             shuffle=False, num_workers=4, pin_memory=True,drop_last=False)
else:
    raise ValueError("Not valid dataset, use -d cifar10 or -d pcam or -d neu") 

# Model
print('==> Building model..')
net = torchvision.models.resnext101_32x8d(pretrained=True)
net.fc = nn.Linear(2048, config['num_classes'])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    
if config['resume']['target_model'] != '':
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(config['resume']['target_model'])
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['ensemble_acc']
    start_epoch = checkpoint['epoch']
    print(start_epoch)
    print(best_acc)

    
for name, param in net.named_parameters():
    if name in ['module.fc.weight', 'module.fc.bias']:
        param.requires_grad = True
        print('done')
    else:
        param.requires_grad = False
    
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=config['lr'])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)


# Training
def train(epoch):
        
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(target_trainloader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(target_trainloader), 'Loss: %.3f | Train Acc: %.3f%% (%d/%d), epoch: %d'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch))
    scheduler.step()

def test(epoch, model_pool_outputs, target_ratio=0.5):
    
    global best_acc
    output_only_targetmodel = []
    output_with_modelpool = []
    groundtruth_label = []
    
    net.eval()
    test_loss = 0
    ensemble_correct = 0
    targetonly_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            
            inputs, targets = inputs.to(device), targets.to(device)
            groundtruth_label.extend(targets.detach().cpu().numpy())

            target_model_output = net(inputs)
            _, target_model_prediction = target_model_output.max(1)
            output_only_targetmodel.extend(target_model_prediction.detach().cpu().numpy())

            if model_pool_outputs is not None:
                model_pool_output = model_pool_outputs[batch_idx] #Data Not shuffled for Test dataset, so we can compare targetmodel output and modelpool output
                #normalize ouputs to [0, 1] before prediction ensemble
                target_model_output -= target_model_output.min(1, keepdim=True)[0]
                target_model_output /= target_model_output.max(1, keepdim=True)[0]
                model_pool_output -= model_pool_output.min(1, keepdim=True)[0]
                model_pool_output /= model_pool_output.max(1, keepdim=True)[0]

                ensemble_output = torch.sum(torch.stack([target_ratio * target_model_output, (1-target_ratio) * model_pool_output]), dim=0) #mean average sum of target output and modelpool output for fair compare
                _, ensemble_prediction = ensemble_output.max(1)
                output_with_modelpool.extend(ensemble_prediction.detach().cpu().numpy())
                
                loss = criterion(ensemble_output, targets)
                test_loss += loss.item()
                total += targets.size(0)
                ensemble_correct += ensemble_prediction.eq(targets).sum().item()
                targetonly_correct += target_model_prediction.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d), epoch:%d'
                % (test_loss/(batch_idx+1), 100.*ensemble_correct/total, ensemble_correct, total, epoch))
            else:
                loss = criterion(target_model_output, targets)
                test_loss += loss.item()
                total += targets.size(0)
                targetonly_correct += target_model_prediction.eq(targets).sum().item()
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d), epoch:%d'
                % (test_loss/(batch_idx+1), 100.*targetonly_correct/total, targetonly_correct, total, epoch))
           
    print(f'target ratio:{target_ratio:0.1f}')
    print(f'ensemble accuracy:{100.*ensemble_correct/total:0.3f}')
    print(f'targetonly accuracy:{100.*targetonly_correct/total:0.3f}')    
     # Save checkpoint.
    target_acc = 100.*targetonly_correct/total
    ensemble_acc = 100.*ensemble_correct/total
#     acc = 100.*targetonly_correct/total

    cm_tg = confusion_matrix(groundtruth_label, output_only_targetmodel, labels=[i for i in range(config['num_classes'])])
    cm_mp = confusion_matrix(groundtruth_label, output_with_modelpool, labels=[i for i in range(config['num_classes'])])
    cm_tg = cm_tg.astype('float') / cm_tg.sum(axis=1)[:, np.newaxis]
    cm_mp = cm_mp.astype('float') / cm_mp.sum(axis=1)[:, np.newaxis]
    cm_tg = cm_tg.diagonal()
    cm_mp = cm_mp.diagonal()

    tg_var =  np.var(cm_tg)
    mp_var = np.var(cm_mp)
    tg_macro_f1 = np.mean(cm_tg)
    mp_macro_f1 = np.mean(cm_mp)
    with open("./checkpoint/log.txt", 'a') as f:
        f.write("epoch: " + json.dumps(epoch)+"  ")
        f.write("targetonly_test_acc: " + json.dumps(target_acc))
        f.write("  ensemble_test_acc: " + json.dumps(ensemble_acc) + "\n")
        f.write("  tg_var: " + json.dumps(tg_var) + "\n")
        f.write("  mp_var: " + json.dumps(mp_var) + "\n")
        f.write("  tg_macro_f1: " + json.dumps(tg_macro_f1) + "\n")
        f.write("  mp_macro_f1: " + json.dumps(mp_macro_f1) + "\n")
    if ensemble_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'target_acc': target_acc,
            'ensemble_acc': ensemble_acc,
            'epoch': epoch,
            'output_only_targetmodel': output_only_targetmodel,
            'output_with_modelpool': output_with_modelpool,
            'groundtruth_label' : groundtruth_label,
            'tg_var' : tg_var,
            'mp_var' : mp_var,
            'tg_macro_f1' : tg_macro_f1,
            'mp_macro_f1' : mp_macro_f1
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/target_model.pth')
        best_acc = ensemble_acc
#         f.write("  best_acc: " + json.dumps(epoch) + "\n")

def predict_model_pool(stage='train'):
    
    loader = None
    if stage == 'train':
        loader = modelpool_trainloader
    elif stage == 'test':
        loader = testloader
    with torch.no_grad():
        model_pool_prediction = np.empty((0,0))
        stacked_target =  np.empty(0)
        output_sizes = []
        for batch_idx, (inputs, targets) in enumerate(loader):        
            stacked_target = np.concatenate((stacked_target, targets.cpu().detach().numpy()), axis=0) if stacked_target.size else targets.cpu().detach().numpy()
            batch_model_prediction = np.empty((0,0))
            for i_model in range(len(model_pool)):
                curr_model = model_pool[i_model].to(device)
                curr_model.eval()
                inputs = inputs.to(device)
                
                
                outputs = curr_model(inputs)
                if batch_idx == 0:
                    output_sizes.extend([outputs.shape[1]])

                batch_model_prediction = np.concatenate((batch_model_prediction, outputs.cpu().detach().numpy()), axis=1) if batch_model_prediction.size else outputs.cpu().detach().numpy() # N, D
            progress_bar(batch_idx, len(loader), ' ') 
            model_pool_prediction = np.concatenate((model_pool_prediction, batch_model_prediction), axis=0) if model_pool_prediction.size else batch_model_prediction

    return model_pool_prediction, stacked_target, output_sizes


def train_relation_net(num_epochs):

    model_pool_prediction, stacked_target, output_sizes = predict_model_pool(stage='train')
    num_models = len(output_sizes)
    input_size = [output_sizes[i] for i in range(len(output_sizes))] #differenent output sizes of model pool into encoder
    encode_size = config['encode_size']
    hidden_size = config['hidden_size']
    num_classes = config['num_classes']
    num_layers = config['num_layers']
    relation_net = lstm_relation_net(num_models, input_size, encode_size, hidden_size, num_classes, num_layers).to(device)
#     relation_net = tf_relation_net(num_models, input_size, encode_size, hidden_size, num_classes, num_layers, num_heads).to(device)
    if device == 'cuda':
        relation_net = torch.nn.DataParallel(relation_net)
    if config['resume']['relation_net'] != '':
        relation_net.load_state_dict(torch.load(config['resume']['relation_net']))


    criterion = nn.CrossEntropyLoss()
    relation_optimizer = torch.optim.AdamW(relation_net.parameters(), lr=5e-5, weight_decay=1e-2)
    relation_scheduler = optim.lr_scheduler.ExponentialLR(relation_optimizer, gamma=0.998)
    
    model_checkpoints_folder = os.path.abspath('./checkpoint')
    save_config_file(model_checkpoints_folder, config['project_name'])
    f = open("./checkpoint/log.txt", "w")

    for epoch in range(num_epochs):
        if epoch != 0: #Use pre-built predictions for first epoch
            model_pool_prediction, stacked_target, output_sizes = predict_model_pool(stage='train') # we need modelpool prediction for every epoch, to apply data augmentation
        
        
        relation_set = RelationData(model_pool_prediction, stacked_target)
        relation_loader = torch.utils.data.DataLoader(relation_set, batch_size=config['batch_size']['train'], shuffle=True, num_workers=4)
        
        train_loss = 0
        correct = 0
        total = 0
        
        relation_net.train()
        for batch_idx, (inputs, targets) in enumerate(relation_loader):
            relation_optimizer.zero_grad()

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = relation_net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            relation_optimizer.step()            
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(relation_loader), 'Loss: %.3f | Relation Train Acc: %.3f%% (%d/%d), epoch: %d'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch))
        relation_scheduler.step()
        with open("./checkpoint/log.txt", 'a') as f:
            f.write("epoch: " + json.dumps(epoch)+"  ")
            f.write("relationnet_train_acc: " + json.dumps(100.*correct/total) + "\n")
    torch.save(relation_net.state_dict(), f'./checkpoint/relation_net.pth')
#     test_relation_net(relation_net)
    return relation_net

def test_relation_net(relation_net):

    model_pool_prediction, stacked_target, output_sizes = predict_model_pool(stage='test')
    num_models = len(output_sizes)
    input_size = [output_sizes[i] for i in range(len(output_sizes))]
    relation_net.eval()

    relation_set = RelationData(model_pool_prediction, stacked_target)
    relation_loader = torch.utils.data.DataLoader(relation_set, batch_size=config['batch_size']['test'], shuffle=False, num_workers=4)
    
    stacked_output = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(relation_loader):
            inputs, targets = inputs.to(device), targets.to(device)  
            cls_output = relation_net(inputs)
#             outputs = (cls_output, dist_output)
            _, predicted = cls_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(relation_loader), 'Relation Test Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))
            stacked_output.append(cls_output)
    with open("./checkpoint/log.txt", 'a') as f:
#         f.write("epoch: " + json.dumps(epoch)+"  ")
        f.write("relationnet_test_acc: " + json.dumps(100.*correct/total) + "\n")
    return stacked_output # Stored Relationnet'soutputs will be averaged with target model's outputs

if __name__ =='__main__':  
    if len(model_pool) == 0: #train target model only
        for epoch in range(start_epoch, start_epoch+config['num_epochs']['target_model']):
            train(epoch)
            test(epoch, None, 1)
    
    else:
        trained_relation_net = train_relation_net(num_epochs=config['num_epochs']['relation_net'])
        relation_net_prediction = test_relation_net(trained_relation_net) # To use trained relationnet output for ensemble.call only once
        for epoch in range(start_epoch, start_epoch+config['num_epochs']['target_model']):
            train(epoch)
            test(epoch, relation_net_prediction, target_ratio=config['target_ratio'])
#             test(epoch, None, 1)                               
        model_checkpoints_folder = os.path.abspath('./checkpoint')
        copy_to_experiment_dir(model_checkpoints_folder, config['project_name'])