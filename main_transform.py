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

from losses import DistillationLoss
import yaml
import json

SEED_NUM = 1995
torch.manual_seed(SEED_NUM)
np.random.seed(SEED_NUM)
random.seed(SEED_NUM)
torch.cuda.manual_seed(SEED_NUM)
torch.cuda.manual_seed_all(SEED_NUM) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
# print(config['model_pool']['cifar10_pretrains'])
# with open('./config.yaml', 'w') as outfile:
#     yaml.dump(config, outfile)
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--num_epochs', '-nepochs', default=200, type=int)
# parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
# parser.add_argument('--resume', '-r', default=None, type=str,
#                     help='resume from checkpoint')
# parser.add_argument('--dataset', '-d', default='cifar10', type=str)
# parser.add_argument('--num_classes', '-n', default=10, type=int)
# parser.add_argument('--batch_size', '-b', default=256, type=int)
# parser.add_argument('--encode_size', '-edim', default=192, type=int)
# parser.add_argument('--hidden_size', '-hdim', default=768, type=int)
# parser.add_argument('--num_layers', '-nlayers', default=12, type=int)
# parser.add_argument('--num_heads', '-nheads', default=3, type=int)
# parser.add_argument('--data_percent', '-p', default=0.05, type=float)
# parser.add_argument('--other_pretrains', '-op', default=False, type=bool)
# parser.add_argument('--cifar10_pretrains', '-cp', default=False, type=bool)
# parser.add_argument('--alpha', default=0.5, type=float)


# args = parser.parse_args()
# print(args)

###cifar10 dataset trained models###



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
# model_pool.append(pool_models.squeezenet1_0(pretrained=True))
    model_pool.append(pool_models.vgg16_bn(pretrained=True))
    model_pool.append(pool_models.densenet161(pretrained=True))
    model_pool.append(pool_models.googlenet(pretrained=True))
# model_pool.append(pool_models.shufflenet_v2_x1_0(pretrained=True))
    model_pool.append(pool_models.mobilenet_v2(pretrained=True))
    model_pool.append(pool_models.inception_v3(pretrained=True))
# # model_pool.append(pool_models.mobilenet_v3_large(pretrained=True))
# model_pool.append(pool_models.mobilenet_v3_small(pretrained=True))
# # model_pool.append(pool_models.resnext50_32x4d(pretrained=True))
# # model_pool.append(pool_models.wide_resnet50_2(pretrained=True))
# # model_pool.append(pool_models.mnasnet1_0(pretrained=True))

# with open('config.yaml', 'w') as outfile:
#     yaml.dump(model_pool, outfile)

# print(pool_models.inception_v3)
# with open('./config.yaml', 'w') as f:
#     yaml.dump(model_pool, f)
print(f'modelpool num_models:{len(model_pool)}')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
relationnet_best_acc=0

start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_train_target = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_relation = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomCrop(224, padding=32),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
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
        target_trainset, batch_size=64, sampler=sampler, num_workers=4, drop_last=True)     
    modelpool_trainloader = torch.utils.data.DataLoader(
        modelpool_trainset, batch_size=config['batch_size'], sampler=sampler, num_workers=4, drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size'], shuffle=False, num_workers=4, drop_last=True)
####CIFAR10####

elif config['dataset'] == 'pcam':
####PCAM####
    labels = pd.read_csv('./data/PCAM/train_labels.csv') # No label for test data! We split train data to train, test.
    train_idx, test_idx = train_test_split(labels.label, random_state=SEED_NUM, test_size=(1-config['data_percent']))
    train_sampler = SubsetRandomSampler(list(train_idx.index))
    test_sampler = SequentialSampler(list(test_idx.index)[:int(0.01 * len(test_idx.index))]) # Use smaller amount for testset
    img_class_dict = {k:v for k, v in zip(labels.id, labels.label)}
    target_trainset = PCAMDataset(datafolder='./data/PCAM/train/', transform=transform_train_target, labels_dict=img_class_dict)
    modelpool_trainset = PCAMDataset(datafolder='./data/PCAM/train/', transform=transform_train_relation, labels_dict=img_class_dict)
    testset = PCAMDataset(datafolder='./data/PCAM/train/', transform=transform_test, labels_dict=img_class_dict)
    target_trainloader = torch.utils.data.DataLoader(
            target_trainset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4, drop_last=True)     
    modelpool_trainloader = torch.utils.data.DataLoader(
        modelpool_trainset, batch_size=config['batch_size'], sampler=train_sampler, num_workers=4, drop_last=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=config['batch_size'], sampler=test_sampler, num_workers=4, drop_last=True)
####PCAM####

elif config['dataset'] == 'neu':
####NEU####
    target_trainset, modelpool_trainset, testset, target_trainloader, modelpool_trainloader, testloader = NEUdataset('./data/NEU', batch_size=config['batch_size'])
####NEU####
else:
    raise ValueError("Not valid dataset, use -d cifar10 or -d pcam or -d neu") 

# Model
print('==> Building model..')
net = torchvision.models.resnext101_32x8d(num_classes=config['num_classes'])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    
if config['resume']['target_model']:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(config['resume']['target_model'])
    # checkpoint = torch.load(f'./checkpoint/ckpt_{args.dataset}_{args.encode_size}_{args.hidden_size}_{args.num_layers}_other_pretrains_{args.other_pretrains}_cifar10_pretrains_{args.cifar10_pretrains}_{args.data_percent}.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(start_epoch)
    print(best_acc)

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

def test(epoch):
    
    global net, best_acc
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
            # output_only_targetmodel.extend(target_model_prediction.detach().cpu().numpy())

#             if model_pool_outputs is not None:
#                 model_pool_output = model_pool_outputs[batch_idx] #Data Not shuffled for Test dataset, so we can compare targetmodel output and modelpool output
#                 #normalize ouputs to [0, 1] before prediction ensemble
#                 target_model_output -= target_model_output.min(1, keepdim=True)[0]
#                 target_model_output /= target_model_output.max(1, keepdim=True)[0]
#                 model_pool_output -= model_pool_output.min(1, keepdim=True)[0]
#                 model_pool_output /= model_pool_output.max(1, keepdim=True)[0]

#                 ensemble_output = torch.sum(torch.stack([target_ratio * target_model_output, (1-target_ratio) * model_pool_output]), dim=0) #mean average sum of target output and modelpool output for fair compare
#                 _, ensemble_prediction = ensemble_output.max(1)
#                 output_with_modelpool.extend(ensemble_prediction.detach().cpu().numpy())
                
#                 loss = criterion(ensemble_output, targets)
#                 test_loss += loss.item()
#                 total += targets.size(0)
#                 ensemble_correct += ensemble_prediction.eq(targets).sum().item()
#                 targetonly_correct += target_model_prediction.eq(targets).sum().item()
            loss = criterion(target_model_output, targets)
            test_loss += loss.item()
            total += targets.size(0)
#             ensemble_correct += ensemble_prediction.eq(targets).sum().item()
            targetonly_correct += target_model_prediction.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d), epoch:%d'
            % (test_loss/(batch_idx+1), 100.*targetonly_correct/total, targetonly_correct, total, epoch))
#     print(f'target ratio:{target_ratio:0.1f}')
#     print(f'ensemble accuracy:{100.*ensemble_correct/total:0.3f}')
#     print(f'targetonly accuracy:{100.*targetonly_correct/total:0.3f}')    
     # Save checkpoint.
    acc = 100.*targetonly_correct/total
    print(acc)
    if epoch > 20 and acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'output_only_targetmodel': output_only_targetmodel,
#             'output_with_modelpool': output_with_modelpool,
            'groundtruth_label' : groundtruth_label
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, f'./weights/pretrained/target_ckpt.pth')
        best_acc = acc
        
def predict_model_pool(stage='train'):
    
    loader = None
    if stage == 'train':
        loader = modelpool_trainloader
    elif stage == 'test':
        loader = testloader
    with torch.no_grad():
        model_pool_prediction = np.empty((0,0))
        teacher_prediction = np.empty((0, 0))
        stacked_target =  np.empty(0)
        output_sizes = []
        for batch_idx, (inputs, targets) in enumerate(loader):        
            stacked_target = np.concatenate((stacked_target, targets.cpu().detach().numpy()), axis=0) if stacked_target.size else targets.cpu().detach().numpy()
            batch_model_prediction = np.empty((0,0))
            batch_teacher_prediction = np.empty((0,0))

            for i_model in range(len(model_pool)):
                curr_model = model_pool[i_model].to(device)
                curr_model.eval()
                inputs = inputs.to(device)
                outputs = curr_model(inputs)
                target_outputs = net(inputs)
                if batch_idx == 0:
                    output_sizes.extend([outputs.shape[1]])
                batch_model_prediction = np.concatenate((batch_model_prediction, outputs.cpu().detach().numpy()), axis=1) if batch_model_prediction.size else outputs.cpu().detach().numpy() # N, D
            batch_teacher_prediction = np.concatenate((batch_teacher_prediction, target_outputs.cpu().detach().numpy()), axis=1) if batch_teacher_prediction.size else target_outputs.cpu().detach().numpy() # N, D
            progress_bar(batch_idx, len(loader), ' ') 
            model_pool_prediction = np.concatenate((model_pool_prediction, batch_model_prediction), axis=0) if model_pool_prediction.size else batch_model_prediction
            teacher_prediction = np.concatenate((teacher_prediction, batch_teacher_prediction), axis=0) if teacher_prediction.size else batch_teacher_prediction
    return model_pool_prediction, stacked_target, output_sizes, teacher_prediction


def train_relation_net(num_epochs, trained_target_model):

    model_pool_prediction, stacked_target, output_sizes, teacher_prediction = predict_model_pool(stage='train')
    num_models = len(output_sizes)
    input_size = [output_sizes[i] for i in range(len(output_sizes))] #differenent output sizes of model pool into encoder
    encode_size = config['encode_size']
    hidden_size = config['hidden_size']
    num_classes = config['num_classes']
    num_layers = config['num_layers']
    num_heads = config['num_heads']
#     relation_net = lstm_relation_net(num_models, input_size, encode_size, hidden_size, num_classes, num_layers).to(device)
    relation_net = tf_relation_net(num_models, input_size, encode_size, hidden_size, num_classes, num_layers, num_heads).to(device)
    
    if config['resume']['relation_net'] != 'None':
        relation_net.load_state_dict(torch.load(config['resume']['relation_net']))

    base_criterion = nn.CrossEntropyLoss()
    criterion = DistillationLoss(base_criterion, teacher_model=trained_target_model,
                 distillation_type=config['distillation_type'], alpha=config['alpha'], tau=config['tau'])
    relation_optimizer = torch.optim.AdamW(relation_net.parameters(), lr=5e-5, weight_decay=1e-2)
    relation_scheduler = optim.lr_scheduler.ExponentialLR(relation_optimizer, gamma=0.998)
    
    model_checkpoints_folder = os.path.abspath('./checkpoint')
    save_config_file(model_checkpoints_folder, config['model_name'])
    f = open("./checkpoint/log.txt", "w")

    for epoch in range(num_epochs):
        if epoch != 0: #Use pre-built predictions for first epoch
            model_pool_prediction, stacked_target, output_sizes, teacher_prediction = predict_model_pool(stage='train') # we need modelpool prediction for every epoch, to apply data augmentation
        
        
        relation_set = RelationData(model_pool_prediction, stacked_target, teacher_prediction)
        relation_loader = torch.utils.data.DataLoader(relation_set, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        
        train_loss = 0
        correct = 0
        total = 0
        
        relation_net.train()
        for batch_idx, (inputs, targets, teacher_outputs) in enumerate(relation_loader):
            relation_optimizer.zero_grad()

            inputs, targets, teacher_outputs = inputs.to(device), targets.to(device), teacher_outputs.to(device)     
#             outputs = relation_net(inputs)
#             loss = criterion(outputs, targets)
            cls_output, dist_output = relation_net(inputs)
            outputs = (cls_output, dist_output)
#             teacher_outputs = net(inputs)
            loss = criterion(inputs, outputs, targets, teacher_outputs)
            loss.backward()
            relation_optimizer.step()            
            
            train_loss += loss.item()
            _, predicted = cls_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(relation_loader), 'Loss: %.3f | Relation Train Acc: %.3f%% (%d/%d), epoch: %d'
                         % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, epoch))
        relation_scheduler.step()
        with open("./checkpoint/log.txt", 'a') as f:
            f.write("epoch: " + json.dumps(epoch)+"  ")
            f.write("train_acc: " + json.dumps(100.*correct/total) + "\n")
    test_relation_net(relation_net)

def test_relation_net(relation_net):
    global best_acc
    global relationnet_best_acc

    model_pool_prediction, stacked_target, output_sizes, teacher_prediction = predict_model_pool(stage='test')
    num_models = len(output_sizes)
    input_size = [output_sizes[i] for i in range(len(output_sizes))]
    relation_net.eval()

    relation_set = RelationData(model_pool_prediction, stacked_target, teacher_prediction)
    relation_loader = torch.utils.data.DataLoader(relation_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    stacked_output = []
    with torch.no_grad():
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, teacher_outputs) in enumerate(relation_loader):
            inputs, targets, teacher_outputs = inputs.to(device), targets.to(device), teacher_outputs.to(device)     
            cls_output, dist_output = relation_net(inputs)
#             outputs = (cls_output, dist_output)
            _, predicted = cls_output.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(relation_loader), 'Relation Test Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > relationnet_best_acc:
        relationnet_best_acc = acc
        print("saving")
        torch.save(relation_net.state_dict(), f'./checkpoint/relation_ckpt.pth')
    with open("./checkpoint/log.txt", 'a') as f:
        f.write("test_acc: " + json.dumps(acc) + "\n")
if __name__ =='__main__':  
    if len(model_pool) == 0: #train target model only
        for epoch in range(start_epoch, start_epoch+num_epochs):
            train(epoch)
            test(epoch)
    
    else:
#         for epoch in range(start_epoch, start_epoch+1):
#             train(epoch)
#             test(epoch)
    
        trained_target_model = net
        trained_relation_net = train_relation_net(num_epochs=config['num_epochs'], trained_target_model=trained_target_model)
        model_checkpoints_folder = os.path.abspath('./checkpoint')
        copy_to_experiment_dir(model_checkpoints_folder, config['model_name'])
#         relation_net_prediction = test_relation_net() # To use trained relationnet output for ensemble.call only once
#         for epoch in range(start_epoch, start_epoch+150):
#             train(epoch)
#             test(epoch, relation_net_prediction, target_ratio=0.5)
                                       
    # output_only_targetmodel = checkpoint['output_only_targetmodel']
    # output_with_modelpool = checkpoint['output_with_modelpool']
    # groundtruth_label = checkpoint['groundtruth_label']
    # cf_matrix = confusion_matrix(output_only_targetmodel, groundtruth_label)
    # make_confusion_matrix(cf_matrix, figsize=(8,10), cbar=False, title='only target model')
    # cf_matrix = confusion_matrix(output_with_modelpool, groundtruth_label)
    # make_confusion_matrix(cf_matrix, figsize=(8,10), cbar=False, title='with model pool')