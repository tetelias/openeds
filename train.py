import argparse
import json
import os
import random
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from albumentations import (CLAHE, CenterCrop, Compose, GaussianBlur,
                        GaussNoise, HorizontalFlip, IAASharpen, Normalize,
                        RandomBrightnessContrast, RandomCrop, RandomGamma,
                        ShiftScaleRotate)
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader


from model.segnet import SegNet
from progress.bar import Bar
from utils.dataset import EdsDS
from utils.logs import AverageMeter, Logger
from utils.losses import DiceFocalWithLogitsLoss
from utils.metric import general_dice
from utils.optim import AdamW
from utils.scheduler import LR_Scheduler, SchedulerConst, SchedulerCosine
from utils.other_utils import get_lr, get_momentum, np_to_base64_utf8_str, set_momentum

finish_lr = 1e-5
lr_mult = np.array([1])
min_mom = 0.85 
max_mom = 0.95  
mom_name = 'betas'
NUM_MODEL_PARAMS = 242664
start_lr = 1e-6
train_sampler = None
warmup_part = 0.25
wd = 1e-4
workers = 4        

def main(config):

    if config.channels == 1:
        mean = [0.467]
        std = [0.271]
    elif config.channels == 3:
        mean = [0.467, 0.467, 0.467]
        std = [0.271, 0.271, 0.271]

    if config.device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{:d}'.format(config.device))
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    train_tfms = Compose([
        ShiftScaleRotate(rotate_limit=15, interpolation=cv2.INTER_CUBIC),
        GaussianBlur(), 
        GaussNoise(),
        HorizontalFlip(),
        RandomBrightnessContrast(),
        Normalize(
            mean=mean,
            std=std,
        ),
        ToTensor()
    ])

    val_tfms = Compose([
        Normalize(
            mean=mean,
            std=std,
        ),
        ToTensor()
    ])   

    SAVEPATH = Path(config.root_dir)
    #Depending on the stage we either create train/validation or test dataset
    if config.stage == 'train':
        train_ds = EdsDS(fldr=SAVEPATH/config.train_dir, channels=config.channels, transform=train_tfms)
        val_ds = EdsDS(fldr=SAVEPATH/config.valid_dir, channels=config.channels, transform=val_tfms)       

        train_loader = DataLoader(
            train_ds, batch_size=config.bs, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=True, sampler=train_sampler)        
        
        checkpoint = 'logger'
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        arch = 'segnet_'
        title = 'Eye_' + arch + 'fast_fd_g{}_'.format(config.gamma)        

        logger = Logger(os.path.join(checkpoint, '{}e{:d}_lr{:.4f}.txt'.format(title, config.ep, config.lr)), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Dice']) 
    elif config.stage == 'test':     
        val_ds = EdsDS(fldr=SAVEPATH/config.test_dir, channels=config.channels, mask=False, transform=val_tfms)

    val_loader = DataLoader(
        val_ds, batch_size=config.bs*2, shuffle=False,
        num_workers=workers, pin_memory=True)

    model = SegNet(channels=config.channels).to(device)

    criterion = DiceFocalWithLogitsLoss(gamma=config.gamma).to(device)

    optimizer = AdamW(model.parameters(), lr=start_lr, betas=(max_mom, 0.999), weight_decay=wd)

    if config.stage == 'train':    
        steps = len(train_loader) * config.ep
        
        schs = []
        schs.append(SchedulerCosine(optimizer, start_lr, config.lr, lr_mult, int(steps * warmup_part), max_mom, min_mom))
        schs.append(SchedulerCosine(optimizer, config.lr, finish_lr, lr_mult, steps - int(steps * warmup_part), min_mom, max_mom))
        lr_scheduler = LR_Scheduler(schs)  
        
        max_dice = 0
        
        for epoch in range(config.ep):                        

            print('\nEpoch: [{:d} | {:d}] LR: {:.10f}|{:.10f}'.format(epoch+1, config.ep, get_lr(optimizer,-1), get_lr(optimizer,0))) 

            # train for one epoch
            train_loss = train(train_loader, model, criterion, optimizer, lr_scheduler, device, config)

            # evaluate on validation set
            valid_loss, dice = validate(val_loader, model, criterion, device, config)
            
            # append logger file
            logger.append([get_lr(optimizer,-1), train_loss, valid_loss, dice])            

            if dice > max_dice:
                max_dice = dice
                model_state = {
                    'epoch': epoch+1,
                    'arch': arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }
                save_model = '{}e{:d}_lr_{:.3f}_max_dice.pth.tar'.format(title, config.ep, config.lr)
                torch.save(model_state, save_model)    
    elif config.stage == 'test':
        checkpoint = torch.load(config.saved_model)
        model.load_state_dict(checkpoint['state_dict'])
        logits = validate(val_loader, model, criterion, device, config)
        preds = np.concatenate([torch.argmax(l, 1).numpy() for l in logits]).astype(np.uint8)
        leng = len(preds)
        data = {}        
        data['num_model_params'] = NUM_MODEL_PARAMS
        data['number_of_samples'] = leng
        data['labels'] = {}
        for i in range(leng):
            data['labels'][val_ds.img_paths[i].stem] = np_to_base64_utf8_str(preds[i])
        with open(SAVEPATH/'{}.json'.format(config.filename), 'w') as f:
            json.dump(data,f)   
        with zipfile.ZipFile(SAVEPATH/'{}.zip'.format(config.filename), "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(SAVEPATH/'{}.json'.format(config.filename))    
        os.remove(SAVEPATH/'{}.json'.format(config.filename))                 


def train(train_loader, model, criterion, optimizer, scheduler, device, config):

    # switch to train mode
    model.train()  

    losses = AverageMeter()
    dice = AverageMeter()

    bar = Bar('Processing', max=len(train_loader))

    for step, (inputs, targets) in enumerate(train_loader):

        images = inputs.cuda(device, non_blocking=True)
        targets = targets.cuda(device, non_blocking=True)
    
        outputs = model(images).squeeze(1)

        loss = criterion(outputs, targets)          

        # measure roc auc and record loss
        losses.update(loss.item(), images.size(0))
        pred_probs = torch.sigmoid(outputs)
        dices = general_dice(targets.detach().cpu().numpy(), torch.argmax(pred_probs,1).detach().cpu().numpy())
        dice.update(dices, images.size(0)) 

        optimizer.zero_grad()

        loss.backward()            

        # compute gradient and do SGD step
        optimizer.step()
        scheduler.step()

        # plot progress
        bar.suffix  = '({batch}/{size})| LR: {lr:} | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Dice: {mean_dice: .4f}'.format(
                    batch=step + 1,
                    size=len(train_loader),
                    lr = get_lr(optimizer,-1),
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    mean_dice=dice.avg
                    )
        bar.next()
    bar.finish()
    return losses.avg


def validate(val_loader, model, criterion, device, config):
    """Calculate loss and top-1 classification accuracy one the validation set."""

    if config.stage == 'train':
        losses = AverageMeter()
        dice = AverageMeter()
        
    # switch to evaluate mode
    model.eval()    

    with torch.no_grad():

        if config.stage == 'train':
            bar = Bar('Processing', max=len(val_loader))
        else:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)            
        
        outs = []

        for step, (inputs, targets) in enumerate(val_loader):

            images = inputs.to(device)
            targets = targets.to(device)
            
            if len(targets.shape) > 1: targets = targets.squeeze()
            outputs = model(images).squeeze(1)

            if config.stage == 'test':
                outs.append(torch.sigmoid(model(images).squeeze(1).detach().cpu()))
                continue
            
            # calculate loss
            loss = criterion(outputs, targets)
            
            # measure roc auc and record loss
            losses.update(loss.item(), images.size(0))
            pred_probs = torch.sigmoid(outputs)
            dices = general_dice(targets.detach().cpu().numpy(), torch.argmax(pred_probs,1).detach().cpu().numpy())       
            dice.update(dices, images.size(0))

            # plot progress
            bar.suffix  = '({batch}/{size}) | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | ACC: {mean_dice: .4f}'.format(
                        batch=step + 1,
                        size=len(val_loader),
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        mean_dice=dice.avg
                        )
            bar.next()

    if config.stage == 'train':
        bar.finish()
        return losses.avg, dice.avg
    else:
        return outs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--ep', type=int, help='Number of epochs', default=40)  
    parser.add_argument('--bs', type=int, help='Batch size for training; testing uses twice as big', default=64) 
    parser.add_argument('--channels', type=int, help='Number of input channels: images are grayscale, so default is 1', default=1)
    parser.add_argument('--device', type=int, help='Choose either "-1" for using CPU or ordinal number for GPU', default="0")
    parser.add_argument('--gamma', type=int, help='Parameter for Focal loss', default=1)
    parser.add_argument('--filename', type=str, help='Name without extension for zip file to save submission in', default="submission")
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.01)
    parser.add_argument('--output_dir', type=str, help='Path to folder relative to root_dir to put results in', default="output/")
    parser.add_argument('--root_dir', type=str, help='Input file name', default="data/")    
    parser.add_argument('--saved_model', type=str, help='Path to saved model to use for inference', default=None)
    parser.add_argument('--stage', type=str, help='Choose either "train" or "test"', default="test")
    parser.add_argument('--test_dir', type=str, help='Path to folder relative to root_dir with training samples', default="test/")
    parser.add_argument('--train_dir', type=str, help='Path to folder relative to root_dir with training samples', default="train/")
    parser.add_argument('--valid_dir', type=str, help='Path to folder relative to root_dir with validation samples', default="validation/")
    
    config = parser.parse_args()
    main(config)