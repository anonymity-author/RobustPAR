import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import log_untils
from tools.utils import AverageMeter, to_scalar, time_str
def batch_trainer(epoch, model,clip_model, train_loader, criterion, optimizer,optimizer1,prompt_optimizer,args):
    model.train()
    clip_model.train()
    epoch_time = time.time()
    loss_meter = AverageMeter()
    batch_num = len(train_loader)
    gt_list = []
    preds_probs = []

    prompt_lr = prompt_optimizer.param_groups[0]['lr']
    lr = optimizer.param_groups[0]['lr']
    lr1=optimizer1.param_groups[0]['lr']


    metric_logger = log_untils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    for step, (imgs, gt_label, imgname) in enumerate(metric_logger.log_every(train_loader, int(batch_num/2), header)):    
        batch_time = time.time()
        imgs, gt_label = imgs.cuda(), gt_label.cuda()
        train_logits,final_similarity,bn_logits2,gt_label1,gt_label2= model(imgs,clip_model=clip_model,gt_label=gt_label)
        if args.use_GL :
            classifier_loss = criterion(train_logits, gt_label1)
            clip_loss = criterion(final_similarity, gt_label)
            loss_2 = criterion(bn_logits2, gt_label2)
            train_loss =  classifier_loss + 0.5 * clip_loss  +loss_2
        else :
            train_loss = criterion(train_logits, gt_label)
        optimizer.zero_grad()
        optimizer1.zero_grad()
        prompt_optimizer.zero_grad()

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(clip_model.parameters(), 10.0)

        optimizer.step()
        prompt_optimizer.step()
        optimizer1.step()
        loss_meter.update(to_scalar(train_loss))

        gt_list.append(gt_label.cpu().numpy())
        train_probs = torch.sigmoid(train_logits)
        preds_probs.append(train_probs.detach().cpu().numpy())

        metric_logger.update(train_loss=train_loss.item())
        if args.use_GL :
            metric_logger.update(classifier_loss=classifier_loss.item())
            metric_logger.update(clip_loss=clip_loss.item())

        metric_logger.update(prompt_lr=prompt_lr)
        metric_logger.update(VTB_lr=lr)


    train_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return train_loss, gt_label, preds_probs


def valid_trainer(model,clip_model, valid_loader, criterion,args):
    model.eval()
    loss_meter = AverageMeter()
    preds_probs = []
    gt_list = []
    with torch.no_grad():
        for step, (imgs, gt_label, imgname) in enumerate(valid_loader):
            imgs = imgs.cuda()
            gt_label = gt_label.cuda() #
            gt_list.append(gt_label.cpu().numpy())
            gt_label[gt_label == -1] = 0
            valid_logits,final_similarity,bn_logits2,_,_ = model(imgs,clip_model=clip_model,gt_label=gt_label,mode='test')
            if args.use_div:
                classifier_loss = criterion(valid_logits, gt_label)
                clip_loss = criterion(final_similarity, gt_label)
                loss_2=criterion(bn_logits2,gt_label)
                valid_loss = classifier_loss + 0.5 * clip_loss+loss_2
            else :
                valid_loss = criterion(valid_logits, gt_label)
            valid_probs = torch.sigmoid(valid_logits)
            loss_2_probs = torch.sigmoid(bn_logits2)
            probs = torch.max(valid_probs,loss_2_probs)
            preds_probs.append(probs.cpu().numpy())
            loss_meter.update(to_scalar(valid_loss))

    valid_loss = loss_meter.avg

    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    return valid_loss, gt_label, preds_probs