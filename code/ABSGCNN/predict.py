import argparse
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.utils.data import DataLoader

from cgcnn.data import StructureData
from cgcnn.data import collate_pool
from cgcnn.model import CrystalGraphConvNet

from sgcnn.SIdata import SIStructureData
from sgcnn.SIdata import SIcollate_pool
from sgcnn.SImodel import SICrystalGraphConvNet

from dgcnn.DSIdata import DSIStructureData
from dgcnn.DSIdata import DSIcollate_pool
from dgcnn.DSImodel import DSICrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('modelpath', help='path to the trained model.')
parser.add_argument('cifpath', help='path to the directory of CIF files.')
parser.add_argument('--structure', '-sc', choices=['vasp', 'cif'],
                    default='vasp', help='load structurs by using *.vasp or '
                                                   '*.cif (default: vasp)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')#action为开关
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

args = parser.parse_args(sys.argv[1:])#解析参数
if os.path.isfile(args.modelpath):
    print("=> loading model params '{}'".format(args.modelpath))
    model_checkpoint = torch.load(args.modelpath,
                                  map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint['args'])
    print("=> loaded model params '{}'".format(args.modelpath))
else:
    print("=> no model params found at '{}'".format(args.modelpath))
args.cuda = not args.disable_cuda and torch.cuda.is_available() #有一个False就Fasle

if model_args.task == 'regression':
    best_rmse_error = 1e10
else:
    best_rmse_error = 0.

def main():
    global args, model_args, best_rmse_error
    # load data
    if model_args.dp.split('-')[0] == 'cg':
        dataset = StructureData(args.cifpath, args.structure, cut_rds=model_args.cut_rds)     
        collate_fn = collate_pool
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)
    elif model_args.dp.split('-')[0] == 'sg':  
        dataset = SIStructureData(args.cifpath, args.structure, model_args.dp.split('-')[1:], cut_rds=model_args.cut_rds, nbr_type=model_args.nbr_type)
        collate_fn = SIcollate_pool  
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)
    elif model_args.dp.split('-')[0] == 'dg':  
        dataset = DSIStructureData(args.cifpath, args.structure, model_args.dp.split('-')[1:], cut_rds=model_args.cut_rds, nbr_type=model_args.nbr_type)
        collate_fn = DSIcollate_pool  
        test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, collate_fn=collate_fn,
                             pin_memory=args.cuda)
   
    # build model
    if model_args.dp.split('-')[0] == 'cg':
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                                           'classification' else False)
    elif model_args.dp.split('-')[0] == 'sg': 
        structures_i, structures_s, _, _ = dataset[0]
        orig_atom_fea_len = structures_i[0].shape[-1]
        nbr_fea_len = [structures_i[1].shape[-1], structures_s[1].shape[-1]]
        model = SICrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                                           'classification' else False)
    elif model_args.dp.split('-')[0] == 'dg': 
        structures_d, structures_i, structures_s, _, _ = dataset[0]
        orig_atom_fea_len = structures_d[0].shape[-1]
        nbr_fea_len = [structures_d[2].shape[-1], structures_i[1].shape[-1], structures_s[1].shape[-1]]
        model = DSICrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=model_args.atom_fea_len,
                                    n_conv=model_args.n_conv,
                                    h_fea_len=model_args.h_fea_len,
                                    n_h=model_args.n_h,
                                    classification=True if model_args.task ==
                                                           'classification' else False)           
#     print(model)  
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if model_args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    
    normalizer = Normalizer(torch.zeros(3))
    
    # optionally resume from a checkpoint
    if os.path.isfile(args.modelpath):
        print("=> loading model '{}'".format(args.modelpath))
        checkpoint = torch.load(args.modelpath,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        normalizer.load_state_dict(checkpoint['normalizer'])
        print("=> loaded model '{}' (epoch {}, validation {})"
              .format(args.modelpath, checkpoint['epoch'],
                      checkpoint['best_cnvtol_error']))
    else:
        print("=> no model found at '{}'".format(args.modelpath))
    
    validate(test_loader, model, criterion, normalizer, test=True)
    
def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == 'regression':
        rmse_errors = AverageMeter()
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []
        test_probs = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    if model_args.dp.split('-')[0] == 'cg':
        for i, (input, target, batch_cif_ids) in enumerate(val_loader):
            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(input[0].cuda(non_blocking=True)),
                                 Variable(input[1].cuda(non_blocking=True)),
                                 input[2].cuda(non_blocking=True),
                                 [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                with torch.no_grad():
                    input_var = (Variable(input[0]),
                                 Variable(input[1]),
                                 input[2],
                                 input[3])
            if model_args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            with torch.no_grad():
                if args.cuda:
                    target_var = Variable(target_normed.cuda(non_blocking=True))
                else:
                    target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            if model_args.task == 'regression':
                rmse_error = rmse(normalizer.denorm(output.data.cpu()), target)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                rmse_errors.update(rmse_error, target.size(0))
                mae_errors.update(mae_error, target.size(0))
                if test:
                    test_pred = normalizer.denorm(output.data.cpu())
                    test_target = target
                    test_preds += test_pred.view(-1).tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.argmax(torch.exp(output.data.cpu()), axis=1)
                    test_target = target
                    test_prob = np.exp(output.data.cpu().numpy())[:, 1]
                    test_preds += test_pred.tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
                    test_probs += test_prob.tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if model_args.task == 'regression':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                          'RMSE {rmse_errors.val:.3f} ({rmse_errors.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, mae_errors=mae_errors, 
                        rmse_errors=rmse_errors))
                    temp = [float(losses.avg), float(mae_errors.avg), float(rmse_errors.avg)]
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        accu=accuracies, prec=precisions, recall=recalls,
                        f1=fscores, auc=auc_scores))
                    temp = [float(losses.avg), float(precisions.avg), float(auc_scores.avg)]   
                    
    elif model_args.dp.split('-')[0] == 'sg': 
        for i, (input_i, input_s, target, batch_cif_ids) in enumerate(val_loader):
            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(input_i[0].cuda(non_blocking=True)),
                                 Variable(torch.cat((input_i[0], torch.zeros(input_i[0].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                                 Variable(input_i[1].cuda(non_blocking=True)),
                                 input_i[2].cuda(non_blocking=True),
                                 [crys_idx_i.cuda(non_blocking=True) for crys_idx_i in input_i[3]],
                                 Variable(input_s[0].cuda(non_blocking=True)),
                                 Variable(torch.cat((input_s[0], torch.zeros(input_s[0].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                                 input_s[1].cuda(non_blocking=True),
                                 Variable(input_s[2].cuda(non_blocking=True)),
                                 [crys_idx_s.cuda(non_blocking=True) for crys_idx_s in input_s[3]])
            else:
                with torch.no_grad():
                    input_var = (Variable(input_i[0]),
                                 Variable(torch.cat((input_i[0], torch.zeros(input_i[0].shape[1]).unsqueeze(0)),dim = 0)),
                                 Variable(input_i[1]),                                 
                                 input_i[2],
                                 input_i[3],
                                 Variable(input_s[0]),
                                 Variable(torch.cat((input_s[0], torch.zeros(input_s[0].shape[1]).unsqueeze(0)),dim = 0)),
                                 Variable(input_s[1]),
                                 input_s[2],
                                 input_s[3])     
            
            if model_args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            with torch.no_grad():
                if args.cuda:
                    target_var = Variable(target_normed.cuda(non_blocking=True))
                else:
                    target_var = Variable(target_normed)
            
            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            if model_args.task == 'regression':
                rmse_error = rmse(normalizer.denorm(output.data.cpu()), target)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                rmse_errors.update(rmse_error, target.size(0))
                mae_errors.update(mae_error, target.size(0))
                if test:
                    test_pred = torch.argmax(torch.exp(output.data.cpu()), axis=1)
                    test_target = target
                    test_preds += test_pred.tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.argmax(torch.exp(output.data.cpu()), axis=1)
                    test_target = target
                    test_prob = np.exp(output.data.cpu().numpy())[:, 1]
                    test_preds += test_pred.tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
                    test_probs += test_prob.tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if model_args.task == 'regression':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                          'RMSE {rmse_errors.val:.3f} ({rmse_errors.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, mae_errors=mae_errors, 
                        rmse_errors=rmse_errors))
                    temp = [float(losses.avg), float(mae_errors.avg), float(rmse_errors.avg)]
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        accu=accuracies, prec=precisions, recall=recalls,
                        f1=fscores, auc=auc_scores))
                    temp = [float(losses.avg), float(precisions.avg), float(auc_scores.avg)]

    elif model_args.dp.split('-')[0] == 'dg': 
        for i, (input_d, input_i, input_s, target, batch_cif_ids) in enumerate(val_loader):
            if args.cuda:
                with torch.no_grad():
                    input_var = (Variable(input_d[0].cuda(non_blocking=True)),
                                 Variable(torch.cat((input_d[1], torch.zeros(input_d[1].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                                 Variable(input_d[2].cuda(non_blocking=True)),
                                 input_d[3].cuda(non_blocking=True),
                                 [crys_idx_d.cuda(non_blocking=True) for crys_idx_d in input_d[4]],
                                 Variable(input_i[0].cuda(non_blocking=True)),
                                 Variable(torch.cat((input_i[0], torch.zeros(input_i[0].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                                 Variable(input_i[1].cuda(non_blocking=True)),
                                 input_i[2].cuda(non_blocking=True),
                                 [crys_idx_i.cuda(non_blocking=True) for crys_idx_i in input_i[3]],
                                 Variable(input_s[0].cuda(non_blocking=True)),
                                 Variable(torch.cat((input_s[0], torch.zeros(input_s[0].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                                 Variable(input_s[1].cuda(non_blocking=True)),
                                 input_s[2].cuda(non_blocking=True),
                                 [crys_idx_s.cuda(non_blocking=True) for crys_idx_s in input_s[3]])
            else:
                with torch.no_grad():
                    input_var = (Variable(input_d[0]),
                                 Variable(torch.cat((input_d[1], torch.zeros(input_d[1].shape[1]).unsqueeze(0)),dim = 0)),
                                 Variable(input_d[2]),                                 
                                 input_d[3],
                                 input_d[4],
                                 Variable(input_i[0]),
                                 Variable(torch.cat((input_i[0], torch.zeros(input_i[0].shape[1]).unsqueeze(0)),dim = 0)),
                                 Variable(input_i[1]),                                 
                                 input_i[2],
                                 input_i[3],
                                 Variable(input_s[0]),
                                 Variable(torch.cat((input_s[0], torch.zeros(input_s[0].shape[1]).unsqueeze(0)),dim = 0)),
                                 Variable(input_s[1]),
                                 input_s[2],
                                 input_s[3])     
            
            if model_args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            with torch.no_grad():
                if args.cuda:
                    target_var = Variable(target_normed.cuda(non_blocking=True))
                else:
                    target_var = Variable(target_normed)
            
            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            if model_args.task == 'regression':
                rmse_error = rmse(normalizer.denorm(output.data.cpu()), target)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                rmse_errors.update(rmse_error, target.size(0))
                mae_errors.update(mae_error, target.size(0))
                if test:
                    test_pred = torch.argmax(torch.exp(output.data.cpu()), axis=1)
                    test_target = target
                    test_preds += test_pred.tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))
                if test:
                    test_pred = torch.argmax(torch.exp(output.data.cpu()), axis=1)
                    test_target = target
                    test_prob = np.exp(output.data.cpu().numpy())[:, 1]
                    test_preds += test_pred.tolist()
                    test_targets += test_target.view(-1).tolist()
                    test_cif_ids += batch_cif_ids
                    test_probs += test_prob.tolist()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if model_args.task == 'regression':
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                          'RMSE {rmse_errors.val:.3f} ({rmse_errors.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses, mae_errors=mae_errors,
                        rmse_errors=rmse_errors))
                    temp = [float(losses.avg), float(mae_errors.avg), float(rmse_errors.avg)]
                else:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, loss=losses,
                        accu=accuracies, prec=precisions, recall=recalls,
                        f1=fscores, auc=auc_scores))
                    temp = [float(losses.avg), float(precisions.avg), float(auc_scores.avg)]

    if model_args.cnvtol == 'STD':
        if model_args.task == 'regression':
            name = 'MAE_' + str(round(float(mae_errors.avg),4))
        else:
            name = 'AUC_' + str(round(float(auc_scores.avg),4))	
    elif model_args.cnvtol == 'PREC':
        name = 'PREC_' + str(round(float(precisions.avg),4))
    elif model_args.cnvtol == 'RMSE':
        name = 'RMSE_' + str(round(float(rmse_errors.avg),4))

    if test:
        star_label = '**'
        import csv
        with open('BestModel/test_results_' + str(accuracies.avg) + '_' + str(precisions.avg) + '_' + str(recalls.avg) + '_' + str(fscores.avg) + '_' + str(auc_scores.avg) + '.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred, prob in zip(test_cif_ids, test_targets, test_preds, test_probs):
                writer.writerow((cif_id, target, pred, prob))
    else:
        star_label = '*'
    if model_args.task == 'regression':
        if model_args.cnvtol == 'STD':
            print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                            mae_errors=mae_errors))
            return mae_errors.avg
        elif model_args.cnvtol == 'RMSE':
            print(' {star} RMSE {rmse_errors.avg:.3f}'.format(star=star_label,
                                                            rmse_errors=rmse_errors))
            return rmse_errors.avg            
    else:
        if model_args.cnvtol == 'STD':
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                     auc=auc_scores))
            return auc_scores.avg
        elif model_args.cnvtol == 'PREC':
            print(' {star} Precision {prec.avg:.3f}'.format(star=star_label,
                                                     prec=precisions))
            return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))

def rmse(prediction, target):
    """
    Computes the root mean squared error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.sqrt(torch.mean(torch.pow(torch.abs(target - prediction),2)))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'BestModel/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        

if __name__ == '__main__':
    main()
