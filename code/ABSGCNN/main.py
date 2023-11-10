import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from cgcnn.data import StructureData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

from sgcnn.SIdata import SIStructureData
from sgcnn.SIdata import SIcollate_pool, SIget_train_val_test_loader
from sgcnn.SImodel import SICrystalGraphConvNet

from dgcnn.DSIdata import DSIStructureData
from dgcnn.DSIdata import DSIcollate_pool, DSIget_train_val_test_loader
from dgcnn.DSImodel import DSICrystalGraphConvNet

parser = argparse.ArgumentParser(description='Crystal Graph Convolutional Neural Networks')
parser.add_argument('data_options', metavar='OPTIONS', nargs='+',
                    help='dataset options, started with the path to root dir, '
                         'then other options')
parser.add_argument('--task', choices=['regression', 'classification'],
                    default='regression', help='complete a regression or '
                                                   'classification task (default: regression)')
parser.add_argument('--structure', '-sc', choices=['vasp', 'cif'],
                    default='vasp', help='load structurs by using *.vasp or '
                                                   '*.cif (default: vasp)')
parser.add_argument('--dp', '--data-processing', default='cg', type = str, help='convert structurs to vector format (default: cg)')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')#action为开关
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run (default: 1000)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: '
                                       '0.01)')
parser.add_argument('--lr-milestones', default=[100], nargs='+', type=int,
                    metavar='N', help='milestones for scheduler (default: '
                                      '[100])')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
train_group = parser.add_mutually_exclusive_group()#互斥参数
train_group.add_argument('--train-ratio', default=None, type=float, metavar='N',
                    help='number of training data to be loaded (default none)')
train_group.add_argument('--train-size', default=None, type=int, metavar='N',
                         help='number of training data to be loaded (default none)')
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument('--val-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of validation data to be loaded (default '
                         '0.1)')
valid_group.add_argument('--val-size', default=None, type=int, metavar='N',
                         help='number of validation data to be loaded (default '
                              '1000)')
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument('--test-ratio', default=0.1, type=float, metavar='N',
                    help='percentage of test data to be loaded (default 0.1)')
test_group.add_argument('--test-size', default=None, type=int, metavar='N',
                        help='number of test data to be loaded (default 1000)')

parser.add_argument('--optim', default='SGD', type=str, metavar='SGD',
                    help='choose an optimizer, SGD or Adam, (default: SGD)')
parser.add_argument('--cnvtol', default='STD', type=str, metavar='STD',
                    help='choose a convergence criteria, (default: STD (regression: MAE; classification: AUC)); PREC (classification); RMSE (regression)')
parser.add_argument('-al', '--atom-fea-len', default=64, metavar='N', 
                    help='number of hidden atom features in conv layers (default 64 (the form is X X-X X-X-X for c si dsi))')
parser.add_argument('-hl', '--h-fea-len', default=128, type=int, metavar='N',
                    help='number of hidden features after pooling (default 128)')
parser.add_argument('-nc', '--n-conv', default=3, metavar='N',
                    help='number of conv layers (default 3 (the form is X X-X X-X-X for c si dsi))')
parser.add_argument('-nh', '--n-h', default=1, type=int, metavar='N',
                    help='number of hidden layers after pooling (default 1)')
parser.add_argument('-r', '--cut-rds', default=8, metavar='N',
                    help='cutoff radius for searching neighbors (default 8 (the form is X X-X X-X-X for c si dsi))')
parser.add_argument('-t', '--nbr-type', default='T', type=str, metavar='T',
                    help='considering the covalent radius in finding neighbor (default T (the form is X X-X X-X-X for c si dsi))')

args = parser.parse_args(sys.argv[1:])#解析参数

args.cuda = not args.disable_cuda and torch.cuda.is_available() #有一个False就Fasle

if args.task == 'regression':
    best_cnvtol_error = 1e10
else:
    best_cnvtol_error = 0.

def main():
    global args, best_cnvtol_error
    #生成打孔画靶文件
    if os.path.exists('target.csv') == False:
        data = pd.read_csv(str(*args.data_options) + '/id_prop.csv', header = None)
        data['2'] = [0]*len(data)
        data['3'] = [0]*len(data)
        data.to_csv('target.csv', header = None, index=None)
    # load data
    if args.dp.split('-')[0] == 'cg':
        args.atom_fea_len, args.n_conv, args.cut_rds, _ = formula_para(args.atom_fea_len, args.n_conv, args.cut_rds, args.nbr_type, 'c')
        dataset = StructureData(*args.data_options, args.structure, cut_rds=args.cut_rds)       
        collate_fn = collate_pool
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            task=args.task,
            return_test=True)
    elif args.dp.split('-')[0] == 'sg':  
        args.atom_fea_len, args.n_conv, args.cut_rds, args.nbr_type = formula_para(args.atom_fea_len, args.n_conv, args.cut_rds, args.nbr_type, 'si')
        dataset = SIStructureData(*args.data_options, args.structure, args.dp.split('-')[1:], cut_rds=args.cut_rds, nbr_type=args.nbr_type)
        collate_fn = SIcollate_pool  
        train_loader, val_loader, test_loader = SIget_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            task=args.task,
            return_test=True)
    elif args.dp.split('-')[0] == 'dg':  
        args.atom_fea_len, args.n_conv, args.cut_rds, args.nbr_type = formula_para(args.atom_fea_len, args.n_conv, args.cut_rds, args.nbr_type, 'dsi')
        dataset = DSIStructureData(*args.data_options, args.structure, args.dp.split('-')[1:], cut_rds=args.cut_rds, nbr_type=args.nbr_type)
        collate_fn = DSIcollate_pool  
        train_loader, val_loader, test_loader = DSIget_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_fn,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            num_workers=args.workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            pin_memory=args.cuda,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            task=args.task,
            return_test=True) 
   
    # obtain target value normalizer
    if args.task == 'classification':
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({'mean': 0., 'std': 1.})
    else:
        if len(dataset) < 500:
            warnings.warn('Dataset has less than 500 data points. '
                          'Lower accuracy is expected. ')
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in
                                sample(range(len(dataset)), 500)]
        if args.dp.split('-')[0] == 'cg':
            _, sample_target, _ = collate_pool(sample_data_list)
        elif args.dp.split('-')[0] == 'sg':  
            _, _, sample_target, _ = SIcollate_pool(sample_data_list)
        elif args.dp.split('-')[0] == 'dg':  
            _, _, _, sample_target, _ = DSIcollate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    if args.dp.split('-')[0] == 'cg':
        structures, _, _ = dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True if args.task ==
                                                           'classification' else False)
    elif args.dp.split('-')[0] == 'sg': 
        structures_i, structures_s, _, _ = dataset[0]
        orig_atom_fea_len = structures_i[0].shape[-1]
        nbr_fea_len = [structures_i[1].shape[-1], structures_s[1].shape[-1]]
        model = SICrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True if args.task ==
                                                           'classification' else False)
    elif args.dp.split('-')[0] == 'dg': 
        structures_d, structures_i, structures_s, _, _ = dataset[0]
        orig_atom_fea_len = structures_d[0].shape[-1]
        nbr_fea_len = [structures_d[2].shape[-1], structures_i[1].shape[-1], structures_s[1].shape[-1]]
        model = DSICrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                    atom_fea_len=args.atom_fea_len,
                                    n_conv=args.n_conv,
                                    h_fea_len=args.h_fea_len,
                                    n_h=args.n_h,
                                    classification=True if args.task ==
                                                           'classification' else False)           
#     print(model)  
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == 'classification':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_cnvtol_error = checkpoint['best_cnvtol_error']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            normalizer.load_state_dict(checkpoint['normalizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones,
                            gamma=0.1)#学习率调整方法
    
    train_data_list = []
    val_data_list = []
    train_cnvtol_list = []
    val_cnvtol_list = []
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        cnvtol_train, train_data =train(train_loader, model, criterion, optimizer, epoch, normalizer)
        train_data_list.append(train_data)
        train_cnvtol_list.append(cnvtol_train)
        # evaluate on validation set
        cnvtol_error, val_data = validate(val_loader, model, criterion, normalizer)
        val_data_list.append(val_data)
        val_cnvtol_list.append(cnvtol_error)
        
        if cnvtol_error != cnvtol_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        # remember the best cnvtol_eror and save checkpoint
        if args.task == 'regression':
            is_best = cnvtol_error < best_cnvtol_error
            best_cnvtol_error = min(cnvtol_error, best_cnvtol_error)
        else:
            is_best = cnvtol_error > best_cnvtol_error
            best_cnvtol_error = max(cnvtol_error, best_cnvtol_error)            
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_cnvtol_error': best_cnvtol_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
            'args': vars(args)
        }, is_best)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    test_cnvtol, test_data = validate(test_loader, model, criterion, normalizer, test=True)
    name = str(round(float(test_cnvtol),4))
    best_checkpoint = torch.load('BestModel/model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    #data_list
    train_data_list = pd.DataFrame(train_data_list)
    val_data_list = pd.DataFrame(val_data_list)
    test_data = pd.DataFrame(test_data)
    writer = pd.ExcelWriter('BestModel/data_'+ args.cnvtol + '_' + name + '.xlsx')
    train_data_list.to_excel(writer,sheet_name='train')
    val_data_list.to_excel(writer,sheet_name='val')
    test_data.to_excel(writer,sheet_name='test')
    writer.close()
    #cnvtol_list
    train_cnvtol_list = pd.DataFrame(train_cnvtol_list)
    val_cnvtol_list = pd.DataFrame(val_cnvtol_list)
    writer_cnvtol = pd.ExcelWriter('BestModel/cnvtol_'+ args.cnvtol + '_' + name + '.xlsx')
    train_cnvtol_list.to_excel(writer_cnvtol,sheet_name='train')
    val_cnvtol_list.to_excel(writer_cnvtol,sheet_name='val')
    writer_cnvtol.close()

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    data_list = []
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
        mae_errors = AverageMeter()
        rmse_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    if args.dp.split('-')[0] == 'cg':
        for i, (input, target, batch_cif_ids) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if args.cuda:
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
            else:
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
            # normalize target
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)

            # punch
            data_target = pd.read_csv('target.csv', header=None)
            if args.task == 'regression':
                diff1 = torch.abs(normalizer.denorm(output.data.cpu()) - target) > 0.35
                diff1 = diff1.numpy().flatten().tolist()
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
                diff2 = torch.abs(normalizer.denorm(output.data.cpu()) - target) >= 0.5
                diff2 = diff2.numpy().flatten().tolist()
                bad_sample2 = idx[diff2]
                for z in bad_sample2:
                    row = data_target[data_target[0].values == z].index.values
                    data_target.update(data_target[3][row]+1)
            else:
                pred_label = np.argmax(np.exp(output.data.cpu()), axis=1)
                target_label = np.squeeze(target)
                diff1 = torch.ne(pred_label, target_label)
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
            data_target.to_csv('target.csv', header = None, index = None)

            # measure accuracy and record loss
            if args.task == 'regression':
                rmse_error = rmse(normalizer.denorm(output.data.cpu()), target)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu(), target.size(0))
                rmse_errors.update(rmse_error, target.size(0))
                mae_errors.update(mae_error, target.size(0))
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if args.task == 'regression':
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                          'RMSE {rmse_errors.val:.3f} ({rmse_errors.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mae_errors=mae_errors, rmse_errors=rmse_errors)
                    )
                    temp = [epoch, float(losses.avg), float(mae_errors.avg), float(rmse_errors.avg)]
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, accu=accuracies,
                        prec=precisions, recall=recalls, f1=fscores,
                        auc=auc_scores)
                    )
                    temp = [epoch, float(losses.avg), float(precisions.avg), float(auc_scores.avg)]
        data_list.append(temp)
                    
    elif args.dp.split('-')[0] == 'sg': 
        for i, (input_i, input_s, target, batch_cif_ids) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            if args.cuda:
                input_var = (Variable(input_i[0].cuda(non_blocking=True)),
                             Variable(torch.cat((input_i[0], torch.zeros(input_i[0].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                             Variable(input_i[1].cuda(non_blocking=True)),
                             input_i[2].cuda(non_blocking=True),
                             [crys_idx_i.cuda(non_blocking=True) for crys_idx_i in input_i[3]],
                             Variable(input_s[0].cuda(non_blocking=True)),
                             Variable(torch.cat((input_s[0], torch.zeros(input_s[0].shape[1]).unsqueeze(0)),dim = 0).cuda(non_blocking=True)),
                             Variable(input_s[1].cuda(non_blocking=True)),
                             Variable(input_s[2].cuda(non_blocking=True)),
                             [crys_idx_s.cuda(non_blocking=True) for crys_idx_s in input_s[3]])
            else:
                input_var = (Variable(input_i[0]),
                             Variable(torch.cat((input_i[0], torch.zeros(input_i[0].shape[1]).unsqueeze(0)),dim = 0)),
                             Variable(input_i[1]),
                             input_i[2],
                             input_i[3],
                             Variable(input_s[0]),
                             Variable(torch.cat((input_s[0], torch.zeros(input_s[0].shape[1]).unsqueeze(0)),dim = 0)),
                             Variable(input_s[1]),
                             Variable(input_s[2]),
                             input_s[3])
            # normalize target
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)
            
            # punch
            data_target = pd.read_csv('target.csv', header=None)
            if args.task == 'regression':
                diff1 = torch.abs(normalizer.denorm(output.data.cpu()) - target) > 0.35
                diff1 = diff1.numpy().flatten().tolist()
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
                diff2 = torch.abs(normalizer.denorm(output.data.cpu()) - target) >= 0.5
                diff2 = diff2.numpy().flatten().tolist()
                bad_sample2 = idx[diff2]
                for z in bad_sample2:
                    row = data_target[data_target[0].values == z].index.values
                    data_target.update(data_target[3][row]+1)
            else:
                pred_label = np.argmax(np.exp(output.data.cpu()), axis=1)
                target_label = np.squeeze(target)
                diff1 = torch.ne(pred_label, target_label)
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
            data_target.to_csv('target.csv', header = None, index = None)

            # measure accuracy and record loss
            if args.task == 'regression':
                rmse_error = rmse(normalizer.denorm(output.data.cpu()), target)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu(), target.size(0))
                rmse_errors.update(rmse_error, target.size(0))
                mae_errors.update(mae_error, target.size(0))
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if args.task == 'regression':
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                          'RMSE {rmse_errors.val:.3f} ({rmse_errors.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mae_errors=mae_errors, rmse_errors=rmse_errors)
                    )
                    temp = [epoch, float(losses.avg), float(mae_errors.avg), float(rmse_errors.avg)]
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, accu=accuracies,
                        prec=precisions, recall=recalls, f1=fscores,
                        auc=auc_scores)
                    )
                    temp = [epoch, float(losses.avg), float(precisions.avg), float(auc_scores.avg)]
        data_list.append(temp)

    elif args.dp.split('-')[0] == 'dg': 
        for i, (input_d, input_i, input_s, target, batch_cif_ids) in enumerate(train_loader):      
            # measure data loading time
            data_time.update(time.time() - end)
            if args.cuda:
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
            # normalize target
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)
            
            # punch
            data_target = pd.read_csv('target.csv', header=None)
            if args.task == 'regression':
                diff1 = torch.abs(normalizer.denorm(output.data.cpu()) - target) > 0.35
                diff1 = diff1.numpy().flatten().tolist()
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
                diff2 = torch.abs(normalizer.denorm(output.data.cpu()) - target) >= 0.5
                diff2 = diff2.numpy().flatten().tolist()
                bad_sample2 = idx[diff2]
                for z in bad_sample2:
                    row = data_target[data_target[0].values == z].index.values
                    data_target.update(data_target[3][row]+1)
            else:
                pred_label = np.argmax(np.exp(output.data.cpu()), axis=1)
                target_label = np.squeeze(target)
                diff1 = torch.ne(pred_label, target_label)
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
            data_target.to_csv('target.csv', header = None, index = None)

            # measure accuracy and record loss
            if args.task == 'regression':
                rmse_error = rmse(normalizer.denorm(output.data.cpu()), target)
                mae_error = mae(normalizer.denorm(output.data.cpu()), target)
                losses.update(loss.data.cpu(), target.size(0))
                rmse_errors.update(rmse_error, target.size(0))
                mae_errors.update(mae_error, target.size(0))
            else:
                accuracy, precision, recall, fscore, auc_score = \
                    class_eval(output.data.cpu(), target)
                losses.update(loss.data.cpu().item(), target.size(0))
                accuracies.update(accuracy, target.size(0))
                precisions.update(precision, target.size(0))
                recalls.update(recall, target.size(0))
                fscores.update(fscore, target.size(0))
                auc_scores.update(auc_score, target.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                if args.task == 'regression':
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})\t'
                          'RMSE {rmse_errors.val:.3f} ({rmse_errors.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, mae_errors=mae_errors, rmse_errors=rmse_errors)
                    )
                    temp = [epoch, float(losses.avg), float(mae_errors.avg), float(rmse_errors.avg)]
                else:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Accu {accu.val:.3f} ({accu.avg:.3f})\t'
                          'Precision {prec.val:.3f} ({prec.avg:.3f})\t'
                          'Recall {recall.val:.3f} ({recall.avg:.3f})\t'
                          'F1 {f1.val:.3f} ({f1.avg:.3f})\t'
                          'AUC {auc.val:.3f} ({auc.avg:.3f})'.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, accu=accuracies,
                        prec=precisions, recall=recalls, f1=fscores,
                        auc=auc_scores)
                    )
                    temp = [epoch, float(losses.avg), float(precisions.avg), float(auc_scores.avg)]
        data_list.append(temp)
        
#     return data_list
    if args.task == 'regression':
        if args.cnvtol == 'STD':
            return mae_errors.avg, data_list
        elif args.cnvtol == 'RMSE':
            return rmse_errors.avg, data_list
    else:
        if args.cnvtol == 'STD':
            return auc_scores.avg, data_list
        elif args.cnvtol == 'PREC':
            return precisions.avg, data_list
        
def validate(val_loader, model, criterion, normalizer, test=False):
    data_list = []
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == 'regression':
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
    if args.dp.split('-')[0] == 'cg':
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
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                with torch.no_grad():
                    target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                with torch.no_grad():
                    target_var = Variable(target_normed)

            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)

            # punch
            data_target = pd.read_csv('target.csv', header=None)
            if args.task == 'regression':
                diff1 = torch.abs(normalizer.denorm(output.data.cpu()) - target) > 0.35
                diff1 = diff1.numpy().flatten().tolist()
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
                diff2 = torch.abs(normalizer.denorm(output.data.cpu()) - target) >= 0.5
                diff2 = diff2.numpy().flatten().tolist()
                bad_sample2 = idx[diff2]
                for z in bad_sample2:
                    row = data_target[data_target[0].values == z].index.values
                    data_target.update(data_target[3][row]+1)
            else:
                pred_label = np.argmax(np.exp(output.data.cpu()), axis=1)
                target_label = np.squeeze(target)
                diff1 = torch.ne(pred_label, target_label)
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
#                    data_target.update(data_target[2][row]+1)
                    data_target.update(data_target[3][row]+1)
            data_target.to_csv('target.csv', header = None, index = None)

            # measure accuracy and record loss
            if args.task == 'regression':
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
                if args.task == 'regression':
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
        data_list.append(temp)   
                    
    elif args.dp.split('-')[0] == 'sg': 
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
            
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                with torch.no_grad():
                    target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                with torch.no_grad():
                    target_var = Variable(target_normed)
            
            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)
            
            # punch
            data_target = pd.read_csv('target.csv', header=None)
            if args.task == 'regression':
                diff1 = torch.abs(normalizer.denorm(output.data.cpu()) - target) > 0.35
                diff1 = diff1.numpy().flatten().tolist()
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
                diff2 = torch.abs(normalizer.denorm(output.data.cpu()) - target) >= 0.5
                diff2 = diff2.numpy().flatten().tolist()
                bad_sample2 = idx[diff2]
                for z in bad_sample2:
                    row = data_target[data_target[0].values == z].index.values
                    data_target.update(data_target[3][row]+1)
            else:
                pred_label = np.argmax(np.exp(output.data.cpu()), axis=1)
                target_label = np.squeeze(target)
                diff1 = torch.ne(pred_label, target_label)
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
#                     data_target.update(data_target[2][row]+1)
                    data_target.update(data_target[3][row]+1)
            data_target.to_csv('target.csv', header = None, index = None)

            # measure accuracy and record loss
            if args.task == 'regression':
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
                if args.task == 'regression':
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
        data_list.append(temp) 

    elif args.dp.split('-')[0] == 'dg': 
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
            
            if args.task == 'regression':
                target_normed = normalizer.norm(target)
            else:
                target_normed = target.view(-1).long()
            if args.cuda:
                with torch.no_grad():
                    target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                with torch.no_grad():
                    target_var = Variable(target_normed)
            
            # compute output
            output = model(*input_var)
            loss = criterion(output, target_var)
            
            # punch
            data_target = pd.read_csv('target.csv', header=None)
            if args.task == 'regression':
                diff1 = torch.abs(normalizer.denorm(output.data.cpu()) - target) > 0.35
                diff1 = diff1.numpy().flatten().tolist()
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
                    data_target.update(data_target[2][row]+1)
                diff2 = torch.abs(normalizer.denorm(output.data.cpu()) - target) >= 0.5
                diff2 = diff2.numpy().flatten().tolist()
                bad_sample2 = idx[diff2]
                for z in bad_sample2:
                    row = data_target[data_target[0].values == z].index.values
                    data_target.update(data_target[3][row]+1)
            else:
                pred_label = np.argmax(np.exp(output.data.cpu()), axis=1)
                target_label = np.squeeze(target)
                diff1 = torch.ne(pred_label, target_label)
                idx = np.array(batch_cif_ids)
                bad_sample1 = idx[diff1]
                for s in bad_sample1:
                    row = data_target[data_target[0].values == s].index.values
#                     data_target.update(data_target[2][row]+1)
                    data_target.update(data_target[3][row]+1)
            data_target.to_csv('target.csv', header = None, index = None)

            # measure accuracy and record loss
            if args.task == 'regression':
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
                if args.task == 'regression':
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
        data_list.append(temp) 
                    
    if test:
        star_label = '**'
        import csv
        if args.task == 'regression':
            with open('BestModel/test_results_' + str(mae_errors.avg) + '_' + str(rmse_errors.avg) + '.csv', 'w') as f:
                writer = csv.writer(f)
                for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                                test_preds):
                    writer.writerow((cif_id, target, pred)) 
        else:
            with open('BestModel/test_results_' + str(accuracies.avg) + '_' + str(precisions.avg) + '_' + str(recalls.avg) + '_' + str(fscores.avg) + '_' + str(auc_scores.avg) + '.csv', 'w') as f:
                writer = csv.writer(f)
                for cif_id, target, pred, prob in zip(test_cif_ids, test_targets,
                                                test_preds, test_probs):
                    writer.writerow((cif_id, target, pred, prob)) 
                    
    else:
        star_label = '*'
    if args.task == 'regression':
        if args.cnvtol == 'STD':
            print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label,
                                                            mae_errors=mae_errors))
            return mae_errors.avg, data_list
        elif args.cnvtol == 'RMSE':
            print(' {star} RMSE {rmse_errors.avg:.3f}'.format(star=star_label,
                                                            rmse_errors=rmse_errors))
            return rmse_errors.avg, data_list            
    else:
        if args.cnvtol == 'STD':
            print(' {star} AUC {auc.avg:.3f}'.format(star=star_label,
                                                     auc=auc_scores))
            return auc_scores.avg, data_list
        elif args.cnvtol == 'PREC':
            print(' {star} Precision {prec.avg:.3f}'.format(star=star_label,
                                                     prec=precisions))
            return auc_scores.avg, data_list

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


def save_checkpoint(state, is_best, filename='BestModel/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'BestModel/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def formula_para(atom_fea_len, n_conv, cut_rds, nbr_type, structure_type):
    if structure_type == 'c':
        atom_fea_len = int(atom_fea_len)
        n_conv = int(n_conv)
        cut_rds = float(cut_rds)
        nbr_type = str(nbr_type)
    elif structure_type == 'si':
        if len(str(atom_fea_len).split('-')) != 2:
            atom_fea_len = [int(str(atom_fea_len).split('-')[0])]*2
        else:
            atom_fea_len = [int(str(atom_fea_len).split('-')[i]) for i in range(2)]
        if len(str(n_conv).split('-')) != 2:
            n_conv = [int(str(n_conv).split('-')[0])]*2
        else:
            n_conv = [int(str(n_conv).split('-')[i]) for i in range(2)]
        if len(str(cut_rds).split('-')) != 2:
            cut_rds = [float(str(cut_rds).split('-')[0])]*2
        else:
            cut_rds = [float(str(cut_rds).split('-')[i]) for i in range(2)]
        if len(nbr_type.split('-')) != 2:
            nbr_type = [nbr_type.split('-')[0]]*2
        else:
            nbr_type = [nbr_type.split('-')[i] for i in range(2)]
    elif structure_type == 'dsi':
        if len(str(atom_fea_len).split('-')) != 3:
            atom_fea_len = [int(str(atom_fea_len).split('-')[0])]*3
        else:
            atom_fea_len = [int(str(atom_fea_len).split('-')[i]) for i in range(3)]
        if len(str(n_conv).split('-')) != 3:
            n_conv = [int(str(n_conv).split('-')[0])]*3
        else:
            n_conv = [int(str(n_conv).split('-')[i]) for i in range(3)]
        if len(str(cut_rds).split('-')) != 3:
            cut_rds = [float(str(cut_rds).split('-')[0])]*3
        else:
            cut_rds = [float(str(cut_rds).split('-')[i]) for i in range(3)]
        if len(nbr_type.split('-')) != 3:
            nbr_type = [nbr_type.split('-')[0]]*3
        else:
            nbr_type = [nbr_type.split('-')[i] for i in range(3)]
    return atom_fea_len, n_conv, cut_rds, nbr_type


if __name__ == '__main__':
    main()
