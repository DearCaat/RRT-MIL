## Modified by DTFD@https://github.com/hrzhang1123/DTFD-MIL
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import os
import json
import wandb
import random
import argparse
import numpy as np
from utils import *
import time
from collections import OrderedDict
# from dataloader import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from Model.Restnet import *
from Model.Attention import Attention_with_Classifier
from Model.Attention import Attention_Gated as Attention
from Model.network import Classifier_1fc, DimReduction
from torch.utils.data import WeightedRandomSampler,RandomSampler

def one_fold(params,k,train_p,train_l,test_p,test_l,val_p,val_l,epoch_step,writer):

        torch.manual_seed(params.seed)
        torch.cuda.manual_seed(params.seed)
        np.random.seed(params.seed)
        random.seed(params.seed)
        torch.cuda.manual_seed_all(params.seed)

        in_chn = 1024 #1024
        classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
        attention = Attention(params.mDim).to(params.device)
        dimReduction = DimReduction(in_chn, params.mDim, numLayer_Res=params.numLayer_Res,dropout=params.dropout,act=params.act).to(params.device)
        attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

        if params.init_type != 'none':
            init_path=os.path.join(params.initfc,'fold_{fold}_best_model.pth'.format(fold=k))
            pre_dic = torch.load(init_path)
            if params.init_type=='fc':
                print("############# Model FC initing <--------")
                dimReduction.load_state_dict(pre_dic['dim_reduction'])
            else:
                print("--------> Model all initing <--------")
                classifier.load_state_dict(pre_dic['classifier'],strict=False)
                attention.load_state_dict(pre_dic['attention'],strict=False)
                dimReduction.load_state_dict(pre_dic['dim_reduction'],strict=False)
                attCls.load_state_dict(pre_dic['att_classifier'],strict=False)
            print('fold_{fold}_Inited'.format(fold=k))

        if params.isPar:
            classifier = torch.nn.DataParallel(classifier)
            attention = torch.nn.DataParallel(attention)
            dimReduction = torch.nn.DataParallel(dimReduction)
            attCls = torch.nn.DataParallel(attCls)

        ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

        if not os.path.exists(params.log_dir):
            os.makedirs(params.log_dir)
        log_dir = os.path.join(params.log_dir, 'log.txt')
        # save_dir = os.path.join(params.log_dir,params.name, 'best_model.pth')
        save_dir = os.path.join(params.log_dir,params.name, 'fold_{fold}_best_model.pth'.format(fold=k))
        if not os.path.exists(os.path.join(params.log_dir,params.name)):
            os.makedirs(os.path.join(params.log_dir,params.name))
        z = vars(params).copy()
        with open(log_dir, 'a') as f:
            f.write(json.dumps(z))
        log_file = open(log_dir, 'a')

        # load data
        SlideNames_train, FeatList_train, Label_train = reOrganize_mDATA_v2(train_p[k],train_l[k],dataset=params.dataset,dataset_root=params.dataset_root)
        
        if params.val_ratio != 0.:
            SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA_v2(val_p[k],val_l[k],dataset=params.dataset,dataset_root=params.dataset_root)
            if params.always_test:
                SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_v2(test_p[k],test_l[k],dataset=params.dataset,dataset_root=params.dataset_root)
        else:
            SlideNames_val, FeatList_val, Label_val = reOrganize_mDATA_v2(test_p[k],test_l[k],dataset=params.dataset,dataset_root=params.dataset_root)
        
        print_log(f'training slides: {len(SlideNames_train)}, validation slides: {len(SlideNames_val)}', log_file)

        trainable_parameters = []
        trainable_parameters += list(classifier.parameters())
        trainable_parameters += list(attention.parameters())
        trainable_parameters += list(dimReduction.parameters())

        optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)
        optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=params.lr,  weight_decay=params.weight_decay)

        if params.sche == 'step':
            scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam0, epoch_step, gamma=params.lr_decay_ratio)
            scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_adam1, epoch_step, gamma=params.lr_decay_ratio)
        elif params.sche == 'cosine':
            scheduler0 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam0, params.EPOCH, 0)
            scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adam1, params.EPOCH, 0)

        best_auc = 0
        best_val_auc_test = 0
        best_epoch = -1
        test_auc = 0
        if params.early_stopping:
            earlystop = EarlyStopping(stop_epoch=70 if params.dataset == 'tcga' else 130,patience=20 if params.dataset == 'tcga' else 30)
        else:
            earlystop = None

        if params.fix_train_random:
            torch.manual_seed(params.seed)
            torch.cuda.manual_seed(params.seed)
            np.random.seed(params.seed)
            random.seed(params.seed)
            torch.cuda.manual_seed_all(params.seed)

        train_time_meter = AverageMeter()
        for ii in range(params.EPOCH):

            for param_group in optimizer_adam1.param_groups:
                curLR = param_group['lr']
                print_log(f' current learn rate {curLR}', log_file )
            
            time = train_attention_preFeature_DTFD(classifier=classifier, dimReduction=dimReduction, attention=attention, UClassifier=attCls, mDATA_list=(SlideNames_train, FeatList_train, Label_train), ce_cri=ce_cri,
                                                    optimizer0=optimizer_adam0, optimizer1=optimizer_adam1, epoch=ii, params=params, f_log=log_file, writer=writer, numGroup=params.numGroup, total_instance=params.total_instance, distill=params.distill_type)
            train_time_meter.update(time)
            print(train_time_meter.avg)
            print_log(f'>>>>>>>>>>> Validation Epoch: {ii}', log_file)
            auc_val,mF1_1,macc_1,mprec_1, mrecal_1,test_loss = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                            UClassifier=attCls, mDATA_list=(SlideNames_val, FeatList_val, Label_val), criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup_test, total_instance=params.total_instance_test, distill=params.distill_type)
            print_log(' ', log_file)
            if params.wandb:
                rowd = OrderedDict([
                    ("val_acc",macc_1),
                    ("val_precesion",mprec_1),
                    ("val_recall",mrecal_1),
                    ("val_fscore",mF1_1),
                    ("val_auc",auc_val),
                    ("val_loss",test_loss.avg),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)
            
            # testing when the training phase
            if params.always_test:
                auc_val_test,mF1_1_test,macc_1_test,mprec_1_test, mrecal_1_test,test_loss_test = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                                                UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup_test, total_instance=params.total_instance_test, distill=params.distill_type)
            
                if params.wandb:
                    rowd = OrderedDict([
                        ("val_acc_test",macc_1_test),
                        ("val_precesion_test",mprec_1_test),
                        ("val_recall_test",mrecal_1_test),
                        ("val_fscore_test",mF1_1_test),
                        ("val_auc_test",auc_val_test),
                        ("val_loss_test",test_loss_test.avg),
                    ])

                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)
                if auc_val_test > best_val_auc_test:
                    best_val_auc_test = auc_val_test

            if ii > int(params.EPOCH*params.save_best_model_stage):
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_epoch = ii

                    if params.isSaveModel:
                        tsave_dict = {
                            'classifier': classifier.state_dict(),
                            'dim_reduction': dimReduction.state_dict(),
                            'attention': attention.state_dict(),
                            'att_classifier': attCls.state_dict()
                        }
                        torch.save(tsave_dict, save_dir)

                print_log(f' test auc: {test_auc}, from epoch {best_epoch}', log_file)

            scheduler0.step()
            scheduler1.step()

            if earlystop is not None:
                earlystop(ii,-auc_val,None)
                stop = earlystop.early_stop
            else:
                stop = False

            if stop:
                break
        if params.wandb:
            wandb.log({
                "val_best_auc":best_auc,
            })
        del SlideNames_train, FeatList_train,Label_train,SlideNames_val, FeatList_val, Label_val
        best_cpt = torch.load(save_dir)
        classifier.load_state_dict(best_cpt['classifier'])
        dimReduction.load_state_dict(best_cpt['dim_reduction'])
        attention.load_state_dict(best_cpt['attention'])
        attCls.load_state_dict(best_cpt['att_classifier'])
        print_log(f'>>>>>>>>>>> Test Epoch: {best_epoch}', log_file)
        SlideNames_test, FeatList_test, Label_test = reOrganize_mDATA_v2(test_p[k],test_l[k],dataset=params.dataset,dataset_root=params.dataset_root)
        print_log(f'test slides: {len(SlideNames_test)}', log_file)
        tauc,tf1,tacc,tprec, trecal,_ = test_attention_DTFD_preFeat_MultipleMean(classifier=classifier, dimReduction=dimReduction, attention=attention,
                                UClassifier=attCls, mDATA_list=(SlideNames_test, FeatList_test, Label_test), criterion=ce_cri, epoch=ii,  params=params, f_log=log_file, writer=writer, numGroup=params.numGroup_test, total_instance=params.total_instance_test, distill=params.distill_type)

        if params.wandb:
            rowd = OrderedDict([
                ("test_acc",tacc),
                ("test_precesion",tprec),
                ("test_recall",trecal),
                ("test_fscore",tf1),
                ("test_auc",tauc),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)

        del SlideNames_test, FeatList_test, Label_test
        
        return tacc,tprec,trecal,tf1,tauc,best_val_auc_test

def main(params):

    # params = parser.parse_args()

    if params.wandb:
        wandb.init(project=params.project, entity='dearcat',name=params.name,config=params)

    epoch_step = json.loads(params.epoch_step)
    writer = SummaryWriter(os.path.join(params.log_dir, 'LOG', params.name))

    # set the random seed
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    if params.dataset.lower() == 'camelyon16':
        label_path=os.path.join(params.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    elif params.dataset.lower() == 'tcga':
        label_path=os.path.join(params.dataset_root,'label.csv')
        p, l = get_patient_label(label_path)
        index = [i for i in range(len(p))]
        random.shuffle(index)
        p = p[index]
        l = l[index]

    if params.cv_fold > 1:
        train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(params.cv_fold, p, l,params.val_ratio,params.val_label_balance)

    acs, pre, rec,fs,auc=[],[],[],[],[]
    val_test_auc=[]
    for k in range(0, params.cv_fold):

        best_test_acc,best_test_prec,best_test_recal,best_test_f1,best_test_auc,best_val_auc_test = one_fold(params,k,train_p,train_l,test_p,test_l,val_p,val_l,epoch_step,writer)

        acs.append(best_test_acc)
        pre.append(best_test_prec)
        rec.append(best_test_recal)
        fs.append(best_test_f1)
        auc.append(best_test_auc)
        if params.always_test:
            val_test_auc.append(best_val_auc_test)
        

    if params.always_test:
        if params.wandb:
            wandb.log({
                "cross_val/val_test_auc_mean":np.mean(np.array(val_test_auc)),
                "cross_val/val_test_auc_std":np.std(np.array(val_test_auc)),

            })
    if params.wandb:
        wandb.log({
            "cross_val/acc_mean":np.mean(np.array(acs)),
            "cross_val/auc_mean":np.mean(np.array(auc)),
            "cross_val/f1_mean":np.mean(np.array(fs)),
            "cross_val/pre_mean":np.mean(np.array(pre)),
            "cross_val/recall_mean":np.mean(np.array(rec)),
            "cross_val/acc_std":np.std(np.array(acs)),
            "cross_val/auc_std":np.std(np.array(auc)),
            "cross_val/f1_std":np.std(np.array(fs)),
            "cross_val/pre_std":np.std(np.array(pre)),
            "cross_val/recall_std":np.std(np.array(rec)),
        })

    print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
    print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
    print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
    print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
    print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

def test_attention_DTFD_preFeat_MultipleMean(mDATA_list, classifier, dimReduction, attention, UClassifier, epoch, criterion=None,  params=None, f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):

    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    SlideNames, FeatLists, Label = mDATA_list
    instance_per_group = total_instance // numGroup

    test_loss0 = AverageMeter()
    test_loss1 = AverageMeter()

    gPred_0 = torch.FloatTensor().to(params.device)
    gt_0 = torch.LongTensor().to(params.device)
    gPred_1 = torch.FloatTensor().to(params.device)
    gt_1 = torch.LongTensor().to(params.device)

    with torch.no_grad():

        numSlides = len(SlideNames)
        numIter = numSlides // params.batch_size_v
        tIDX = list(range(numSlides))

        for idx in range(numIter):

            tidx_slide = tIDX[idx * params.batch_size_v:(idx + 1) * params.batch_size_v]
            slide_names = [SlideNames[sst] for sst in tidx_slide]
            tlabel = [Label[sst] for sst in tidx_slide]
            label_tensor = torch.LongTensor(tlabel).to(params.device)
            batch_feat = [ FeatLists[sst].to(params.device) for sst in tidx_slide ]

            for tidx, tfeat in enumerate(batch_feat):
                tslideName = slide_names[tidx]
                tslideLabel = label_tensor[tidx].unsqueeze(0)
                midFeat = dimReduction(tfeat)

                AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

                allSlide_pred_softmax = []

                for jj in range(params.num_MeanInference):

                    feat_index = list(range(tfeat.shape[0]))
                    random.shuffle(feat_index)
                    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
                    index_chunk_list = [sst.tolist() for sst in index_chunk_list]

                    slide_d_feat = []
                    slide_sub_preds = []
                    slide_sub_labels = []

                    for tindex in index_chunk_list:
                        slide_sub_labels.append(tslideLabel)
                        idx_tensor = torch.LongTensor(tindex).to(params.device)
                        tmidFeat = midFeat.index_select(dim=0, index=idx_tensor)

                        tAA = AA.index_select(dim=0, index=idx_tensor)
                        tAA = torch.softmax(tAA, dim=0)
                        tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                        tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

                        tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                        slide_sub_preds.append(tPredict)

                        patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                        patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                        patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                        _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

                        if distill == 'MaxMinS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx_min = sort_idx[-instance_per_group:].long()
                            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'MaxS':
                            topk_idx_max = sort_idx[:instance_per_group].long()
                            topk_idx = topk_idx_max
                            d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                            slide_d_feat.append(d_inst_feat)
                        elif distill == 'AFS':
                            slide_d_feat.append(tattFeat_tensor)

                    slide_d_feat = torch.cat(slide_d_feat, dim=0)
                    slide_sub_preds = torch.cat(slide_sub_preds, dim=0)
                    slide_sub_labels = torch.cat(slide_sub_labels, dim=0)

                    gPred_0 = torch.cat([gPred_0, slide_sub_preds], dim=0)
                    gt_0 = torch.cat([gt_0, slide_sub_labels], dim=0)
                    loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
                    test_loss0.update(loss0.item(), numGroup)

                    gSlidePred = UClassifier(slide_d_feat)
                    allSlide_pred_softmax.append(torch.softmax(gSlidePred, dim=1))

                allSlide_pred_softmax = torch.cat(allSlide_pred_softmax, dim=0)
                allSlide_pred_softmax = torch.mean(allSlide_pred_softmax, dim=0).unsqueeze(0)
                gPred_1 = torch.cat([gPred_1, allSlide_pred_softmax], dim=0)
                gt_1 = torch.cat([gt_1, tslideLabel], dim=0)

                loss1 = F.nll_loss(allSlide_pred_softmax, tslideLabel)
                test_loss1.update(loss1.item(), 1)

    gPred_0 = torch.softmax(gPred_0, dim=1)
    gPred_0 = gPred_0[:, -1]
    gPred_1 = gPred_1[:, -1]

    macc_0, mprec_0, mrecal_0, mspec_0, mF1_0, auc_0 = eval_metric(gPred_0, gt_0)
    macc_1, mprec_1, mrecal_1, mspec_1, mF1_1, auc_1 = eval_metric(gPred_1, gt_1)

    print_log(f'  First-Tier acc {macc_0}, precision {mprec_0}, recall {mrecal_0}, specificity {mspec_0}, F1 {mF1_0}, AUC {auc_0}', f_log)
    print_log(f'  Second-Tier acc {macc_1}, precision {mprec_1}, recall {mrecal_1}, specificity {mspec_1}, F1 {mF1_1}, AUC {auc_1}', f_log)

    writer.add_scalar(f'auc_0 ', auc_0, epoch)
    writer.add_scalar(f'auc_1 ', auc_1, epoch)

    return auc_1,mF1_1.cpu(),macc_1.cpu(),mprec_1.cpu(), mrecal_1.cpu(),test_loss1


def train_attention_preFeature_DTFD(mDATA_list, classifier, dimReduction, attention, UClassifier,  optimizer0, optimizer1, epoch, ce_cri=None, params=None,
                                          f_log=None, writer=None, numGroup=3, total_instance=3, distill='MaxMinS'):
    start = time.time()
    SlideNames_list, mFeat_list, Label_dict = mDATA_list

    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()

    instance_per_group = total_instance // numGroup

    Train_Loss0 = AverageMeter()
    Train_Loss1 = AverageMeter()

    numSlides = len(SlideNames_list)
    numIter = numSlides // params.batch_size

    if params.weighted:
        weights  = make_weights_for_balanced_classes_split(Label_dict)
        if params.fix_loader_random:
            big_seed_list = 7784414403328510413
            generator = torch.Generator()
            generator.manual_seed(big_seed_list)  
            tIDX = list(WeightedRandomSampler(weights,len(weights),generator=generator))
        else:
            tIDX = list(WeightedRandomSampler(weights,len(weights)))
    else:
        if params.fix_loader_random:
            big_seed_list = 7784414403328510413
            generator = torch.Generator()
            generator.manual_seed(big_seed_list)  
            tIDX = list(RandomSampler(range(numSlides),generator=generator))
        else:
            tIDX = list(RandomSampler(range(numSlides)))


    # torch.autograd.set_detect_anomaly(True)
    for idx in range(numIter):

        tidx_slide = tIDX[idx * params.batch_size:(idx + 1) * params.batch_size]

        tslide_name = [SlideNames_list[sst] for sst in tidx_slide]
        tlabel = [Label_dict[sst] for sst in tidx_slide]

        label_tensor = torch.LongTensor(tlabel).to(params.device)

        for tidx, (tslide, slide_idx) in enumerate(zip(tslide_name, tidx_slide)):
            tslideLabel = label_tensor[tidx].unsqueeze(0)

            slide_pseudo_feat = []
            slide_sub_preds = []
            slide_sub_labels = []

            tfeat_tensor = mFeat_list[slide_idx]
            tfeat_tensor = tfeat_tensor.to(params.device)

            feat_index = list(range(tfeat_tensor.shape[0]))
            if params.patch_shuffle:
                random.shuffle(feat_index)
            index_chunk_list = np.array_split(np.array(feat_index), numGroup)
            index_chunk_list = [sst.tolist() for sst in index_chunk_list]

            for tindex in index_chunk_list:
                slide_sub_labels.append(tslideLabel)
                subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=torch.LongTensor(tindex).to(params.device))
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)

                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
                tPredict = classifier(tattFeat_tensor)  ### 1 x 2
                slide_sub_preds.append(tPredict)

                patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
                patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
                patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

                _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

                MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   ##########################
                max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
                af_inst_feat = tattFeat_tensor

                if distill == 'MaxMinS':
                    slide_pseudo_feat.append(MaxMin_inst_feat)
                elif distill == 'MaxS':
                    slide_pseudo_feat.append(max_inst_feat)
                elif distill == 'AFS':
                    slide_pseudo_feat.append(af_inst_feat)

            slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

            ## optimization for the first tier
            slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
            slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
            loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
            optimizer0.zero_grad()
            loss0.backward(retain_graph=True)
            if params.grad_clipping > 0:
                torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
                torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
            

            ## optimization for the second tier
            gSlidePred = UClassifier(slide_pseudo_feat)
            loss1 = ce_cri(gSlidePred, tslideLabel).mean()
            optimizer1.zero_grad()
            loss1.backward()
            if params.grad_clipping > 0:
                torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), params.grad_clipping)
            optimizer0.step()
            optimizer1.step()

            Train_Loss0.update(loss0.item(), numGroup)
            Train_Loss1.update(loss1.item(), 1)

        if idx % params.train_show_freq == 0:
            tstr = 'epoch: {} idx: {}'.format(epoch, idx)
            tstr += f' First Loss : {Train_Loss0.avg}, Second Loss : {Train_Loss1.avg} '
            print_log(tstr, f_log)

    end = time.time()
    writer.add_scalar(f'train_loss_0 ', Train_Loss0.avg, epoch)
    writer.add_scalar(f'train_loss_1 ', Train_Loss1.avg, epoch)
    return end-start

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


def print_log(tstr, f):
    # with open(dir, 'a') as f:
    f.write('\n')
    f.write(tstr)
    print(tstr)


def reOrganize_mDATA_test(mDATA):

    tumorSlides = os.listdir(testMask_dir)
    tumorSlides = [sst.split('.')[0] for sst in tumorSlides]

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name in tumorSlides:
            label = 1
        else:
            label = 0
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label

def reOrganize_mDATA_v2(mDATA,label,dataset,dataset_root):

    SlideNames = []
    FeatList = []
    Label = []

    if dataset == 'tcga':
        all_pts = os.listdir(os.path.join(dataset_root,'pt_files'))

        slide_name_list = []
        slide_label = []

        for i,_patient_name in enumerate(mDATA):

            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                slide_name_list.append(_sides[_idx])
                slide_label.append(label[i])
        for i in range(len(slide_name_list)):

            SlideNames.append(slide_name_list[i][:-3])

            _label = 0 if slide_label[i] == 'LUAD' else 1
            Label.append(_label)
            file_path = slide_name_list[i]
            tfeat = torch.load(os.path.join(dataset_root,'pt_files',file_path))
            FeatList.append(tfeat)
   
    else:
        for i,slide_name in enumerate(mDATA):
            SlideNames.append(slide_name)

            if dataset == 'camelyon16':
                dir_path=dataset_root
                file_path = os.path.join(dir_path, 'pt',slide_name+'.pt')

                tfeat = torch.load(file_path)
                FeatList.append(tfeat)
                if int(label[i]) == 1:
                    Label.append(1)
                else:
                    Label.append(0)       

    return SlideNames, FeatList, Label


def reOrganize_mDATA(mDATA):

    SlideNames = []
    FeatList = []
    Label = []
    for slide_name in mDATA.keys():
        SlideNames.append(slide_name)

        if slide_name.startswith('tumor'):
            label = 1
        elif slide_name.startswith('normal'):
            label = 0
        else:
            raise RuntimeError('Undefined slide type')
        Label.append(label)

        patch_data_list = mDATA[slide_name]
        featGroup = []
        for tpatch in patch_data_list:
            tfeat = torch.from_numpy(tpatch['feature'])
            featGroup.append(tfeat.unsqueeze(0))
        featGroup = torch.cat(featGroup, dim=0) ## numPatch x fs
        FeatList.append(featGroup)

    return SlideNames, FeatList, Label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DTFD Training Script')
    testMask_dir = '' ## Point to the Camelyon test set mask location

    parser.add_argument('--name', default='abc', type=str)
    parser.add_argument('--project', default='mil', type=str)
    parser.add_argument('--dataset_root', default='/data/xxx/TransMIL', type=str, help='dataset root path')
    parser.add_argument('--EPOCH', default=200, type=int)
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--epoch_step', default='[100]', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--isPar', default=False, type=bool)
    parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
    parser.add_argument('--train_show_freq', default=40, type=int)
    parser.add_argument('--droprate', default='0', type=float)
    parser.add_argument('--droprate_2', default='0', type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--sche', default='step', type=str)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--batch_size_v', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_cls', default=2, type=int)
    parser.add_argument('--dataset', default='camelyon16', type=str)
    parser.add_argument('--mDATA0_dir_train0', default='', type=str)  ## Train Set
    parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
    parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
    parser.add_argument('--numGroup', default=4, type=int)
    parser.add_argument('--total_instance', default=4, type=int)
    parser.add_argument('--numGroup_test', default=4, type=int)
    parser.add_argument('--total_instance_test', default=4, type=int)
    parser.add_argument('--mDim', default=512, type=int)
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--isSaveModel', action='store_false')
    parser.add_argument('--debug_DATA_dir', default='', type=str)
    parser.add_argument('--numLayer_Res', default=0, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--num_MeanInference', default=1, type=int)
    parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
    parser.add_argument('--val_ratio', default=0., type=float,)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--weighted', action='store_true')
    parser.add_argument('--save_best_model_stage', default=0., type=float)
    parser.add_argument('--fix_loader_random', action='store_true')
    parser.add_argument('--fix_train_random', action='store_true')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--val_label_balance', action='store_true')
    parser.add_argument('--patch_shuffle', action='store_true')
    parser.add_argument('--act', default='relu', type=str)  
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--init_type', default='none', type=str, help='[none,fc,all]')
    parser.add_argument('--initfc', default='./debug_log', type=str)
    parser.add_argument('--always_test', action='store_true')
    parser.add_argument('--no_log', action='store_true')

    params = parser.parse_args()

    main(params=params)

