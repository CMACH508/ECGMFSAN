import sys
sys.path.append('../')
import argparse
import torch
import torch.nn  as nn
from Models.classification_model import AFNET_share, ADDneck
from torch.autograd import Variable
import math
from Losses.lmmd_ml import LMMD_loss
import os
from Data_preprocessing.Data_preprocessing import CreateDatasetIters,cal_cls_fusion_factor
from Losses.my_focal_loss import my_focal_loss
from Evaluation.evaluation import compute_auc,compute_f_measure,get_challenge_score
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='setting for MFSAN')
parser.add_argument('--batchsize', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--iter', type=int, default=15000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=8, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="/data/zhuyc/OfficeHome/",
                    help='the path to load the data')
parser.add_argument('--source1_dir', type=str, default="Art",  # Art  Clipart   Product   Real World
                    help='the name of the source dir')
parser.add_argument('--source2_dir', type=str, default="Clipart",
                    help='the name of the source dir')
parser.add_argument('--source3_dir', type=str, default="Real World",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",
                    help='the name of the test dir')
parser.add_argument('--log_dir', type=str, default='./log_27_21K_W/CEGP_C_MFSAN_6class_4e_2_batch_64_dim512_focal_weighted_mean_mlmmd_disc_4sonnet/',
                    help='the path of log file')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
'''
*********************************************************************************************

source and target config

*********************************************************************************************
'''
'''创建源域和目标域的迭代器'''
source_dirs = [
    '../dataset/CPSC/',
    '../dataset/CPSC_E/',
    '../dataset/Georgia/',
    '../dataset/PTB-XL/',
]
source1_dir = source_dirs[1]
source2_dir = source_dirs[2]
source3_dir = source_dirs[3]
target_dir = source_dirs[0]
print('*** dataset setting***')
print('source 1 : ', source1_dir.split('/')[-2])
print('source 2 : ', source2_dir.split('/')[-2])
print('source 3 : ', source3_dir.split('/')[-2])
print('target   : ', target_dir.split('/')[-2])
print()
args.norm = True
args.require_val = False
if args.require_val:
    source1_loader, source1_val = CreateDatasetIters(source1_dir, requried_Val_iter=True, norm=args.norm,
                                                     batch_size=args.batchsize)
    source2_loader, source2_val = CreateDatasetIters(source2_dir, requried_Val_iter=True, norm=args.norm,
                                                     batch_size=args.batchsize)
    source3_loader, source3_val = CreateDatasetIters(source3_dir, requried_Val_iter=True, norm=args.norm,
                                                     batch_size=args.batchsize)
else:
    source1_loader = CreateDatasetIters(source1_dir, requried_Val_iter=False, norm=args.norm,
                                        batch_size=args.batchsize)
    source2_loader = CreateDatasetIters(source2_dir, requried_Val_iter=False, norm=args.norm,
                                        batch_size=args.batchsize)
    source3_loader = CreateDatasetIters(source3_dir, requried_Val_iter=False, norm=args.norm,
                                        batch_size=args.batchsize)

target_loader = CreateDatasetIters(target_dir, norm=args.norm, batch_size=args.batchsize)
weight=cal_cls_fusion_factor([source1_loader,source2_loader,source3_loader])
weight=weight.cuda()

writer = SummaryWriter(args.log_dir)
test_interval = 50


class MFSAN(nn.Module):

    def __init__(self, num_classes=24):
        super(MFSAN, self).__init__()
        self.sharedNet = AFNET_share(inc=12)
        self.sonnet1 = ADDneck(512, 256)
        self.sonnet2 = ADDneck(512, 256)
        self.sonnet3 = ADDneck(512, 256)
        self.sonnet4 = ADDneck(512,256)
        self.cls_fc_son1 = nn.Sequential(nn.Linear(512, num_classes),
                                         nn.Sigmoid())
        self.cls_fc_son2 = nn.Sequential(nn.Linear(512, num_classes),
                                         nn.Sigmoid())
        self.cls_fc_son3 = nn.Sequential(nn.Linear(512, num_classes),
                                         nn.Sigmoid())
        self.cls_fc_son4 = nn.Sequential(nn.Linear(512, num_classes),
                                         nn.Sigmoid())
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.cls_loss = my_focal_loss()
        self.disc_loss=nn.MSELoss()
        self.lmmd=LMMD_loss(class_num=num_classes)

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):

        if self.training == True:

            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            data_tgt_son1 = self.sonnet1(data_tgt)

            data_tgt_son1 = data_tgt_son1.view(data_tgt_son1.size(0), -1)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)

            data_tgt_son2 = self.sonnet2(data_tgt)
            data_tgt_son2 = self.avgpool(data_tgt_son2)
            data_tgt_son2 = data_tgt_son2.view(data_tgt_son2.size(0), -1)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)

            data_tgt_son3 = self.sonnet3(data_tgt)
            data_tgt_son3 = self.avgpool(data_tgt_son3)
            data_tgt_son3 = data_tgt_son3.view(data_tgt_son3.size(0), -1)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)

            data=self.sonnet4(data_src)
            data=self.avgpool(data)
            data=data.view(data.size(0),-1)
            pred_src_son4=self.cls_fc_son4(data)
            cls_loss4=self.cls_loss(pred_src_son4,label_src)

            if mark == 1:
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss = self.lmmd.get_loss(data_src, data_tgt_son1,label_src,pred_tgt_son1)



                l1_loss=self.disc_loss(data_tgt_son1,data_tgt_son2)
                l1_loss+=self.disc_loss(data_tgt_son1,data_tgt_son3)
                pred_src = self.cls_fc_son1(data_src)


                cls_loss = self.cls_loss(pred_src, label_src).sum()

                return cls_loss, mmd_loss, l1_loss / 2,cls_loss4

            if mark == 2:
                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss = self.lmmd.get_loss(data_src, data_tgt_son2,label_src,pred_tgt_son2)


                l1_loss = self.disc_loss(data_tgt_son2, data_tgt_son1)
                l1_loss += self.disc_loss(data_tgt_son2, data_tgt_son3)
                pred_src = self.cls_fc_son2(data_src)


                cls_loss = self.cls_loss(pred_src, label_src)

                return cls_loss, mmd_loss, l1_loss / 2,cls_loss4

            if mark == 3:
                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss = self.lmmd.get_loss(data_src, data_tgt_son3,label_src,pred_tgt_son3)


                l1_loss = self.disc_loss(data_tgt_son3, data_tgt_son1)
                l1_loss += self.disc_loss(data_tgt_son3, data_tgt_son2)
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = self.cls_loss(pred_src, label_src)

                return cls_loss, mmd_loss, l1_loss / 2,cls_loss4

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data)
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data)
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.cls_fc_son3(fea_son3)


            fea_son4 = self.sonnet4(data)
            fea_son4 = self.avgpool(fea_son4)
            fea_son4 = fea_son4.view(fea_son4.size(0), -1)
            pred4 = self.cls_fc_son4(fea_son4)



            return pred1, pred2, pred3,pred4

def calMean(pred1,pred2,pred3,pred4,alpha=0.40):
    return (1-alpha)*(weight[0][:]*pred1+weight[1][:]*pred2+weight[2][:]*pred3)+alpha*pred4
def train(model, optimizer=None, scheduler=None):
    # 最后的全连接层学习率为前面的10倍
    best_challengescore=0.0
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_loader)


    correct = 0
    scale_factor = 1e-3
    total_cls_loss1 = 0
    total_cls_loss2 = 0
    total_cls_loss3 = 0

    total_mmd_loss1 = 0
    total_mmd_loss2 = 0
    total_mmd_loss3 = 0

    total_l1_loss1 = 0
    total_l1_loss2 = 0
    total_l1_loss3 = 0

    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0

    for i in range(1, args.iter + 1):
        model.train()


        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        # source_data, source_label = Variable(source_data), Variable(source_label)
        # target_data = Variable(target_data)
        optimizer.zero_grad()

        # cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 1)
        cls_loss, mmd_loss, l1_loss,cls_loss4 = model(source_data, target_data, source_label, 1)

        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        gamma *= scale_factor
        loss = cls_loss4+cls_loss + gamma * (mmd_loss + l1_loss)
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)

        total_cls_loss1 += cls_loss.item()
        total_mmd_loss1 += mmd_loss.item()
        total_l1_loss1 += l1_loss.item()
        total_loss1 += loss.item()
        loss.backward()
        optimizer.step()

        if i % test_interval == 0:

            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / args.iter, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
            writer.add_scalar(tag='loss1', scalar_value=total_loss1)
            writer.add_scalar(tag='cls_loss1', scalar_value=total_cls_loss1)
            writer.add_scalar(tag='mmd_loss1', scalar_value=total_mmd_loss1)
            writer.add_scalar(tag='l1_loss1', scalar_value=total_l1_loss1)
            total_loss1 = 0
            total_cls_loss1 = 0
            total_mmd_loss1 = 0
            total_l1_loss1 = 0

        # if i % 3 == 2:
        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()

        optimizer.zero_grad()


        cls_loss, mmd_loss, l1_loss,cls_loss4 = model(source_data, target_data, source_label, 2)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        gamma *= scale_factor
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss = cls_loss4+cls_loss + gamma * (mmd_loss  + l1_loss)

        total_cls_loss2 += cls_loss.item()
        total_mmd_loss2 += mmd_loss.item()
        total_l1_loss2 += l1_loss.item()
        total_loss2 += loss.item()

        loss.backward()
        optimizer.step()

        if i % test_interval == 0:
            # print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
            #     i, 100. * i / args.iter, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / args.iter, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
            writer.add_scalar(tag='loss2', scalar_value=total_loss2)
            writer.add_scalar(tag='cls_loss2', scalar_value=total_cls_loss2)
            writer.add_scalar(tag='mmd_loss2', scalar_value=total_mmd_loss2)
            writer.add_scalar(tag='l1_loss2', scalar_value=total_l1_loss2)
            total_loss2 = 0
            total_cls_loss2 = 0
            total_mmd_loss2 = 0
            total_l1_loss2 = 0

        # source3
        try:
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, __ = target_iter.next()
        if args.cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        optimizer.zero_grad()

        # cls_loss, mmd_loss, l1_loss = model(source_data, target_data, source_label, source_label, 3)
        cls_loss, mmd_loss, l1_loss,cls_loss4 = model(source_data, target_data, source_label, 3)
        gamma = 2 / (1 + math.exp(-10 * (i) / (args.iter))) - 1
        gamma *= scale_factor
        loss =cls_loss4+ cls_loss + gamma * (mmd_loss + l1_loss)
        # loss = cls_loss + gamma * (mmd_loss + l1_loss)
        total_cls_loss3 += cls_loss.item()
        total_mmd_loss3 += mmd_loss.item()
        total_l1_loss3 += l1_loss.item()
        total_loss3 += loss.item()
        loss.backward()
        optimizer.step()

        if i % test_interval == 0:
            # print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
            #     i, 100. * i / args.iter, loss.data[0], cls_loss.data[0], mmd_loss.data[0], l1_loss.data[0]))
            print(
                'Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / args.iter, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))
            writer.add_scalar(tag='loss3', scalar_value=total_loss3)
            writer.add_scalar(tag='cls_loss3', scalar_value=total_cls_loss3)
            writer.add_scalar(tag='mmd_loss3', scalar_value=total_mmd_loss3)
            writer.add_scalar(tag='l1_loss3', scalar_value=total_l1_loss3)
            total_loss3 = 0
            total_cls_loss3 = 0
            total_mmd_loss3 = 0
            total_l1_loss3 = 0


        if i % (test_interval) == 0:
            model.eval()
            # if i % ( 1) == 0:
            challenge_score=test(model)
            if challenge_score>best_challengescore:
                best_challengescore = challenge_score
                torch.save(model,args.log_dir+'best.pth')
            model.train()
            if scheduler is not None:
                scheduler.step()

@torch.no_grad()
def test(model):
    # model.eval()
    print('True')


    thres1, thres2, thres3,thres4 = 0.5, 0.5, 0.5,0.5
    for i, (data, target) in enumerate(target_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        pred1, pred2, pred3,pred4 = model(data)

        if i == 0:
            All_pred1 = pred1
            All_pred2 = pred2
            All_pred3 = pred3
            All_pred4 = pred4

            #All_pred_mean = (pred1 + pred2 + pred3+pred4) / 4
            All_pred_mean = calMean(pred1 , pred2 , pred3,pred4)



            All_label = target
        else:
            All_pred1 = torch.cat([All_pred1, pred1], dim=0)
            All_pred2 = torch.cat([All_pred2, pred2], dim=0)
            All_pred3 = torch.cat([All_pred3, pred3], dim=0)
            All_pred4 = torch.cat([All_pred4, pred4], dim=0)
            #All_pred_mean = torch.cat([All_pred_mean, (pred1 + pred2 + pred3+pred4) / 4], dim=0)
            All_pred_mean = torch.cat([All_pred_mean, calMean(pred1 , pred2 , pred3,pred4)], dim=0)
            All_label = torch.cat([All_label, target], dim=0)

    binary_outputs1 = (All_pred1.detach().cpu().numpy() >= thres1).astype('int')
    f1_score_1 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs1)
    print('f1 score by source 1:', f1_score_1.mean())
    writer.add_scalar(tag='F1Score_1', scalar_value=f1_score_1)
    score_1 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred1.detach().cpu().numpy())
    print('score by source 1:', score_1)
    writer.add_scalar(tag='score_1',scalar_value=score_1)

    del All_pred1

    binary_outputs2 = (All_pred2.detach().cpu().numpy() >= thres2).astype('int')
    f1_score_2 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs2)
    print('f1 score by source 2:', f1_score_2.mean())
    writer.add_scalar(tag='F1Score_2', scalar_value=f1_score_2)

    score_2 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred2.detach().cpu().numpy())
    print('score by source 2:', score_2)
    writer.add_scalar(tag='score_2', scalar_value=score_2)
    del All_pred2

    binary_outputs3 = (All_pred3.detach().cpu().numpy() >= thres3).astype('int')
    f1_score_3 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs3)
    writer.add_scalar(tag='F1Score_3', scalar_value=f1_score_3)
    print('f1 score by source 3:', f1_score_3.mean())
    score_3 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred3.detach().cpu().numpy())
    print('score by source 3:', score_3)
    writer.add_scalar(tag='score_3', scalar_value=score_3)
    del All_pred3

    binary_outputs4 = (All_pred4.detach().cpu().numpy() >= thres4).astype('int')
    f1_score_4 = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs4)
    writer.add_scalar(tag='F1Score_4', scalar_value=f1_score_4)
    print('f1 score by source 4:', f1_score_4.mean())
    score_4 = get_challenge_score(All_label.detach().cpu().numpy(), All_pred4.detach().cpu().numpy())
    print('score by source 4:', score_4)
    writer.add_scalar(tag='score_4', scalar_value=score_4)
    del All_pred4

    #majority voting
    #binary_outputs_mean = ((binary_outputs1 + binary_outputs2 + binary_outputs3+binary_outputs4) >= 2).astype('int')
    # print('sum:', sum(binary_outputs_mean))
    # print(binary_outputs_mean.shape, All_label.detach().cpu().numpy().shape)

    # Mean Mix
    binary_outputs_mean = (All_pred_mean.detach().cpu().numpy() > 0.5)
    f1_score_mean = compute_f_measure(All_label.detach().cpu().numpy(), binary_outputs_mean)
    print('f1 score by MSFAN:', f1_score_mean.mean())
    writer.add_scalar(tag='F1Score_MSFAN', scalar_value=f1_score_mean)
    score_MFSAN = get_challenge_score(All_label.detach().cpu().numpy(), All_pred_mean.detach().cpu().numpy())
    print('score by source _mean:', score_MFSAN)
    writer.add_scalar(tag='score_MFSAN', scalar_value=score_MFSAN)
    print('')
    del All_pred_mean

    del All_label
    return score_MFSAN




if __name__ == '__main__':
    model = MFSAN(num_classes=24)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-9)
    T_max = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=2e-9, last_epoch=-1)
    model.cuda()
    train(model, optimizer, scheduler)
