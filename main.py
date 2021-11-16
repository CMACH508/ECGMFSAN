import torch
import argparse
import os
from Models.ECGMFSAN import ECGMFSAN
from train import train
from torch.utils.tensorboard import SummaryWriter
from Data_preprocessing.get_Iters_and_weight import get_iters_and_weight

parser=argparse.ArgumentParser(description='setteing for ECGMSFAN')
parser.add_argument('--batchsize',type=int ,default=64,help='input batch size for model training')
parser.add_argument('--steps',type=int,default=15000,help='number of steps to train (default: 15000)')
parser.add_argument('--source1_dir',type=str,default="../challenge2020/dataset/CPSC/",help='path for data source 1')
parser.add_argument('--source2_dir',type=str,default="../challenge2020/dataset/CPSC/",help='path for data source 2')
parser.add_argument('--source3_dir',type=str,default="../challenge2020/dataset/CPSC/",help='path for data source 3')
parser.add_argument('--target_dir',type=str,default="../challenge2020/dataset/CPSC/",help='path for target')
parser.add_argument('--log_dir',type=str,default="./log",help="path for log and saved model (default: ./log/)")
parser.add_argument('--lr',type=float,default=3e-4,help='learning rate')
parser.add_argument('--cuda',type=bool,default=True,help='use cuda device for model training (default: true)')
parser.add_argument('--test_interval',type=int,default=50,help='interval for test (default: 50)')
parser.add_argument('--norm',type=bool,default=False,help='normalization (default: true)')
args=parser.parse_args()


if __name__ == '__main__':
    print(args)
    if not(os.path.exists(args.log_dir)):
        os.makedirs(args.log_dir)
    source1_loader,source2_loader,source3_loader,target_loader,weight=get_iters_and_weight(source1_dir=args.source1_dir,
                                                                                           source2_dir=args.source2_dir,
                                                                                           source3_dir=args.source3_dir,
                                                                                           target_dir=args.target_dir,
                                                                                           batch_size=args.batchsize,
                                                                                           norm=args.norm,
                                                                                           cuda=args.cuda)

    writer=SummaryWriter(args.log_dir)
    model = ECGMFSAN(num_classes=24)
    if args.cuda:
        model=model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=3e-4,
                                 eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=5,
                                                           eta_min=2e-9,
                                                           last_epoch=-1)
    train(model,
          source1_loader,
          source2_loader,
          source3_loader,
          target_loader,
          weight,
          args.log_dir,
          optimizer=optimizer,
          scheduler=scheduler,
          epochs=args.steps,
          cuda=args.cuda,
          test_interval=args.test_interval,
          writer=writer)