'''
this file is for quick validation of saved model, as long as datasets been placed the right position weith proper name,
run this .py file should quickly show results shown in paper.
'''
import os
import torch
import argparse
from Evaluation.test import test
from Data_preprocessing.get_Iters_and_weight import get_iters_and_weight
from torch.utils.tensorboard import SummaryWriter

parser=argparse.ArgumentParser(description='setteing for ECGMSFAN')
parser.add_argument('--batchsize',type=int ,default=64,help='input batch size for model training')
parser.add_argument('--norm',type=bool,default=False,help='normalization (default: true)')
parser.add_argument('--cuda',type=bool,default=True,help='use cuda device for model training (default: true)')
parser.add_argument('--log_dir',type=str,default="./log/quick_validation",help='path for quick validation log')
parser.add_argument('--CPSC_dir',type=str,default="./datasets/CPSC/",help='path for CPSC dataset')
parser.add_argument('--CPSC_E_dir',type=str,default="./datasets/CPSC_E/",help='path for CPSC_E dataset')
parser.add_argument('--G_dir',type=str,default="./datasets/Georgia/",help='path for Georgia dataset')
parser.add_argument('--PTB_XL_dir',type=str,default="./datasets/PTB-XL/",help='path for PTB-XL dataset')
parser.add_argument('--PTB_XL_10k_dir',type=str,default="./datasets/PTB-XL_10k/",help='path for PTB-XL_10k dataset')
parser.add_argument('--CCEG_P',type=str,default="./SavedModels/CCEG_P.pth",help='path for CCEG_P model compared with Hasani')
parser.add_argument('--CCEP_G',type=str,default="./SavedModels/CCEP_G.pth",help='path for CCEP_G model compared with Hasani')
parser.add_argument('--CEGP_C',type=str,default="./SavedModels/CEGP_C.pth",help='path for CECG_C model compared with Hasani')
parser.add_argument('--CGP_CE',type=str,default="./SavedModels/CGP_CE.pth",help='path for CGP_CE model compared with Hasani')
parser.add_argument('--CCEG_P_U',type=str,default="./SavedModels/CCEG_P_U.pth",help='path for CCEG_P model compared with Uguz')
parser.add_argument('--CCEP_G_U',type=str,default="./SavedModels/CCEP_G_U.pth",help='path for CCEP_G model compared with Uguz')
parser.add_argument('--CEGP_C_U',type=str,default="./SavedModels/CEGP_C_U.pth",help='path for CECG_C model compared with Uguz')
parser.add_argument('--CGP_CE_U',type=str,default="./SavedModels/CGP_CE_U,pth",help='path for CGP_CE model compared with Uguz')
args=parser.parse_args()

def quickEvaluation(source1_dir,source2_dir,source3_dir,target_dir,model_path):
    source1_loader, source2_loader, source3_loader, target_loader, weight = get_iters_and_weight(
        source1_dir=source1_dir,
        source2_dir=source2_dir,
        source3_dir=source3_dir,
        target_dir=target_dir,
        batch_size=args.batchsize,
        norm=args.norm,
        cuda=args.cuda)
    model=torch.load(model_path)
    test(model.cuda(),target_loader=target_loader,weight=weight,cuda=args.cuda,writer=None)



if __name__ == '__main__':
    print(args)
    if not(os.path.exists(args.log_dir)):
        os.makedirs(args.log_dir)
    writer = SummaryWriter(args.log_dir)

    print("***************************************************************************************************")
    print("Comparison wtih Hasani")
    print("C,CE,G->P")
    quickEvaluation(source1_dir=args.CPSC_dir,source2_dir=args.CPSC_E_dir,source3_dir=args.G_dir,target_dir=args.PTB_XL_10k_dir,model_path=args.CCEG_P)
    print("C,CE,P->G")
    quickEvaluation(source1_dir=args.CPSC_dir, source2_dir=args.CPSC_E_dir, source3_dir=args.PTB_XL_10k_dir,
                    target_dir=args.G_dir, model_path=args.CCEP_G)
    print("CE,G,P->C")
    quickEvaluation(source1_dir=args.CPSC_E_dir, source2_dir=args.G_dir, source3_dir=args.PTB_XL_10k_dir,
                    target_dir=args.CPSC_dir, model_path=args.CEGP_C)
    print("C,G,P->CE")
    quickEvaluation(source1_dir=args.CPSC_dir, source2_dir=args.G_dir , source3_dir=args.PTB_XL_10k_dir,
                    target_dir=args.CPSC_E_dir, model_path=args.CGP_CE)

    print("Comparison wtih Uguz")
    print("C,CE,G->P")
    quickEvaluation(source1_dir=args.CPSC_dir, source2_dir=args.CPSC_E_dir, source3_dir=args.G_dir,
                    target_dir=args.PTB_XL_dir, model_path=args.CCEG_P_U)
    print("C,CE,P->G")
    quickEvaluation(source1_dir=args.CPSC_dir, source2_dir=args.CPSC_E_dir, source3_dir=args.PTB_XL_dir,
                    target_dir=args.G_dir, model_path=args.CCEP_G_U)
    print("CE,G,P->C")
    quickEvaluation(source1_dir=args.CPSC_E_dir, source2_dir=args.G_dir, source3_dir=args.PTB_XL_dir,
                    target_dir=args.CPSC_dir, model_path=args.CEGP_C_U)
    print("C,G,P->CE")
    quickEvaluation(source1_dir=args.CPSC_dir, source2_dir=args.G_dir, source3_dir=args.PTB_XL_dir,
                    target_dir=args.CPSC_E_dir, model_path=args.CGP_CE_U)
    print("***************************************************************************************************")
    print("All test done!")

