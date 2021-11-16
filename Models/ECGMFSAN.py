import torch.nn as nn

from Losses.lmmd_ml import LMMD_loss
from Losses.my_focal_loss import my_focal_loss
from Models.classification_model import ADDneck,AFNET_share



class ECGMFSAN(nn.Module):

    def __init__(self, num_classes=24):
        super(ECGMFSAN, self).__init__()
        self.sharedNet = AFNET_share(inc=12)
        self.sonnet1 = ADDneck(512, 256)
        self.sonnet2 = ADDneck(512, 256)
        self.sonnet3 = ADDneck(512, 256)
        self.sonnet4 = ADDneck(512 ,256)
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
        self.disc_loss =nn.MSELoss()
        self.lmmd =LMMD_loss(class_num=num_classes)

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

            data =self.sonnet4(data_src)
            data =self.avgpool(data)
            data =data.view(data.size(0) ,-1)
            pred_src_son4 =self.cls_fc_son4(data)
            cls_loss4 =self.cls_loss(pred_src_son4 ,label_src)

            if mark == 1:
                data_src = self.sonnet1(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss = self.lmmd.get_loss(data_src, data_tgt_son1 ,label_src ,pred_tgt_son1)



                l1_loss =self.disc_loss(data_tgt_son1 ,data_tgt_son2)
                l1_loss +=self.disc_loss(data_tgt_son1 ,data_tgt_son3)
                pred_src = self.cls_fc_son1(data_src)


                cls_loss = self.cls_loss(pred_src, label_src).sum()

                return cls_loss, mmd_loss, l1_loss / 2 ,cls_loss4

            if mark == 2:
                data_src = self.sonnet2(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss = self.lmmd.get_loss(data_src, data_tgt_son2 ,label_src ,pred_tgt_son2)


                l1_loss = self.disc_loss(data_tgt_son2, data_tgt_son1)
                l1_loss += self.disc_loss(data_tgt_son2, data_tgt_son3)
                pred_src = self.cls_fc_son2(data_src)


                cls_loss = self.cls_loss(pred_src, label_src)

                return cls_loss, mmd_loss, l1_loss / 2 ,cls_loss4

            if mark == 3:
                data_src = self.sonnet3(data_src)
                data_src = self.avgpool(data_src)
                data_src = data_src.view(data_src.size(0), -1)
                mmd_loss = self.lmmd.get_loss(data_src, data_tgt_son3 ,label_src ,pred_tgt_son3)


                l1_loss = self.disc_loss(data_tgt_son3, data_tgt_son1)
                l1_loss += self.disc_loss(data_tgt_son3, data_tgt_son2)
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = self.cls_loss(pred_src, label_src)

                return cls_loss, mmd_loss, l1_loss / 2 ,cls_loss4

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

            return pred1, pred2, pred3 ,pred4