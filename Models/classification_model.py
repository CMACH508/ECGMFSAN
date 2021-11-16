import torch
import torch.nn as nn
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
class BasicBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(BasicBlock,self).__init__()
        self.conv=nn.Sequential(nn.Conv1d(in_channels=inchannel,out_channels=outchannel,kernel_size=7,stride=1,padding=3),
                                #nn.InstanceNorm1d(outchannel),
                                nn.BatchNorm1d(outchannel,affine=True,track_running_stats=True),
                                nn.LeakyReLU(0.2,inplace=True),

                                nn.Conv1d(in_channels=outchannel, out_channels=outchannel, kernel_size=7, stride=1,
                                          padding=3),
                                #nn.BatchNorm1d(outchannel,momentum=0.9,affine=True,track_running_stats=True),
                                nn.LeakyReLU(0.2,inplace=True),
                                nn.Conv1d(in_channels=outchannel, out_channels=outchannel, kernel_size=7, stride=1,
                                          padding=3),
                               #nn.BatchNorm1d(outchannel,momentum=0.9,affine=True,track_running_stats=True),

                                nn.LeakyReLU(0.2,inplace=True),
                                )
    def forward(self, x):
        return self.conv(x)
class BottleBlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        self.inc=inchannel
        self.ouc=outchannel
        super(BottleBlock,self).__init__()
        self.basicblock=BasicBlock(inchannel=inchannel,outchannel=outchannel)
        self.skip=nn.Conv1d(in_channels=inchannel,out_channels=outchannel,kernel_size=1)
    def forward(self,x):
        x1=x if self.inc==self.ouc else self.skip(x)
        x2=self.basicblock(x)
        return x1+x2

class CA(nn.Module):
    def __init__(self,inchannel):
        super(CA,self).__init__()
        self.GAP=nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.model=nn.Sequential(
                                 nn.Linear(inchannel,int(inchannel/4)),
                                 nn.LeakyReLU(0.2,inplace=True),
                                 nn.Linear(int(inchannel/4),inchannel),
                                 nn.Sigmoid())
    def forward(self, x):
        x=self.GAP(x)
        x=x.view(x.shape[0],-1)
        x=self.model(x)
        return x
class CABlock(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(CABlock,self).__init__()
        self.inc=inchannel
        self.outc=outchannel
        self.basicblock=BasicBlock(inchannel,outchannel)
        self.ca=CA(outchannel)
        self.skip = nn.Conv1d(in_channels=inchannel, out_channels=outchannel, kernel_size=1)
    def forward(self,x):
        skip_x=x if self.inc==self.outc else self.skip(x)
        x_conv=self.basicblock(x)
        attn=torch.unsqueeze(self.ca(x_conv),2)
        attn=attn.expand_as(x_conv)
        return skip_x+x_conv*attn

class AFNET(nn.Module):
    def __init__(self,num_classes=24):
        super(AFNET,self).__init__()
        self.model=nn.Sequential(nn.Conv1d(in_channels=12,out_channels=64,kernel_size=7,stride=1,padding=3),
                                 nn.LeakyReLU(0.2,inplace=True),
                                 nn.MaxPool1d(2,2),
                                 BottleBlock(64,64),
                                 #nn.Dropout(),
                                 nn.MaxPool1d(2,2),
                                 BottleBlock(64,128),
                                 nn.MaxPool1d(2,2),
                                 CABlock(128,256),
                                 nn.MaxPool1d(2,2),
                                 CABlock(inchannel=256,outchannel=512),
                                 CABlock(inchannel=512,outchannel=512),
                                 #nn.AdaptiveAvgPool1d(1),
                                 #nn.Dropout()
                                 )
        self.addneck=ADDneck(inplanes=512,planes=256)
        self.classifier=nn.Sequential(
            #nn.Linear(512,num_classes),
            nn.Linear(512,num_classes),
            nn.Sigmoid()
                                      )
        self.avgpool=nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        #x=torch.unsqueeze(x,1)
        x=self.model(x)
        #x=self.addneck(x)
        x=self.avgpool(x)
        x=torch.squeeze(x)
        x=self.classifier(x)
        return x

class AFNET_share(nn.Module):
    def __init__(self,inc=12):
        super(AFNET_share,self).__init__()
        self.model=nn.Sequential(nn.Conv1d(in_channels=inc,out_channels=64,kernel_size=7,stride=1,padding=3),
                                 nn.LeakyReLU(0.2,inplace=True),
                                 nn.MaxPool1d(2,2),
                                 BottleBlock(64,64),
                                 #nn.Dropout(),
                                 nn.MaxPool1d(2,2),

                                 )
    def forward(self, x):
        x=self.model(x)
        return x
class ADDneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ADDneck, self).__init__()
        self.model=nn.Sequential(
            BottleBlock(64, 128),
            nn.MaxPool1d(2, 2),
            CABlock(128, 256),
            nn.MaxPool1d(2, 2),
            CABlock(inchannel=256, outchannel=512),
            CABlock(inchannel=512, outchannel=512),

        )

        # self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(planes)
        # self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = nn.BatchNorm1d(planes)
        # self.conv3 = nn.Conv1d(planes, planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm1d(planes)
        # self.relu = nn.ReLU(inplace=True)
        # self.stride = stride

        self.avg_pool=nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x=self.model(x)
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        out=self.avg_pool(x)
        return out

class AFNET_DANN(nn.Module):
    def __init__(self,inc,num_classes=24,alpha=1,domain_classes=6):
        super(AFNET_DANN,self).__init__()
        self.model=nn.Sequential(nn.Conv1d(in_channels=inc,out_channels=64,kernel_size=7,stride=1,padding=3),
                                 nn.LeakyReLU(0.2,inplace=True),
                                 nn.MaxPool1d(2,2),
                                 BottleBlock(64,64),
                                 #nn.Dropout(),
                                 nn.MaxPool1d(2,2),
                                 BottleBlock(64,128),
                                 nn.MaxPool1d(2,2),
                                 CABlock(128,256),
                                 nn.MaxPool1d(2,2),
                                 CABlock(inchannel=256,outchannel=512),
                                 CABlock(inchannel=512,outchannel=512),
                                 nn.AdaptiveAvgPool1d(1),
                                 #nn.Dropout()
                                 )
        self.classifier = nn.Sequential(nn.Conv1d(512, num_classes,1,1),
                                        nn.Sigmoid()
                                        )
        self.domain_classifier = nn.Sequential(  # nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(512, 128, 1, 1),
            nn.ReLU(),
            nn.Conv1d(128, domain_classes, 1, 1),
            #nn.Softmax(dim=1)
        )


    def forward(self, x, alpha):
        # x=torch.unsqueeze(x,1)
        feature = self.model(x)
        label_output = self.classifier(feature)
        reversed_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reversed_feature)

        label_output = label_output.view(x.shape[0], -1)
        domain_output = domain_output.view(x.shape[0], -1)
        return label_output, domain_output
def test():
    data=torch.randn((16,12,5000))
    net=AFNET()
    true=torch.ones((16,111))*0.5
    #net=CABlock(16,64)
    res=net(data)
    loss=nn.BCELoss()
    print(data.shape)
    print(res.shape)
    loss1=loss(res,true)
    loss1.backward()
    print(loss1)

if __name__ == '__main__':
    test()
