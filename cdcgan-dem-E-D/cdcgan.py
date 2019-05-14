from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt
#from model.Discriminator import Discriminator
#from model.Generator import Generator

##############################Constructor##############
class Generator(nn.Module):
    def __init__(self, nc, ngf):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nc,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16 x 64
        self.layer2 = nn.Sequential(nn.Conv2d(ngf,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 8 x 8 x 128
        
        self.layer3 = nn.Sequential(nn.Conv2d(ngf*2,ngf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 4 x 4 x 256                     
        # 4 x 4 x 256
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf*4,ngf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf*2),
                                 nn.ReLU())
        # 8 x 8 x 128
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(ngf*2,ngf,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ngf),
                                 nn.ReLU())
        # 16 x 16 x 64
        self.layer6 = nn.Sequential(nn.ConvTranspose2d(ngf,nc,kernel_size=4,stride=2,padding=1),
                                 nn.Tanh())
        # 32 x 32 x 1
    def forward(self,_cpLayer):
        out = self.layer1(_cpLayer)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out
class Discriminator(nn.Module):
    def __init__(self,nc,ndf):
        super(Discriminator,self).__init__()
        self.layer1_image = nn.Sequential(nn.Conv2d(nc,ndf/2,kernel_size=4,stride=2,padding=1),
                                 #nn.BatchNorm2d(ndf/2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16
        self.layer1_cp = nn.Sequential(nn.Conv2d(nc,ndf/2,kernel_size=4,stride=2,padding=1),
                                 #nn.BatchNorm2d(ndf/2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 16 x 16
        self.layer2 = nn.Sequential(nn.Conv2d(ndf,ndf*2,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*2),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 8 x 8
        
        self.layer3 = nn.Sequential(nn.Conv2d(ndf*2,ndf*4,kernel_size=4,stride=2,padding=1),
                                 nn.BatchNorm2d(ndf*4),
                                 nn.LeakyReLU(0.2,inplace=True))
        # 4 x 4
        
        self.layer4 = nn.Sequential(nn.Conv2d(ndf*4,1,kernel_size=4,stride=1,padding=0),
                                 nn.Sigmoid())
        # 1
        
    def forward(self,dem,_cpLayer):
        
        out_1 = self.layer1_image(dem)
        out_2 = self.layer1_cp(_cpLayer)        
        out = self.layer2(torch.cat((out_1,out_2),1))
        out = self.layer3(out)
        out = self.layer4(out)
        return out


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nthread', type=int,default=1, help="number of workers/subprocess")
parser.add_argument('--ncp', type=int, default=100, help='size of the controlpoints')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nk', type=int, default=1, help='k times D for one G')
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataset', default='DEM', help='which dataset to train on, DEM')
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--npre', type=int,default=0, help="pre-training epoch times")
parser.add_argument('--logfile', default='errlog.txt', help="pre-training epoch times")

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True



###############   DATASET   ##################
if(opt.dataset=='DEM'):
    dataset = dset.ImageFolder(root='../../data/DEM-300m-normed/',transform=transforms.Compose([     
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                               ]))

loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.batchSize,
                                     shuffle = True,drop_last=True,num_workers=opt.nthread)

###############   MODEL and initialization  ####################
ndf = opt.ndf
ngf = opt.ngf
ncp=opt.ncp
nc = 1
#The Lowest and Highest elevation for dem re-normalization
L_ele=-7
H_ele=6999

netD = Discriminator(nc, ndf)
netG = Generator(nc, ngf)
if (opt.netG!='' and opt.netD!=''):
    netG.load_state_dict(torch.load(opt.netG))
    netD.load_state_dict(torch.load(opt.netD))
if(opt.cuda):
    netD.cuda()
    netG.cuda()
    
###############   self-defined FUNCTION   ####################
def ControlPointsImage(dems,ncp):# Uniform sampling
    y_cpLayer = torch.FloatTensor(opt.batchSize,nc,opt.imageSize,opt.imageSize).zero_()
    y_cpLayer = Variable(y_cpLayer)
    if(opt.cuda):
        y_cpLayer = y_cpLayer.cuda()
    cp=[]
    x_index=[]
    y_index=[]
    step=float((opt.imageSize-1)/(np.sqrt(ncp)-1))
    #print step
    for i in range(0,int(np.floor(np.sqrt(ncp)))):
        x_index.append(i*step)
        y_index.append(i*step)
    for n in range(0,opt.batchSize):
        cp.append([])
        for i in x_index:
            for j in y_index:
                cp[n].append([dems[n,0,int(round(i)),int(round(j))],int(round(i)),int(round(j))])# extract dem control point function
    for i in range(0,opt.batchSize): 
        for _cp in cp[i]:
            h=_cp[0]
            x=_cp[1]
            y=_cp[2]
            y_cpLayer[i,0,x,y]=h
    return y_cpLayer

def ControlPointsImage_random(dems,ncp):# Random sampling
    y_cpLayer = torch.FloatTensor(opt.batchSize,nc,opt.imageSize,opt.imageSize).zero_()
    y_cpLayer = Variable(y_cpLayer)
    if(opt.cuda):
        y_cpLayer = y_cpLayer.cuda()
    cp=[]
    for n in range(0,opt.batchSize):
        cp.append([])
        for ite in range(0,ncp):
                i=random.randint(0,31)
                j=random.randint(0,31)
                cp[n].append([dems[n,0,i,j],i,j])# extract dem control point function
    for i in range(0,opt.batchSize): 
        for _cp in cp[i]:
            h=_cp[0]
            x=_cp[1]
            y=_cp[2]
            y_cpLayer[i,0,x,y]=h
    return y_cpLayer


###########   LOSS & OPTIMIZER   ##########
criterion = nn.BCELoss(size_average=True)
optimizerD = torch.optim.Adam(netD.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(),lr=opt.lr, betas=(opt.beta1, 0.999))

criterionDEM = nn.MSELoss(size_average=True)

##########   GLOBAL VARIABLES   ###########
cpLayer = torch.FloatTensor(opt.batchSize,nc,opt.imageSize,opt.imageSize).zero_() # all ncp control points in one layer bs*nc*imagesize*imagesize for D
dems = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize) # real data for D
cpLayer = Variable(cpLayer)
dems = Variable(dems)

reallabel = torch.FloatTensor(opt.batchSize)
fakelabel = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
reallabel=Variable(reallabel)
fakelabel=Variable(fakelabel)
reallabel.data.resize_(opt.batchSize).fill_(real_label)
fakelabel.data.resize_(opt.batchSize).fill_(fake_label)

if(opt.cuda):
    cpLayer = cpLayer.cuda()
    dems = dems.cuda()
    reallabel=reallabel.cuda()
    fakelabel=fakelabel.cuda()


    
########### Training   ###########
for epoch in range(opt.npre+1,opt.npre+opt.niter+1):    
    for i, (images,_) in enumerate(loader):
        errlog=open(opt.logfile,'a')
        ########### fDx ###########
        dems.data.copy_(images[:,0,:,:])
        cpLayer=ControlPointsImage(dems,ncp)             
        for k in range(0,opt.nk):
            netD.zero_grad()
            
            # train with real data, resize real because last batch may has less than           
            output = netD(dems,cpLayer)# input real image and cpLayer both bs*nc*imagesize*imagesize
            errD_real = criterion(output,reallabel)
            errD_real.backward()
            
            # train with fake data
            fake = netG(cpLayer)          
            # detach gradients here so that gradients of G won't be updated
            output = netD(fake.detach(),cpLayer) # input fake image and cpLayer both bs*nc*imagesize*imagesize
            errD_fake = criterion(output,fakelabel)
            errD_fake.backward()

            errD = errD_fake + errD_real
            optimizerD.step()

        ########### fGx ###########
        netG.zero_grad()
        output = netD(fake,cpLayer)
        errG = criterion(output,reallabel)
        errG.backward()
        optimizerG.step()
        
        if(i == 0):
        ########### Logging #########
            vutils.save_image(fake.data,'%s/images/epoch_%03d_batch_%03d_fake.png' % (opt.outf, epoch,i),normalize=True)
            vutils.save_image(dems.data,'%s/images/epoch_%03d_batch_%03d_real.png' % (opt.outf, epoch,i),normalize=True)
        if(i % 10 == 0):
        ########### Logging #########
            dems.data.copy_(L_ele+(dems.data/2+0.5)*(H_ele-L_ele))
            fake.data.copy_(L_ele+(fake.data/2+0.5)*(H_ele-L_ele))
            errDem = criterionDEM(fake,dems)
            
            errlog.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f MSE: %.2f\n'
                  % (epoch, opt.npre+opt.niter+1, i, len(loader),
                     errD.data[0], errG.data[0],errDem.data[0]))
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f MSE: %.2f'
                  % (epoch, opt.npre+opt.niter+1, i, len(loader),
                     errD.data[0], errG.data[0],errDem.data[0]))    
        errlog.close()
    torch.save(netG.state_dict(), '%s/nets/netG_epoch_%03d.pth' % (opt.outf,epoch))
    torch.save(netD.state_dict(), '%s/nets/netD_epoch_%03d.pth' % (opt.outf,epoch))