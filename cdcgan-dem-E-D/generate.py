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


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nthread', type=int,default=1, help="number of workers/subprocess")
parser.add_argument('--ncp', type=int, default=100, help='size of the controlpoints')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default='outfile_generate_loss', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--dataset', default='DEM', help='which dataset to train on, DEM')
parser.add_argument('--netG', default='outfile', help="path to netG (to continue training)")
parser.add_argument('--logfile', default='outfile_generate_loss/100samples/errlog.txt', help="logfile to record error")

#intepretate noise and read /print noise
L_ele=-7
H_ele=6999

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
###############   Training DATASET   ##################
if(opt.dataset=='DEM'):
    dataset = dset.ImageFolder(root='../data/DEM-300m-normed-test/',transform=transforms.Compose([     
                                   transforms.Scale(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                               ]))

loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = opt.batchSize,
                                     shuffle = True ,drop_last=True,num_workers=opt.nthread)

criterion = nn.MSELoss(size_average=True)

###############   self-defined FUNCTION   ####################
def ControlPointsImage(dems,ncp):
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
def KrigeDems(dems,ncp):
    cpDems=[]
    krigeDemsTensor =torch.FloatTensor(opt.batchSize,nc,opt.imageSize,opt.imageSize).zero_()# all ncp control points in one layer
    krigeDemsTensor =Variable(krigeDemsTensor)
    if(opt.cuda):
        krigeDemsTensor = krigeDemsTensor.cuda()
    x_index=[]
    y_index=[]
    step=float((opt.imageSize-1)/(np.sqrt(ncp)-1))   
    #print step,pad
    for i in range(0,int(np.floor(np.sqrt(ncp)))):
        x_index.append(i*step)
        y_index.append(i*step)
    for n in range(0,opt.batchSize):
        cpDems.append([])
        for i in x_index:
            for j in y_index:
                cpDems[n].append([int(round(j)),int(round(i)),dems.data[n,0,int(round(j)),int(round(i))]])# extract dem control point function
        cpDems[n]=np.array(cpDems[n])
        
    gridx = np.arange(0.0, float(opt.imageSize), 1)
    gridy = np.arange(0.0, float(opt.imageSize), 1)
    for n in range(0,opt.batchSize):
        OK = OrdinaryKriging(cpDems[n][:, 0], cpDems[n][:, 1], cpDems[n][:, 2], variogram_model='spherical',
                         verbose=False, enable_plotting=False, nlags=100)
        z,ss=OK.execute('grid',gridx,gridy,backend='loop')
        for i in gridx:
            for j in gridy:
                    krigeDemsTensor[n,0,i,j]=z[int(j)][int(i)]# different transpose for the z and dems
        #kt.write_asc_grid(gridx, gridy, z, filename="output"+str(n)+".asc")
    return krigeDemsTensor


ncp=opt.ncp 
nc = 1
###########   GLOBAL VARIABLES   #########
dems = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize) # real data
dems = Variable(dems)

mdems = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize) # real data with meter scale
mdems = Variable(mdems)

cpLayer = torch.FloatTensor(opt.batchSize,nc,opt.imageSize,opt.imageSize).zero_() # all ncp control points in one layer bs*nc*imagesize*imagesize for G
cpLayer = Variable(cpLayer)
#krigeDems = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize) # real data
#krigeDems = Variable(krigeDems)
if(opt.cuda):
    dems = dems.cuda()
    mdems = mdems.cuda()
    cpLayer = cpLayer.cuda()
    #krigeDems = krigeDems.cuda()
    
    
###########   Generate   ###########  
dataiter = iter(loader) # generate one batch fake images for a run
images, labels = dataiter.next()
dems.data.copy_(images[:,0,:,:])
cpLayer=ControlPointsImage(dems,ncp)

###########   Load netG   ###########
vutils.save_image(dems.data,'%s/gen_real.png' % (opt.outf),normalize=True)
test=['001','010','020','030','040','050','060','070','080','090','100','110','120','130','140','150','160','170','180','190','200']
_dir=opt.netG
for epoch in test:
    opt.epoch=int(epoch)
    opt.netG=_dir+'/nets/netG_epoch_'+epoch+'.pth'
    assert opt.netG != '', "netG must be provided!"
    
    netG = Generator(nc, opt.ngf)
    netG.load_state_dict(torch.load(opt.netG))
    if(opt.cuda):
        netG.cuda()

    fake = netG(cpLayer)
    mdems.data.copy_(dems.data)

    #mdems.data.copy_(L_ele+(dems.data/2+0.5)*(H_ele-L_ele))
    #krigeDems=KrigeDems(mdems,ncp)



    vutils.save_image(fake.data,'%s/gen_epoch_%03d_fake.png' % (opt.outf,opt.epoch),normalize=True)
    #vutils.save_image(dems.data,'%s/gen_epoch_%03d_real.png' % (opt.outf,opt.epoch),normalize=True)
    
    ##visualize krige result
    #krigeDems.data.copy_(((krigeDems.data-L_ele)/(H_ele-L_ele)-0.5)*2)
    #vutils.save_image(krigeDems.data,'%s/gen_epoch_%03d_krige.png' % (opt.outf,opt.epoch),normalize=True)

    mdems.data.copy_(L_ele+(mdems.data/2+0.5)*(H_ele-L_ele))
    fake.data.copy_(L_ele+(fake.data/2+0.5)*(H_ele-L_ele))
    #krigeDems.data.copy_(L_ele+(krigeDems.data/2+0.5)*(H_ele-L_ele))

    errG = criterion(fake,mdems)
    #errK = criterion(krigeDems,dems)
    print ('errG: %f'% (errG.data[0]))
    #print ('errG: %f errK: %f'% (errG.data[0],errK.data[0]))




