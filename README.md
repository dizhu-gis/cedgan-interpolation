# cedgan-interpolation
(Source code) A spatial interpolation method: conditional encoder-decoder generative adversarial networks

## Usage: 
### an example of calling the pre-trained model (200 epoches of training) with 10x10 sampled images
%run generate.py --batchSize 64 --netG outfile_100_samples --dataset DEM --ncp 100 --outf outfile_generate_loss/100samples

#### all the optional parameters:
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

## Citation
Please cite our paper our CEDGAN helps you in your own work:
@article{zhu2019spatial,
  title={Spatial interpolation using conditional generative adversarial neural networks},
  author={Zhu, Di and Cheng, Ximeng and Zhang, Fan and Yao, Xin and Gao, Yong and Liu, Yu},
  journal={International Journal of Geographical Information Science},
  pages={1--24},
  year={2019},
  publisher={Taylor \& Francis}
}

