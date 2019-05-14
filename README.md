# cedgan-interpolation
(Source code) A spatial interpolation method: conditional encoder-decoder generative adversarial networks

# Usage: an example of calling the pre-trained model (200 epoches of training) with 10x10 sampled images
%run generate.py --batchSize 64 --netG outfile_100_samples --dataset DEM --ncp 100 --outf outfile_generate_loss/100samples


# Citation

Please cite our paper our CEDGAN helps you in your own work:
@article{zhu2019spatial,
  title={Spatial interpolation using conditional generative adversarial neural networks},
  author={Zhu, Di and Cheng, Ximeng and Zhang, Fan and Yao, Xin and Gao, Yong and Liu, Yu},
  journal={International Journal of Geographical Information Science},
  pages={1--24},
  year={2019},
  publisher={Taylor \& Francis}
}

