'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Simple PyTorch MNIST example - training & testing
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_nndct.apis import torch_quantizer, dump_xmodel
from collections import OrderedDict

import argparse
import sys
import os
import shutil

from common import *


DIVIDER = '-----------------------------------------'


torchvision.datasets.MNIST.resources = [
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
]


def train_test(build_dir, batchsize):

    dset_dir = build_dir + '/dataset'
    float_model = build_dir + '/float_model'

    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
        device = torch.device('cpu')

    model = CNN().to(device)
    # new_model = New_CNN().to(device)
    # # save_path = os.path.join(float_model, 'test_ELM_model.pth')
    # for k,v in model.state_dict().items():
    #     print(k)
    # print(DIVIDER)
    # for k,v in new_model.state_dict().items():
    #     print(k)

    #image datasets
    train_dataset = torchvision.datasets.MNIST(dset_dir, 
                                               train=True, 
                                               download=True,
                                               transform=train_transform)
    test_dataset = torchvision.datasets.MNIST(dset_dir,
                                              train=False, 
                                              download=True,
                                              transform=test_transform)

    #data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batchsize, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize, 
                                              shuffle=False)


    # training with test after each epoch
    test(model, device, test_loader)
    # save the trained model
    shutil.rmtree(float_model, ignore_errors=True)    
    os.makedirs(float_model)   
    save_path = os.path.join(float_model, 'ELM_model.pth')
    torch.save(model.state_dict(), save_path) 
    print('ELM model written to',save_path)

    return

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])

        new_state_dict[name] = v
    return new_state_dict

def quantize(build_dir,quant_mode,batchsize):

  dset_dir = build_dir + '/dataset'
  float_model = build_dir + '/float_model'
  quant_model = build_dir + '/quant_model'


  # use GPU if available   
  if (torch.cuda.device_count() > 0):
    print('You have',torch.cuda.device_count(),'CUDA devices available')
    for i in range(torch.cuda.device_count()):
      print(' Device',str(i),': ',torch.cuda.get_device_name(i))
    print('Selecting device 0..')
    device = torch.device('cuda:0')
  else:
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

  # load trained model
  model = CNN().to(device)
  new_model = ELM_CNN().to(device)
  state_dict = torch.load(os.path.join(float_model, 'ELM_model.pth'))
#   model.load_state_dict(torch.load(os.path.join(float_model,'ELM_model.pth')))
  
  new_dict = copyStateDict(state_dict)
  keys=[]
  for k,v in new_dict.items():
    if k.startswith('network.10') or k.startswith('network.9'):
        continue
    keys.append(k)  
  # print(keys) 
  new_dict = {k:new_dict[k] for k in keys}
  new_model.load_state_dict(new_dict) 
  print(DIVIDER)
  print("load new model success!")
#   force to merge BN with CONV for better quantization accuracy
  optimize = 1

  # override batchsize if in test mode
  if (quant_mode=='test'):
    batchsize = 1
  
  rand_in = torch.randn([batchsize, 1, 28, 28])
  quantizer = torch_quantizer(quant_mode, new_model, (rand_in), output_dir=quant_model) 
  new_model.eval()
  if quant_mode == 'calib':
    quantizer.export_quant_config()
  if quant_mode == 'test':
    quantizer.export_xmodel(deploy_check=False, output_dir=quant_model)
  return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='build',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-q',  '--quant_mode', type=str, default='test',choices=['calib','test'], help='Quantization mode (calib or test). Default is calib')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ',args.batchsize)
    print(DIVIDER)
    train_test(args.build_dir, args.batchsize)
    # quantize(args.build_dir,'calib',args.batchsize)
    quantize(args.build_dir,args.quant_mode,args.batchsize)
    return



if __name__ == '__main__':
    run_main()
