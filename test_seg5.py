######################################################################
# The testing codes for semantic style transfer
######################################################################

import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from model_seg5 import Model
import numpy as np
import os
import time

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--content', '-c', type=str, default=None,
                        help='Content image path e.g. content.jpg')
    parser.add_argument('--style', '-s', type=str, default=None,
                        help='Style image path e.g. image.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None,
                        help='Output path for generated image, no need to add ext, e.g. out')
    parser.add_argument('--alpha', '-a', type=float, default=1,
                        help='alpha control the fusion degree in Adain')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='save directory for result and loss')

    args = parser.parse_args()
    args.model_state_path = './result/model_state/rice.pth'
    args.content_dir = './data/val.txt'
    args.style = './data/2017-9-30/images/DJI_0426_4_4.png'
    args.out_dir = './result/test_outputs'

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    # set model
    model = Model()
    if args.model_state_path is not None:
        model.load_state_dict(torch.load(args.model_state_path, map_location=lambda storage, loc: storage))
    model = model.to(device)
    
    style = Image.open(args.style)
    style = trans(style).unsqueeze(0).to(device)
    # style1 = Image.open(args.style1)
    # style1 = trans(style1).unsqueeze(0).to(device)
    # style2 = Image.open(args.style2)
    # style2 = trans(style2).unsqueeze(0).to(device)

    fid = open(args.content_dir, "r")
    count = 0
    pad = nn.ReflectionPad2d(50)
    start_time = time.time()
    with torch.no_grad():
        for item in fid:
            date = item.strip().split("/")[2]
            # if date==args.style1.strip().split("/")[2]:
            #     style = style1
            # if date==args.style2.strip().split("/")[2]:
            #     style = style2
            content = Image.open(item.strip())
            content = trans(content).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model.generate(content, style)
            out = denorm(out, device)
            #out = out[:,:,50:50+600,50:50+800]

            saved_name = args.out_dir+"/"+date+"/"+item.strip().split("/")[-1]
            #saved_name = "temp.png"
            save_image(out, saved_name, nrow=1)
            count = count+1
            print(count,item)
    fid.close()
    end_time = time.time()
    duration = end_time-start_time
    print(count, duration)


if __name__ == '__main__': 
    main()
