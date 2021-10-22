import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io, transform
from PIL import Image
import random
import torchvision.transforms.functional as F

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.ToTensor(),
                            normalize])


def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


class SemanticDataset(Dataset):
    def __init__(self, image_dir, mask_dir, name_file, transforms=trans):
        self.images_pairs = []
        name_fid = open(name_file)
        count = 0
        for item in name_fid:
            count = count+1
            item = item.strip()
            self.images_pairs.append([os.path.join(image_dir,item), os.path.join(mask_dir,item)])
        name_fid.close()
        self.transforms = trans
        self.output_size = (256,256)
        
    def __len__(self):
        return len(self.images_pairs)
    
    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw
    
    def __getitem__(self, index):
        content_image, content_mask = self.images_pairs[index]
        content_image = Image.open(content_image)
        content_mask = Image.open(content_mask)
        
        i, j, h, w = self.get_params(content_image, self.output_size)
        content_image = F.crop(content_image, i, j, h, w)
        content_mask = F.crop(content_mask, i, j, h, w)
        
        # content_image = io.imread(content_image, plugin='pil')
        # style_image = io.imread(style_image, plugin='pil')
        # Unfortunately,RandomCrop doesn't work with skimage.io
        if self.transforms:
            content_image = self.transforms(content_image)
            content_mask = torch.tensor(np.array(content_mask))
        return content_image, content_mask

class PreprocessDataset(Dataset):
    def __init__(self, name_file, transforms=trans):
        self.images_pairs = []
        
        self.content_images = []
        name_fid = open(name_file)
        count = 0
        for item in name_fid:
            count = count+1
            item = item.strip()
            self.content_images.append(item)
        name_fid.close()
        self.style_images = []
        name_fid = open(name_file)
        count = 0
        for item in name_fid:
            count = count+1
            item = item.strip()
            self.style_images.append(item)
        name_fid.close()
        random.shuffle(self.style_images)
        for content_image,style_image in zip(self.content_images,self.style_images):
            self.images_pairs.append([content_image,style_image])
        
        self.transforms = trans
        self.output_size = (256,256)

    @staticmethod
    def _resize(source_dir, target_dir):
        print(f'Start resizing {source_dir} ')
        for i in tqdm(os.listdir(source_dir)):
            filename = os.path.basename(i)
            try:
                image = io.imread(os.path.join(source_dir, i))
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    H, W, _ = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    io.imsave(os.path.join(target_dir, filename), image)
                if len(image.shape) == 2:
                    H, W = image.shape
                    if H < W:
                        ratio = W / H
                        H = 512
                        W = int(ratio * H)
                    else:
                        ratio = H / W
                        W = 512
                        H = int(ratio * W)
                    #image = transform.resize(image, (H, W), mode='reflect', anti_aliasing=True)
                    image = transform.resize(image, (H, W), mode='reflect', preserve_range=True, anti_aliasing=True).astype('uint8') 
                    io.imsave(os.path.join(target_dir, filename), image)
            except:
                continue

    def __len__(self):
        return len(self.images_pairs)
    
    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __getitem__(self, index):
        content_image, style_image = self.images_pairs[index]
        content_mask = Image.open(content_image.replace("images","masks"))
        style_mask = Image.open(style_image.replace("images","masks"))
        content_image = Image.open(content_image)
        style_image = Image.open(style_image)
        
        i, j, h, w = self.get_params(content_image, self.output_size)
        content_image = F.crop(content_image, i, j, h, w)
        content_mask = F.crop(content_mask, i, j, h, w)
        i, j, h, w = self.get_params(content_image, self.output_size)
        style_image = F.crop(style_image, i, j, h, w)
        style_mask = F.crop(style_mask, i, j, h, w)
        
        # content_image = io.imread(content_image, plugin='pil')
        # style_image = io.imread(style_image, plugin='pil')
        # Unfortunately,RandomCrop doesn't work with skimage.io
        if self.transforms:
            content_image = self.transforms(content_image)
            style_image = self.transforms(style_image)
            content_mask = torch.tensor(np.array(content_mask))
            style_mask = torch.tensor(np.array(style_mask))
        return content_image, content_mask, style_image, style_mask
