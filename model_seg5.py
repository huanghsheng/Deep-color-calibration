import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import torch


def calc_mean_std(features, mask):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features1 = features.reshape(batch_size, c, -1)
    mask1 = F.interpolate((mask==1).unsqueeze(0).float(), size=features.size()[2:]).repeat(batch_size,c,1,1)
    mask1 = mask1.reshape(batch_size, c, -1).bool()
    features_mean = features1[mask1].reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features1[mask1].reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    #features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    #features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, content_mask, style_features, style_mask):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features, content_mask)
    style_mean, style_std = calc_mean_std(style_features, style_mask)
    #normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    normalized_features = (content_features - content_mean) + style_mean
    return normalized_features


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]
        self.slice2 = vgg[2: 7]
        self.slice3 = vgg[7: 12]
        #self.slice4 = vgg[12: 21]
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, images, output_last_feature=False):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        #h4 = self.slice4(h3)
        if output_last_feature:
            return h3
        else:
            return h1, h2, h3


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(256, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        #h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)
        return h

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = Decoder()
        self.semantic_decoder = Decoder()

    def generate(self, content_images, style_images, alpha=1.0):
        content_features = self.vgg_encoder(content_images, output_last_feature=False)
        style_features = self.vgg_encoder(style_images, output_last_feature=False) 
        content_mask = self.semantic_decoder(content_features[2]).argmax(dim=1)
        style_mask = self.semantic_decoder(style_features[2]).argmax(dim=1)
        t = adain(content_features[2], content_mask, style_features[2], style_mask)
        t = alpha * t + (1 - alpha) * content_features[2]
        out = self.decoder(t)
        return out

    @staticmethod
    def calc_content_loss(out_features, t, mask):
        mask1 = F.interpolate((mask==1).unsqueeze(0).float(), size=out_features.shape[2:]).repeat(out_features.size(0),out_features.size(1),1,1).bool()                   
        out_features = out_features[mask1]
        t = t[mask1]
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, content_mask, style_middle_features, style_mask):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = calc_mean_std(c, content_mask)
            s_mean, s_std = calc_mean_std(s, style_mask)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, content_mask, style_images, style_mask, alpha=1.0, lam=1.5):
        #content_mask_back = content_mask.clone()
        #style_mask_back = style_mask.clone()
        content_features = self.vgg_encoder(content_images, output_last_feature=False)
        #content_mask = F.interpolate(content_mask.unsqueeze(0).float(), size=[64,64]).squeeze(0).unsqueeze(1)
        style_features = self.vgg_encoder(style_images, output_last_feature=False)
        #style_mask = F.interpolate(style_mask.unsqueeze(0).float(), size=[64,64]).squeeze(0).unsqueeze(1)
        
        t = adain(content_features[2], content_mask, style_features[2], style_mask)
        t = alpha * t + (1 - alpha) * content_features[2]
        out = self.decoder(t)

        #output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        lam = 2.5
        loss_c_local = self.calc_content_loss(output_middle_features[1], content_features[1], content_mask)
        loss_s_local = self.calc_style_loss(output_middle_features, content_mask, style_middle_features, style_mask)
        loss_local = loss_c_local + lam * loss_s_local
        
        reconstruct_images = self.decoder(content_features[2])
        reconstruct_loss = torch.norm((reconstruct_images-content_images), p=1)
        
        loss = loss_local+0.2*reconstruct_loss*1e-4
        
        out_mask = self.semantic_decoder(content_features[2])
        out_mask, content_mask = out_mask.cuda(), content_mask.long().cuda()
        loss_semantic = cross_entropy2d(out_mask, content_mask)
        
        return loss,loss_semantic,loss_c_local,loss_s_local,reconstruct_loss
    
    def semantic_backward(self, content_images, content_mask):
        content_features = self.vgg_encoder(content_images, output_last_feature=False)
        out_mask = self.semantic_decoder(content_features[2])
        out_mask, content_mask = out_mask.cuda(), content_mask.long().cuda()
        loss = cross_entropy2d(out_mask, content_mask)
        return out_mask,loss
    
    def semantic_forward(self, content_images):
        content_features = self.vgg_encoder(content_images, output_last_feature=False)
        out_mask = self.semantic_decoder(content_features[2])
        out_mask = out_mask.argmax(dim=1)
        out_mask = out_mask.cuda()
        return out_mask



















