
import time
import argparse
from DWGAN import DWGenerator,SSD_SNGANDiscriminator
import sys
sys.path.append('../..')
from dataset_load import Tiny_ImageNet_dataset,CustomCIFAR10Dataset,GTSRB_dataset
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn
from torch.utils.tensorboard import SummaryWriter
import pickle
import torch.nn.functional as F
import numpy as np
import os
import torch
import os.path as path
from torchvision import transforms
from torchvision import utils
from torchvision.models import vgg16
from perceptual import LossNetwork


normalize_params={
    'cifar10':{'mean':(0.4914, 0.4822, 0.4465),'std':(0.2023, 0.1994, 0.2010)},
    'tiny_imagenet':{'mean':(0.4793, 0.4463, 0.3910),'std':(0.2296, 0.2260, 0.2239)},
    'gtsrb':{'mean':(0., 0., 0.),'std':(1., 1., 1.)},
}


def netC_normalize(tensor):
    MEAN=normalize_params[args.dataset]['mean']
    STD=normalize_params[args.dataset]['std']
    MEAN = torch.tensor(MEAN).view(3, 1, 1).cuda()
    STD = torch.tensor(STD).view(3, 1, 1).cuda()
    return (tensor -MEAN)/STD

def netGAN_denormalize(normalized_image, mean=[0.5,0.5,0.5 ], std=[0.5,0.5,0.5 ]):
    mean = torch.tensor(mean).view(3, 1, 1).cuda()
    std = torch.tensor(std).view(3, 1, 1).cuda()
    return normalized_image * std + mean

def set_seed(seed):
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
# --- Parse config ---#
parser = argparse.ArgumentParser(description='frequency GAN perturb')
parser.add_argument('--dataset',type=str,default='tiny_imagenet',choices=['cifar10','tiny_imagenet','gtsrb'])
parser.add_argument('--version',type=int,default=10, help='training version')
parser.add_argument('--fea_w',type=float,default=1, help='weight of feature loss')
parser.add_argument('--epsilon',type=float,default=0.01, help='pertubation range')
parser.add_argument('--Dreal_w',type=float,default=1.0, help='weight of feature loss')
parser.add_argument('--perc_w',type=float,default=0)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train',  action='store_true',default=False)
group.add_argument('--test',  action='store_true',default=False)
# --- Parse hyper-parameters train --- #
parser.add_argument('--learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('--train_batch_size', help='Set the training batch size', default=64, type=int)
parser.add_argument('--train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--load_ckpt',  type=int,default=0)
args = parser.parse_args()

# ---dataset--- #
train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5 ), (0.5,0.5,0.5 )),
    ])
if args.dataset == 'cifar10':
    train_data = CustomCIFAR10Dataset(train=True,transform=train_transform).get_dataset()
if args.dataset == 'tiny_imagenet':
    train_data = Tiny_ImageNet_dataset(train=True,transform=train_transform).get_dataset()
elif args.dataset == 'gtsrb':
    train_data = GTSRB_dataset(train=True,transform=train_transform).get_dataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

# --- path  --- #
args.root_path=os.path.join('output',f'{args.dataset}',f'v{args.version}',f'f{args.fea_w}',f'e{args.epsilon}',f'p{args.perc_w}')
args.model_save_dir=os.path.join(args.root_path,'model')
args.img_path = path.join(args.root_path, 'imgs')
args.sample_path= path.join(args.root_path, 'samples')
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
if not os.path.exists(args.img_path):
    os.makedirs(args.img_path)
if not os.path.exists(args.sample_path):
    os.makedirs(args.sample_path)

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
if args.dataset=='tiny_imagenet':
    Generator = DWGenerator(64).cuda()
    Discriminator = SSD_SNGANDiscriminator(64).cuda()
if args.dataset=='cifar10' or args.dataset == 'gtsrb':
    Generator = DWGenerator(32).cuda()
    Discriminator = SSD_SNGANDiscriminator(32).cuda()


try:
    Generator.load_state_dict(torch.load(os.path.join(args.model_save_dir, f'G_epoch{args.load_ckpt}.pkl')))
    Discriminator.load_state_dict(torch.load(os.path.join(args.model_save_dir, f'D_epoch{args.load_ckpt}.pkl')))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

netC = vgg16_bn(weights=None)
in_feature = netC.classifier[-1].in_features
if args.dataset == 'cifar10':
    netC.classifier[-1] = torch.nn.Linear(in_feature, 10)
    netC.load_state_dict(torch.load('../../model/cifar10/Source_Model/source_model.pth'))

elif args.dataset == 'gtsrb':
    netC.classifier[-1] = torch.nn.Linear(in_feature, 43)
    netC.load_state_dict(torch.load('../../model/gtsrb/Source_Model/source_model.pth'))

elif args.dataset == 'tiny_imagenet':
    netC.classifier[-1] = torch.nn.Linear(in_feature, 100)
    netC.load_state_dict(torch.load('../../model/tiny_imagenet/Source_Model/source_model.pth'))
netC = netC.cuda()
if args.train:
# --- Build optimizer --- #
    G_optimizer = torch.optim.Adam(Generator.parameters(), lr=0.0001)
    scheduler_G = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, milestones=[3000, 5000, 8000], gamma=0.5)
    D_optimizer = torch.optim.Adam(Discriminator.parameters(), lr=0.0001)
    scheduler_D = torch.optim.lr_scheduler.MultiStepLR(D_optimizer, milestones=[5000, 7000, 8000], gamma=0.5)

    # --- Loss --- #
    def adversarial_loss(D, real_img, fake_img):
        real_loss = torch.mean((D(real_img) - 1) ** 2)
        fake_loss = torch.mean(D(fake_img) ** 2)
        return real_loss + fake_loss


    def reconstruction_loss(real_img, fake_img):
        return torch.mean((real_img - fake_img) ** 2)


    def feature_loss(real_features, fake_features):
        pred = torch.max(real_features, 1)[1].data.squeeze().detach()
        return  torch.nn.CrossEntropyLoss()(fake_features,pred)

    def margin_loss(prob):
        pred,indices=torch.topk(prob,2)
        return torch.dist(pred[:,0],pred[:,1]+args.distance,p=2)

    ##perceptual loss
    vgg_model = vgg16(weights=True)
    vgg_model = vgg_model.features[:16].to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()

    summary_path=os.path.join(args.root_path,'summary')
    os.makedirs(summary_path,exist_ok=True)
    writer = SummaryWriter(log_dir=summary_path)

# --- Strat training --- #
    iteration = 0
    best_psnr = 0
    for epoch in range(args.load_ckpt+1,args.train_epoch):
        start_time = time.time()
        Generator.train()
        Discriminator.train()
        netC.eval()
        for batch_idx, (real_images, real_labels) in enumerate(train_loader):
            iteration += 1
            real_images = real_images.to(device)
            fake_images = Generator(real_images)

            # fake_images=torch.clip(real_images+args.epsilon*perturbation,-1,1)

            #D training
            Discriminator.zero_grad()
            out_spectral_real, out_spatial_real = Discriminator(real_images)
            out_spectral_fake, out_spatial_fake = Discriminator(fake_images)
            errC =F.relu(1.0 - out_spectral_real).mean() + F.relu(1.0 + out_spectral_fake).mean()
            out_real = 0.5 * out_spectral_real.detach() + 0.5 * out_spatial_real
            out_fake = 0.5 * out_spectral_fake.detach() + 0.5 * out_spatial_fake
            errD =F.relu(1.0 - out_real).mean() + F.relu(1.0 + out_fake).mean()
            D_loss = errD + args.Dreal_w*errC
            D_loss.backward(retain_graph=True)

            #G training
            adversarial_loss =-out_fake.mean()
            perceptual_loss = loss_network(fake_images, real_images)
            features_real = netC(netC_normalize(netGAN_denormalize(real_images)))
            features_fake = netC(netC_normalize(netGAN_denormalize(fake_images)))
            feat_loss = feature_loss(features_real, features_fake)
            Generator.zero_grad()

            G_loss = adversarial_loss - args.fea_w*feat_loss+args.perc_w * perceptual_loss
            G_loss.backward()
            D_optimizer.step()
            G_optimizer.step()
        scheduler_G.step()
        scheduler_D.step()

        if epoch % 1 == 0:
            print(f"epoch:{epoch}",
                  f"out_spectral_real:{out_spectral_real.mean()},",
                  f"out_spatial_real:{out_spatial_real.mean()},",
                  f"out_spectral_fake:{out_spectral_fake.mean()},",
                  f"out_spatial_fake:{out_spatial_fake.mean()},",
                  f'errD: {errD.item()}',
                  f'errC: {errC.item()}',
                  f"loss_G:{G_loss},",
                  f"loss_D:{D_loss},",
                  f"adversarial loss:{adversarial_loss.mean()},",
                  f"feature loss:{feat_loss.mean()},",
                  f"perceptual_loss: {perceptual_loss.item()}",
                  )
            writer.add_scalars('G training', {'Generator total loss': G_loss.item(),
                                              'adversarial_loss': adversarial_loss.item(),
                                              'perceptual_loss': perceptual_loss.item(),
                                              'feature_loss': feat_loss.item()}, iteration)
            writer.add_scalars('D training', {'Driscriminator loss': D_loss.item(),
                                              'errD': errD.item(),
                                              'errC': errC.item(),
                                              },
                               iteration)
        if epoch % 10 == 0:
            ###save image
            print('save images...')
            Generator.eval()
            samples=iter(train_loader).__next__()
            imgs = samples[0].cuda()
            labels = samples[1].cuda()
            fake_=Generator(imgs)
            utils.save_image(netGAN_denormalize(fake_), '{}/{}_imgs.png'.format(args.img_path,str(epoch).zfill(5)),nrow=8)

            ###save model
            torch.save(Generator.state_dict(), os.path.join(args.model_save_dir, 'G_epoch' + str(epoch) + '.pkl'))
            torch.save(Discriminator.state_dict(), os.path.join(args.model_save_dir, 'D_epoch' + str(epoch) + '.pkl'))

else:
    with torch.no_grad():
        sample_img=[]
        sample_label=[]
        print('sample...')
        Generator.eval()
        while len(sample_img)<200:
            samples=iter(train_loader).__next__()
            imgs = samples[0].cuda()
            labels = samples[1].cuda()
            fake_=Generator(imgs)
            pred_f = torch.argmax(netC(netC_normalize(netGAN_denormalize(fake_))), dim=1)
            pred_r = torch.argmax(netC(netC_normalize(netGAN_denormalize(imgs))), dim=1)
            sample_img = sample_img+list(torch.unbind(fake_[torch.where(pred_f != pred_r)]))
            sample_label= sample_label+list(torch.unbind(pred_r[torch.where(pred_f != pred_r)]))
        with open(os.path.join(args.sample_path, f'{args.load_ckpt}.pkl'), 'wb') as f:
            sample_img = torch.stack(sample_img[:200])
            sample_label = torch.stack(sample_label[:200])
            pickle.dump({'data':netC_normalize(netGAN_denormalize(sample_img[:200])), 'label': sample_label[:200]}, f)
