import time
import random
import argparse
from utils import *
from hsvrgb import *
from torchvision import models
from dataprocess import *
from torchvision import transforms
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image
from model3D import *
from model import *
from CR import *
#from prefetch_generator import BackgroundGenerator
from ssim import SSIM
#import kornia
from HSVNEW import *
print(f'cuda version:{torch.version.cuda}')
# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f'device:{device}')


'''class DataloaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())'''


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# set random seed
set_seed()
# Set the parameters of the model
parser = argparse.ArgumentParser()
parser.add_argument('--is_split', type=bool, default=False, help='if split the dataset')
parser.add_argument('--train_set', type=str, default='randomtrain3', help="name of the train set")
parser.add_argument('--train_set_input', type=str, default='randomtrain3/input_split', help="name of the train input set")
parser.add_argument('--train_set_label', type=str, default='randomtrain3/label_split', help="name of the train label set")
parser.add_argument('--val_set_input', type=str, default='evaltrans/input', help="name of the validation input set")
parser.add_argument('--val_set_label', type=str, default='evaltrans/label', help="name of the validation label set")
parser.add_argument('--output_dir', type=str, default='test/output', help="test output")
parser.add_argument('--test_set', type=str, default='test/input', help="test input")
parser.add_argument('--image_height', type=int, default=64, help="the height of image input")
parser.add_argument('--image_width', type=int, default=64, help="the width of image input")
parser.add_argument('--stride', type=int, default=32, help="the stride to cut images")
parser.add_argument('--input_channel', type=int, default=3, help="the channel of the input")
parser.add_argument('--output_channel', type=int, default=3, help="the channel of the output")
parser.add_argument('--epoch', type=int, default=60, help="the number of epoch")
parser.add_argument('--batch_size', type=int, default=32, help="the size of batch")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="the learning rate")
parser.add_argument('--lr_decay_steps', type=int, default=10, help="steps of learning rate decay")
parser.add_argument('--lr_decay_rate', type=float, default=0.6, help="rate of learning rate decay")
parser.add_argument('--checkpoint_dir', type=str, default="checkpoint", help="name of the checkpoint directory")
parser.add_argument('--D', type=int, default=8, help="the number of RDBs")
parser.add_argument('--D1', type=int, default=4, help="the number of RDB1s")
parser.add_argument('--C', type=int, default=6, help="the number of conv layers in each RDB")
parser.add_argument('--G', type=int, default=32, help="the channel of feature maps")
parser.add_argument('--G0', type=int, default=64, help="the channel of feature maps")
parser.add_argument('--kernel_size', type=int, default=3, help="the size of kernel")
args = parser.parse_args(args=[])

# Crop the training dataset
print('\nPreparing data...\n')
if args.is_split:
    image_crop(args.train_set,
               args.image_height,
               args.image_width,
               args.stride,
               args.stride
               )
# load data
train_data = imgdata(args.train_set_input, args.train_set_label)
valid_data = imgdata_val(args.val_set_input, args.val_set_label)

train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, 2)
print(f'Number of validating images:{len(valid_loader)}')

# set the model
net = RDN(args.input_channel,
          args.output_channel,
          args.D,
          args.C,
          args.G,
          args.G0,
          args.kernel_size
          )
net1 = DRDN(args.input_channel,
            args.output_channel,
            args.D1,
            args.C,
            args.G,
            args.G0,
            args.kernel_size
            )
print(net)
print(net1)
# Initialize the parameters of the net
net.initialize_weight()
net1.initialize_weight()
net = nn.DataParallel(net, device_ids=[0, 1])
net.cuda()
net1 = nn.DataParallel(net1, device_ids=[0, 1])
net1.cuda()
#net.to(device)
#net1.to(device)

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = 0
    variance = sigma * 1.5
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class L1improved(nn.Module):
    def __init__(self):
        super(L1improved, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        diff = torch.add(x, -y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class Perceptual(nn.Module):
    def __init__(self):
        super(Perceptual, self).__init__()
        self.model = models.vgg16(pretrained=True).features[0:4]
        self.model.eval()
        self.model.cuda()

    def forward(self, x, y):
        out1 = self.model(x)
        out2 = self.model(y)
        loss = loss_fo(out1, out2)
        return loss


class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        horizontal = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        horizontal = torch.FloatTensor(horizontal).unsqueeze(0).unsqueeze(0).cuda()
        horizontal = torch.cat([horizontal, horizontal, horizontal], 1)
        vertical = [[-1., -2., 1.], [0., 0., 0.], [1., 2., 1.]]
        vertical = torch.FloatTensor(vertical).unsqueeze(0).unsqueeze(0).cuda()
        vertical = torch.cat([vertical, vertical, vertical], 1)
        self.weightx = nn.Parameter(data=horizontal, requires_grad=False)
        self.weighty = nn.Parameter(data=vertical, requires_grad=False)

    def forward(self, x, y):
        x1 = F.conv2d(x, self.weightx)
        x2 = F.conv2d(x, self.weightx)
        y1 = F.conv2d(y, self.weightx)
        y2 = F.conv2d(y, self.weightx)
        torch.where(y1 >= 0, y1, -y1)
        torch.where(y2 >= 0, y1, -y2)
        torch.where(x1 >= 0, x1, -x1)
        torch.where(x2 >= 0, x1, -x2)
        x = x1 + x2
        y = y1 + y2
        out = improvedl1(x, y)
        return out


class DopLoss(nn.Module):
    def __init__(self):
        super(DopLoss, self).__init__()

    @staticmethod
    def forward(img, label):
        pred_J = img
        pred_S0 = 0.5 * (pred_J[:, 0, :, :] + pred_J[:, 1, :, :] + pred_J[:, 2, :, :] + pred_J[:, 3, :, :])
        pred_S1 = pred_J[:, 0, :, :] - pred_J[:, 2, :, :]
        pred_S2 = pred_J[:, 1, :, :] - pred_J[:, 3, :, :]
        pred_S0 = torch.clamp(pred_S0, 1e-8, 1)
        pred_S1 = torch.clamp(pred_S1, 1e-8, 1)
        pred_S2 = torch.clamp(pred_S2, 1e-8, 1)
        pred_Dop = torch.div(torch.sqrt(pred_S1 ** 2 + pred_S2 ** 2), pred_S0 + 0.0001)

        GT_S0 = 0.5 * (label[:, 0, :, :] + label[:, 1, :, :] + label[:, 2, :, :] + label[:, 3, :, :])
        GT_S1 = label[:, 0, :, :] - label[:, 2, :, :]
        GT_S2 = label[:, 1, :, :] - label[:, 3, :, :]

        GT_Dop = torch.div(torch.sqrt(GT_S1 ** 2 + GT_S2 ** 2), GT_S0 + 0.0001)
        loss = improvedl1(pred_Dop, GT_Dop)

        return loss


class AopLoss(nn.Module):
    def __init__(self):
        super(AopLoss, self).__init__()

    @staticmethod
    def forward(img, label):
        pred_J = img
        pred_S0 = 0.5 * (pred_J[:, 0, :, :] + pred_J[:, 1, :, :] + pred_J[:, 2, :, :] + pred_J[:, 3, :, :])
        pred_S1 = pred_J[:, 0, :, :] - pred_J[:, 2, :, :]
        pred_S2 = pred_J[:, 1, :, :] - pred_J[:, 3, :, :]
        pred_S0 = torch.clamp(pred_S0, 1e-8, 1)
        pred_S1 = torch.clamp(pred_S1, 1e-8, 1)
        pred_S2 = torch.clamp(pred_S2, 1e-8, 1)
        pred_Aop = torch.atan2(pred_S2, pred_S1 + 0.0001) / 2
        pred_Aop = (pred_Aop + math.pi / 2.0) / math.pi

        GT_S0 = 0.5 * (label[:, 0, :, :] + label[:, 1, :, :] + label[:, 2, :, :] + label[:, 3, :, :])
        GT_S1 = label[:, 0, :, :] - label[:, 2, :, :]
        GT_S2 = label[:, 1, :, :] - label[:, 3, :, :]

        GT_Aop = torch.atan2(GT_S2, GT_S1 + 0.0001) / 2
        GT_Aop = (GT_Aop + math.pi / 2.0) / math.pi
        loss = improvedl1(pred_Aop, GT_Aop)
        return loss


# class LAB(nn.Module):
#     def __init__(self):
#         super(LAB, self).__init__()
#
#     @staticmethod
#     def forward(mi):
#         mi = torch.clamp(mi, 0, 1)
#         mi = kornia.color.bgr_to_rgb(mi)
#         mi = kornia.color.rgb_to_lab(mi)
#         mi[:, 0:1, :, :] = mi[:, 0:1, :, :] / 100.
#         mi[:, 1:2, :, :] = (mi[:, 1:2, :, :] + 127.0) / 254.
#         mi[:, 2:3, :, :] = (mi[:, 2:3, :, :] + 127.0) / 254
#         return mi


# class HSV(nn.Module):
#     def __init__(self):
#         super(HSV, self).__init__()
#
#     @staticmethod
#     def forward(im):
#         im = torch.clamp(im, 0, 1)
#         prediction = kornia.color.bgr_to_rgb(im)
#         prediction = kornia.color.rgb_to_hsv(prediction)
#         prediction[:, 0:1, :, :] = prediction[:, 0:1, :, :] / (2 * math.pi)
#         return prediction


class Concathsv(nn.Module):
    def __init__(self):
        super(Concathsv, self).__init__()

    @staticmethod
    def forward(imm, laa):
        imm0 = (imm[:, :, 0:1, :, :]).squeeze(2)
        imm45 = (imm[:, :, 1:2, :, :]).squeeze(2)
        imm90 = (imm[:, :, 2:3, :, :]).squeeze(2)

        laa0 = (laa[:, :, 0:1, :, :]).squeeze(2)
        laa45 = (laa[:, :, 1:2, :, :]).squeeze(2)
        laa90 = (laa[:, :, 2:3, :, :]).squeeze(2)

        perc1 = perceptualloss(hsv(imm0), hsv(laa0))
        perc2 = perceptualloss(hsv(imm45), hsv(laa45))
        perc3 = perceptualloss(hsv(imm90), hsv(laa90))
        Final = (perc1 + perc2 + perc3) / 3
        ''' imm0 = (hsv(imm0)).unsqueeze(2)
        imm45 = (hsv(imm45)).unsqueeze(2)
        imm90 = (hsv(imm90)).unsqueeze(2)
        Final = torch.cat([imm0, imm45, imm90], 2)'''
        return Final


class Concatrgb(nn.Module):
    def __init__(self):
        super(Concatrgb, self).__init__()

    @staticmethod
    def forward(imm, laa):
        imm0 = (imm[:, :, 0:1, :, :]).squeeze(2)
        imm45 = (imm[:, :, 1:2, :, :]).squeeze(2)
        imm90 = (imm[:, :, 2:3, :, :]).squeeze(2)

        laa0 = (laa[:, :, 0:1, :, :]).squeeze(2)
        laa45 = (laa[:, :, 1:2, :, :]).squeeze(2)
        laa90 = (laa[:, :, 2:3, :, :]).squeeze(2)

        perc1 = perceptualloss(imm0, laa0)
        perc2 = perceptualloss(imm45, laa45)
        perc3 = perceptualloss(imm90, laa90)
        Final = (perc1 + perc2 + perc3) / 3
        ''' imm0 = (hsv(imm0)).unsqueeze(2)
        imm45 = (hsv(imm45)).unsqueeze(2)
        imm90 = (hsv(imm90)).unsqueeze(2)
        Final = torch.cat([imm0, imm45, imm90], 2)'''
        return Final


class Concatlab(nn.Module):
    def __init__(self):
        super(Concatlab, self).__init__()

    @staticmethod
    def forward(imm, laa):
        imm0 = (imm[:, :, 0:1, :, :]).squeeze(2)
        imm45 = (imm[:, :, 1:2, :, :]).squeeze(2)
        imm90 = (imm[:, :, 2:3, :, :]).squeeze(2)
        laa0 = (laa[:, :, 0:1, :, :]).squeeze(2)
        laa45 = (laa[:, :, 1:2, :, :]).squeeze(2)
        laa90 = (laa[:, :, 2:3, :, :]).squeeze(2)

        perc1 = perceptualloss(lab(imm0), lab(laa0))
        perc2 = perceptualloss(lab(imm45), lab(laa45))
        perc3 = perceptualloss(lab(imm90), lab(laa90))
        Final = (perc1 + perc2 + perc3) / 3
        '''imm0 = (lab(imm0)).unsqueeze(2)
        imm45 = (lab(imm45)).unsqueeze(2)
        imm90 = (lab(imm90)).unsqueeze(2)
        Final = torch.cat([imm0, imm45, imm90], 2)'''
        return Final


loss_fn = nn.L1Loss()
loss_fo = nn.MSELoss()
perceptualloss = Perceptual()
gradient = Gradient()
aop = AopLoss()
dop = DopLoss()
improvedl1 = L1improved()
blur = get_gaussian_kernel().cuda()
# lab = LAB()
# hsv = HSV()
ssim = SSIM()
concathsv = Concathsv()
concatlab = Concatlab()
concatrgb = Concatrgb()
color = RGB_HSV()
optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
optimizer1 = torch.optim.Adam(net1.parameters(), args.learning_rate)

# scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_steps * len(train_loader), args.lr_decay_rate)
scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, args.lr_decay_steps * len(train_loader), args.lr_decay_rate)

# summary
date = time.strftime('%Y.%m.%d', time.localtime(time.time()))
log_path = os.path.join('log_dir', date)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_path, filename_suffix='123456')

# Set initial checkpoint
iter = 0
last_epoch = 0

# If there exists a checkpoint, load it.
checkpoint_path = os.path.join(args.checkpoint_dir, date)
Epoch = args.epoch
if os.path.isdir(checkpoint_path):
    checkpointd = torch.load(checkpoint_path + '/checkpoint_d.pkl')
    checkpoint0 = torch.load(checkpoint_path + '/checkpoint_0.pkl')

    net.load_state_dict(checkpointd['model_state_dict'])
    net1.load_state_dict(checkpoint0['model_state_dict'])

    optimizer.load_state_dict(checkpointd['optimizer_state_dict'])
    optimizer1.load_state_dict(checkpoint0['optimizer_state_dict'])

    scheduler.load_state_dict(checkpointd['scheduler_state_dict'])
    scheduler1.load_state_dict(checkpoint0['scheduler_state_dict'])

    last_epoch = checkpointd['epoch']
    Epoch = args.epoch - last_epoch
    iter = checkpointd['iter']

print("\nNow start training!\n")
time0 = time.time()
min_psnr = 10
for epoch in range(Epoch):
    loss_mean = 0
    loss_mean1 = 0
    loss_mean2 = 0
    loss_mean3 = 0
    loss_mean4 = 0
    loss_mean5 = 0
    loss_mean6 = 0

    net.train()
    net1.train()

    for i, data in enumerate(train_loader):#一般ｌｏａｄｅｒ就是加载候一个批次一个批次的数据
        # forward
        img, label = data
        #img = img.to(device)

        img = img.cuda()
        label = label.cuda()
        #label = label.to(device)

        img = net(img)

        img0 = img[:, :, 1:64:2, 1:64:2]
        img45 = img[:, :, 0:64:2, 1:64:2]
        img90 = img[:, :, 0:64:2, 0:64:2]
        img135 = img[:, :, 1:64:2, 0:64:2]

        img0 = img0.unsqueeze(2)
        img45 = img45.unsqueeze(2)
        img90 = img90.unsqueeze(2)
        img135 = img135.unsqueeze(2)

        img1 = torch.cat([img0, img45, img90, img135], 2)

        imageoutput = net1(img1)

        label0 = label[:, :, 1:64:2, 1:64:2]
        label45 = label[:, :, 0:64:2, 1:64:2]
        label90 = label[:, :, 0:64:2, 0:64:2]
        label135 = label[:, :, 1:64:2, 0:64:2]
        labelS0 = 0.5 * (0.5 * label0 + 0.5 * label90 + 0.5 * label45 + 0.5 * label135)
        labelS1 = label0 - label90
        labelS2 = label45 - label135
        labelr = torch.cat(
            [label0[:, 2:3, :, :], label45[:, 2:3, :, :], label90[:, 2:3, :, :], label135[:, 2:3, :, :]], 1)
        labelg = torch.cat(
            [label0[:, 1:2, :, :], label45[:, 1:2, :, :], label90[:, 1:2, :, :], label135[:, 1:2, :, :]], 1)
        labelb = torch.cat(
            [label0[:, 0:1, :, :], label45[:, 0:1, :, :], label90[:, 0:1, :, :], label135[:, 0:1, :, :]], 1)

        label0 = label0.unsqueeze(2)
        label45 = label45.unsqueeze(2)
        label90 = label90.unsqueeze(2)
        label135 = label135.unsqueeze(2)
        labell = torch.cat([label0, label45, label90, label135], 2)
        '''labelrS0 = 0.5 * (0.5 * (label0[:, 2:3, :, :] + label90[:, 2:3, :, :]) + 0.5 * (
                    label45[:, 2:3, :, :] + label135[:, 2:3, :, :]))
        labelgS0 = 0.5 * (0.5 * (label0[:, 1:2, :, :] + label90[:, 1:2, :, :]) + 0.5 * (
                    label45[:, 1:2, :, :] + label135[:, 1:2, :, :]))
        labelbS0 = 0.5 * (0.5 * (label0[:, 0:1, :, :] + label90[:, 0:1, :, :]) + 0.5 * (
                    label45[:, 0:1, :, :] + label135[:, 0:1, :, :]))
        labelrS1 = label0[:, 2:3, :, :] - label90[:, 2:3, :, :]
        labelgS1 = label0[:, 1:2, :, :] - label90[:, 1:2, :, :]
        labelbS1 = label0[:, 0:1, :, :] - label90[:, 0:1, :, :]
        labelrS2 = label45[:, 2:3, :, :] - label135[:, 2:3, :, :]
        labelgS2 = label45[:, 1:2, :, :] - label135[:, 1:2, :, :]
        labelbS2 = label45[:, 0:1, :, :] - label135[:, 0:1, :, :]
        labelSr = torch.cat([labelrS0, labelrS1, labelrS2], 1)
        labelSg = torch.cat([labelgS0, labelgS1, labelgS2], 1)
        labelSb = torch.cat([labelbS0, labelbS1, labelbS2], 1)

       '''

        imageoutput0 = imageoutput[:, :, 0:1, :, :]
        imageoutput45 = imageoutput[:, :, 1:2, :, :]
        imageoutput90 = imageoutput[:, :, 2:3, :, :]
        imageoutput135 = imageoutput[:, :, 3:4, :, :]

        imageoutput0 = imageoutput0.squeeze(2)
        imageoutput45 = imageoutput45.squeeze(2)
        imageoutput90 = imageoutput90.squeeze(2)
        imageoutput135 = imageoutput135.squeeze(2)
        imgS0 = 0.5 * (0.5 * imageoutput0 + 0.5 * imageoutput90 + 0.5 * imageoutput45 + 0.5 * imageoutput135)
        imgS1 = imageoutput0 - imageoutput90
        imgS2 = imageoutput45 - imageoutput135
        imgS0 = torch.clamp(imgS0, 0, 1)
        '''imgS1 = imageoutput0 - imageoutput90
        imgS2 = 2 * imageoutput45 - imageoutput0 - imageoutput90
        imgS0 = torch.clamp(imgS0, 0, 1)
        imgS1 = torch.clamp(imgS1, 0, 1)
        imgS2 = torch.clamp(imgS2, 0, 1)'''
        imageoutputr = torch.cat([imageoutput0[:, 2:3, :, :], imageoutput45[:, 2:3, :, :], imageoutput90[:, 2:3, :, :],
                                  imageoutput135[:, 2:3, :, :]],
                                 1)

        imageoutputg = torch.cat(
            [imageoutput0[:, 1:2, :, :], imageoutput45[:, 1:2, :, :], imageoutput90[:, 1:2, :, :],
             imageoutput135[:, 1:2, :, :]], 1)

        imageoutputb = torch.cat(
            [imageoutput0[:, 0:1, :, :], imageoutput45[:, 0:1, :, :], imageoutput90[:, 0:1, :, :],
             imageoutput135[:, 0:1, :, :]], 1)

        '''imageutputrS0 = 0.5 * (0.5 * (imageoutput0[:, 2:3, :, :] + imageoutput90[:, 2:3, :, :]) + 0.5 * (
                    imageoutput45[:, 2:3, :, :] + imageoutput135[:, 2:3, :, :]))
        imageutputrS1 = imageoutput0[:, 2:3, :, :] - imageoutput90[:, 2:3, :, :]
        imageutputrS2 = imageoutput45[:, 2:3, :, :] - imageoutput135[:, 2:3, :, :]
        Sr = torch.cat([imageutputrS0, imageutputrS1, imageutputrS2], 1)

        imageutputgS0 = 0.5 * (0.5 * (imageoutput0[:, 1:2, :, :] + imageoutput90[:, 1:2, :, :]) + 0.5 * (
                    imageoutput45[:, 1:2, :, :] + imageoutput135[:, 1:2, :, :]))
        imageutputgS1 = imageoutput0[:, 1:2, :, :] - imageoutput90[:, 1:2, :, :]
        imageutputgS2 = imageoutput45[:, 1:2, :, :] - imageoutput135[:, 1:2, :, :]
        Sg = torch.cat([imageutputgS0, imageutputgS1, imageutputgS2], 1)

        imageutputbS0 = 0.5 * (0.5 * (imageoutput0[:, 0:1, :, :] + imageoutput90[:, 0:1, :, :]) + 0.5 * (
                    imageoutput45[:, 0:1, :, :] + imageoutput135[:, 0:1, :, :]))
        imageutputbS1 = imageoutput0[:, 0:1, :, :] - imageoutput90[:, 0:1, :, :]
        imageutputbS2 = imageoutput45[:, 0:1, :, :] - imageoutput135[:, 0:1, :, :]
        Sb = torch.cat([imageutputbS0, imageutputbS1, imageutputbS2], 1)'''

        # backward
        optimizer.zero_grad()
        optimizer1.zero_grad()
        loss1 = perceptualloss(imgS0, labelS0)
        d = improvedl1(imageoutput[:, 0:1, :, :, :], labell[:, 0:1, :, :, :])
        e = improvedl1(imageoutput[:, 1:2, :, :, :], labell[:, 1:2, :, :, :])
        f = improvedl1(imageoutput[:, 2:3, :, :, :], labell[:, 2:3, :, :, :])
        q = d+e+f
        dd = d/q
        ee = e/q
        ff = f/q
        loss2 = dd*d+ee*e+ff*f
        '''g = dop(imageoutputr, labelr)
        h = dop(imageoutputg, labelg)
        k = dop(imageoutputb, labelb)
        r = g + h + k
        gg = g / r
        hh = h / r
        kk = k / r
        loss3 = gg * g + hh * h + kk * k

        d = aop(imageoutputr, labelr)
        e = aop(imageoutputg, labelg)
        f = aop(imageoutputb, labelb)
        q = d + e + f
        dd = d / q
        ee = e / q
        ff = f / q
        loss4 = dd * d + ee * e + ff * f'''
        a = loss1 + loss2
        loss = (loss1/a)*loss1 + (loss2/a)*loss2
        loss.backward()
        iter += 1
        loss_mean += loss.item()
        loss_mean1 += loss1.item()
        loss_mean2 += loss2.item()
        #loss_mean3 += loss3.item()
        #loss_mean4 += loss4.item()
        # Update weight
        optimizer.step()
        optimizer1.step()
        scheduler.step()
        scheduler1.step()

        if iter % 20 == 0:
            psnr = cal_psnr(imgS0, labelS0).item()
            ssimm = ssim(imgS0, labelS0).item()
            print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] time[{:.4f}min]'
                  ' Sum:[{:.7f}] L1:[{:.7f}] Stokes:[{:.7f}] DOP:[{:.7f}] AOP:[{:.7f}] SSIM[{:.4f}] PSNR[{:.4f}]'.format(
                epoch + 1 + last_epoch,
                Epoch + last_epoch,
                i + 1,
                len(train_loader),
                (time.time() - time0) / 60,
                loss_mean / 20,
                loss_mean1 / 20,
                loss_mean2 / 20,
                loss_mean3 / 20,
                loss_mean4 / 20,
                ssimm,
                psnr)
            )

            loss_mean = 0
            loss_mean1 = 0
            loss_mean2 = 0
            loss_mean3 = 0
            loss_mean4 = 0
            loss_mean5 = 0
            loss_mean6 = 0
            writer.add_scalars('PSNR', {'Train': psnr}, iter)
        writer.add_scalars('Loss', {'Train': loss.item()}, iter)

    if not os.path.isdir('evaltrans/output/%s/' % (epoch + 1 + last_epoch)):
        os.makedirs('evaltrans/output/%s/' % (epoch + 1 + last_epoch))

    # Evaluating the net after 1 epoch and save the outcome
    net.eval()
    net1.eval()

    with torch.no_grad():
        val_loss_mean = 0
        val_psnr = 0
        toPIL = transforms.ToPILImage()
        for j, data in enumerate(valid_loader):
            # 前向传播forward
            img, label = data
            img = img.cuda()
            label = label.cuda()
            #img = img.to(device)
            #label = label.to(device)
            img = net(img)
            labeleval0 = label[:, :, 1:1224:2, 1:1024:2]
            labeleval45 = label[:, :, 0:1224:2, 1:1024:2]
            labeleval90 = label[:, :, 0:1224:2, 0:1024:2]
            labeleval135 = label[:, :, 1:1224:2, 0:1024:2]
            labelevalS0 = 0.5 * (0.5 * labeleval0 + 0.5 * labeleval90 + 0.5 * labeleval45 + 0.5 * labeleval135)

            img0 = img[:, :, 1:1224:2, 1:1024:2]
            img45 = img[:, :, 0:1224:2, 1:1024:2]
            img90 = img[:, :, 0:1224:2, 0:1024:2]
            img135 = img[:, :, 1:1224:2, 0:1024:2]

            img0 = img0.unsqueeze(2)
            img45 = img45.unsqueeze(2)
            img90 = img90.unsqueeze(2)
            img135 = img135.unsqueeze(2)

            img = torch.cat([img0, img45, img90, img135], 2)

            imageoutput = net1(img)

            imageoutput0 = imageoutput[:, :, 0:1, :, :]
            imageoutput45 = imageoutput[:, :, 1:2, :, :]
            imageoutput90 = imageoutput[:, :, 2:3, :, :]
            imageoutput135 = imageoutput[:, :, 3:4, :, :]

            imageoutput0 = imageoutput0.squeeze(2)
            imageoutput45 = imageoutput45.squeeze(2)
            imageoutput90 = imageoutput90.squeeze(2)
            imageoutput135 = imageoutput135.squeeze(2)
            predS = 0.5 * (0.5 * imageoutput0 + 0.5 * imageoutput90 + 0.5 * imageoutput45 + 0.5 * imageoutput135)
            predS0 = torch.clamp(predS, 0, 1)
            predSF = predS.squeeze(0)
            labelevalS0 = labelevalS0.squeeze(0)
            img = toPIL(predSF)
            labelfinal = toPIL(labelevalS0)
            b1, g1, r1 = labelfinal.split()
            b, g, r = img.split()
            val_psnr += cal_psnr(predSF, labelevalS0).item()
            final = Image.merge('RGB', (r, g, b))
            labelfinal = Image.merge('RGB', (r1, g1, b1))
            final.save('evaltrans/output/%s/' % (epoch + 1 + last_epoch) + str(j + 1) + '_out.png')
            labelfinal.save('evaltrans/output/%s/' % (epoch + 1 + last_epoch) + str(j + 1) + '_labelout.png')
        if (val_psnr / len(valid_loader)) > min_psnr:
            min_psnr = (val_psnr / len(valid_loader))
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "scheduler_state_dict": scheduler.state_dict(),
                          "epoch": last_epoch + epoch + 1,
                          'iter': iter}
            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path)
            path_checkpoint = os.path.join(checkpoint_path, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)
            checkpoint = {"model_state_dict": net1.state_dict(),
                          "optimizer_state_dict": optimizer1.state_dict(),
                          "scheduler_state_dict": scheduler1.state_dict(),
                          "epoch": last_epoch + epoch + 1,
                          'iter': iter}
            if not os.path.isdir(checkpoint_path):
                os.makedirs(checkpoint_path)
            path_checkpoint = os.path.join(checkpoint_path, "checkpoint_best0.pkl")
            torch.save(checkpoint, path_checkpoint)
        # writer.add_scalars('Loss', {'Valid': val_loss_mean / len(valid_loader)}, iter)
        writer.add_scalars('PSNR', {'Valid': val_psnr / len(valid_loader)}, iter)

    checkpoint = {"model_state_dict": net.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "scheduler_state_dict": scheduler.state_dict(),
                  "epoch": last_epoch + epoch + 1,
                  'iter': iter}
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    path_checkpoint = os.path.join(checkpoint_path, "checkpoint_d.pkl")
    torch.save(checkpoint, path_checkpoint)

    checkpoint = {"model_state_dict": net1.state_dict(),
                  "optimizer_state_dict": optimizer1.state_dict(),
                  "scheduler_state_dict": scheduler1.state_dict(),
                  "epoch": last_epoch + epoch + 1,
                  'iter': iter}
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    path_checkpoint = os.path.join(checkpoint_path, "checkpoint_0.pkl")
    torch.save(checkpoint, path_checkpoint)
