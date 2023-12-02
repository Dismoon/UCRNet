import time
import random
import argparse
from torchvision import models
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from PIL import Image

from model3D import *
from model import *
from dataprocess import *
from utils import *

print(f'cuda version:{torch.version.cuda}')
# Set device
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# set random seed
set_seed()
# Set the parameters of the model
########################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--is_split', type=bool, default=True, help='if split the dataset')
parser.add_argument('--train_set', type=str, default='rawtrain', help="name of the train set")
parser.add_argument('--train_set_input', type=str, default='rawtrain/input_split', help="name of the train input set")
parser.add_argument('--train_set_label', type=str, default='rawtrain/label_split', help="name of the train label set")
parser.add_argument('--val_set_input', type=str, default='evalF/input', help="name of the validation input set")
parser.add_argument('--val_set_label', type=str, default='evalF/label', help="name of the validation label set")
parser.add_argument('--output_dir', type=str, default='test/output', help="test output")
parser.add_argument('--test_set', type=str, default='test/input', help="test input")
parser.add_argument('--image_height', type=int, default=64, help="the height of image input")
parser.add_argument('--image_width', type=int, default=64, help="the width of image input")
parser.add_argument('--stride1', type=int, default=32, help="the length stride to cut images")
parser.add_argument('--stride2', type=int, default=32, help="the width stride to cut images")
parser.add_argument('--input_channel', type=int, default=3, help="the channel of the input")
parser.add_argument('--input_channel1', type=int, default=3, help="the channel of the input")
parser.add_argument('--output_channel', type=int, default=3, help="the channel of the output")
parser.add_argument('--output_channel1', type=int, default=3, help="the channel of the output")
parser.add_argument('--epoch', type=int, default=64, help="the number of epoch")
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
########################################################################################################################
# Crop the training dataset
print('\nPreparing data...\n')

# When training, set true. When testing, set false.
if args.is_split:
    image_crop(args.train_set,
               args.image_height,
               args.image_width,
               args.stride1,
               args.stride2
               )
# load data
train_data = imgdata(args.train_set_input, args.train_set_label)
valid_data = imgdata_val(args.val_set_input, args.val_set_label)

train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, 1)
print(f'Number of validating images:{len(valid_loader)}')

# set the model
net = RDN(args.input_channel1,
          args.output_channel1,
          args.D,
          args.C,
          args.G,
          args.G0,
          args.kernel_size
          )

net1 = RDN(args.input_channel,
            args.output_channel,
            args.D1,
            args.C,
            args.G,
            args.G0,
            args.kernel_size
            )


net.initialize_weight()
net1.initialize_weight()
net.to(device)
net1.to(device)

########################################################################################################################
# set loss
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

class DopLoss(nn.Module):
    def __init__(self):
        super(DopLoss, self).__init__()

    @staticmethod
    def forward(img, label):
        pred_J = img
        pred_S0 = 0.5*(pred_J[:, 0, :, :] + pred_J[:, 2, :, :])
        pred_S1 = 0.5*(pred_J[:, 0, :, :] - pred_J[:, 2, :, :])
        pred_S2 = 0.5*(2*pred_J[:, 1, :, :] - pred_S0)
        pred_S0 = torch.clamp(pred_S0, 1e-8, 1)
        pred_S1 = torch.clamp(pred_S1, 1e-8, 1)
        pred_S2 = torch.clamp(pred_S2, 1e-8, 1)
        pred_Dop = torch.div(torch.sqrt(pred_S1 ** 2 + pred_S2 ** 2), pred_S0 + 0.0001)

        GT_S0 = 0.5*(label[:, 0, :, :] + label[:, 2, :, :])
        GT_S1 = 0.5*(label[:, 0, :, :] - label[:, 2, :, :])
        GT_S2 = 0.5*(2*label[:, 1, :, :] - GT_S0)

        GT_Dop = torch.div(torch.sqrt(GT_S1 ** 2 + GT_S2 ** 2), GT_S0 + 0.0001)
        loss = improvedl1(pred_Dop, GT_Dop)

        return loss


class AopLoss(nn.Module):
    def __init__(self):
        super(AopLoss, self).__init__()

    @staticmethod
    def forward(img, label):
        pred_J = img
        pred_S0 = 0.5*(pred_J[:, 0, :, :] + pred_J[:, 2, :, :])
        pred_S1 = 0.5*(pred_J[:, 0, :, :] - pred_J[:, 2, :, :])
        pred_S2 = 0.5*(2 * pred_J[:, 1, :, :] - pred_S0)
        pred_S0 = torch.clamp(pred_S0, 1e-8, 1)
        pred_S1 = torch.clamp(pred_S1, 1e-8, 1)
        pred_S2 = torch.clamp(pred_S2, 1e-8, 1)
        pred_Aop = torch.atan2(pred_S2, pred_S1 + 0.0001) / 2
        pred_Aop = (pred_Aop + math.pi / 2.0) / math.pi

        GT_S0 = 0.5*(label[:, 0, :, :] + label[:, 2, :, :])
        GT_S1 = 0.5*(label[:, 0, :, :] - label[:, 2, :, :])
        GT_S2 = 0.5*(2 * label[:, 1, :, :] - GT_S0)

        GT_Aop = torch.atan2(GT_S2, GT_S1 + 0.0001) / 2
        GT_Aop = (GT_Aop + math.pi / 2.0) / math.pi
        loss = improvedl1(pred_Aop, GT_Aop)
        return loss

loss_fn = nn.L1Loss()
loss_fo = nn.MSELoss()
perceptualloss = Perceptual()
aop = AopLoss()
dop = DopLoss()
improvedl1 = L1improved()
########################################################################################################################
optimizer = torch.optim.Adam(net.parameters(), args.learning_rate)
optimizer1 = torch.optim.Adam(net1.parameters(), args.learning_rate)
print("初始化的学习率：", optimizer.defaults['lr'])
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

# The process of traning
    net.train()
    net1.train()

    for i, data in enumerate(train_loader):

        img, label = data
        img = img.to(device)
        label = label.to(device)
        imgdolp = net(img)

        img0 = imgdolp[:, :, 1:args.image_height:2, 1:args.image_width:2]
        img45 = imgdolp[:, :, 0:args.image_height:2, 1:args.image_width:2]
        img90 = imgdolp[:, :, 0:args.image_height:2, 0:args.image_width:2]
        img135 = imgdolp[:, :, 1:args.image_height:2, 0:args.image_width:2]

        img0 = img0.unsqueeze(2)
        img45 = img45.unsqueeze(2)
        img90 = img90.unsqueeze(2)
        img135 = img135.unsqueeze(2)

        img1 = torch.cat([img0, img45, img90], 2)
        imageoutput = net1(img1)

        label0 = label[:, :, 1:args.image_height:2, 1:args.image_width:2]
        label45 = label[:, :, 0:args.image_height:2, 1:args.image_width:2]
        label90 = label[:, :, 0:args.image_height:2, 0:args.image_width:2]
        label135 = label[:, :, 1:args.image_height:2, 0:args.image_width:2]
        labelS0 = 0.5 * label0 + 0.5 * label90

        label0 = label0.unsqueeze(2)
        label45 = label45.unsqueeze(2)
        label90 = label90.unsqueeze(2)
        label135 = label135.unsqueeze(2)
        labell = torch.cat([label0, label45, label90], 2)

        imageoutput0 = imageoutput[:, :, 0:1, :, :]
        imageoutput45 = imageoutput[:, :, 1:2, :, :]
        imageoutput90 = imageoutput[:, :, 2:3, :, :]
        #imageoutput135 = imageoutput[:, :, 3:4, :, :]

        imageoutput0 = imageoutput0.squeeze(2)
        imageoutput45 = imageoutput45.squeeze(2)
        imageoutput90 = imageoutput90.squeeze(2)
        #imageoutput135 = imageoutput135.squeeze(2)

        imgS0 = 0.5 * imageoutput0 + 0.5 * imageoutput90
        imgS0 = torch.clamp(imgS0, 0, 1)
        # backward
        optimizer.zero_grad()
        optimizer1.zero_grad()
        # loss: perceptual+L1
        loss1 = perceptualloss(imgS0, labelS0)
        loss2 = improvedl1(imageoutput, labell)

        sum = loss1+loss2
        a = loss1/sum
        b = loss2/sum
        loss = a*loss1 + b*loss2
        loss.backward()
        iter += 1
        loss_mean += loss.item()
        loss_mean1 += loss1.item()
        loss_mean2 += loss2.item()
        # Update weight
        optimizer.step()
        optimizer1.step()
        scheduler.step()
        scheduler1.step()

        if iter % 50 == 0:
            psnr = cal_psnr(imgS0, labelS0).item()
            ssimm = ssim(imgS0, labelS0).item()
            print('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] time[{:.4f}min]'
                  ' Sum:[{:.4f}] per:[{:.4f}] L1:[{:.4f}]PSNR[{:.4f}]'.format(
                epoch + 1 + last_epoch,
                Epoch + last_epoch,
                i + 1,
                len(train_loader),
                (time.time() - time0) / 60,
                loss_mean / 50,
                loss_mean1 / 50,
                loss_mean2 / 50,
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

    if not os.path.isdir('evalspecific/outputno1/%s/' % (epoch + 1 + last_epoch)):
        os.makedirs('evalspecific/outputno1/%s/' % (epoch + 1 + last_epoch))

    # Evaluating the net after 1 epoch and save the outcome
    net.eval()
    net1.eval()

    with torch.no_grad():
        val_loss_mean = 0
        val_psnr = 0
        toPIL = transforms.ToPILImage()
        for j, dataval in enumerate(valid_loader):
            imgval, labelval = dataval
            imgval = imgval.to(device)
            labelval = labelval.to(device)
            imgval = net(imgval)
            labeleval0 = labelval[:, :, 1:1224:2, 1:1024:2]
            labeleval45 = labelval[:, :, 0:1224:2, 1:1024:2]
            labeleval90 = labelval[:, :, 0:1224:2, 0:1024:2]

            labelevalS0 = 0.5 * labeleval0 + 0.5 * labeleval90

            img0 = imgval[:, :, 1:1224:2, 1:1024:2]
            img45 = imgval[:, :, 0:1224:2, 1:1024:2]
            img90 = imgval[:, :, 0:1224:2, 0:1024:2]
            img135 = imgval[:, :, 1:1224:2, 0:1024:2]
            img0 = img0.unsqueeze(2)
            img45 = img45.unsqueeze(2)
            img90 = img90.unsqueeze(2)
            img135 = img135.unsqueeze(2)

            img = torch.cat([img0, img45, img90], 2)

            imageoutput = net1(img)
            imageoutput0 = imageoutput[:, :, 0:1, :, :]
            imageoutput45 = imageoutput[:, :, 1:2, :, :]
            imageoutput90 = imageoutput[:, :, 2:3, :, :]
          #  imageoutput135 = imageoutput[:, :, 3:4, :, :]
            imageoutput0 = imageoutput0.squeeze(2)
            imageoutput45 = imageoutput45.squeeze(2)
            imageoutput90 = imageoutput90.squeeze(2)
           # imageoutput135 = imageoutput135.squeeze(2)

            predS = 0.5 * imageoutput0 + 0.5 * imageoutput90
            predS0 = torch.clamp(predS, 0, 1)
            predSF = predS0.squeeze(0)
            labelevalS0 = labelevalS0.squeeze(0)
            xxx = toPIL(predSF)
            labelfinal = toPIL(labelevalS0)

            b1, g1, r1 = labelfinal.split()
            b, g, r = xxx.split()
            val_psnr += cal_psnr(predS, labelevalS0).item()
            final = Image.merge('RGB', (r, g, b))
            labelfinal = Image.merge('RGB', (r1, g1, b1))

            final.save('evalspecific/output/%s/' % (epoch + 1 + last_epoch) + str(j + 1) + '_out.png')
            labelfinal.save('evalspecific/output/%s/' % (epoch + 1 + last_epoch) + str(j + 1) + '_labelout.png')
########################################################################################################################
# Saving best checkpoint and other checkpoint
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
########################################################################################################################