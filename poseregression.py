import argparse
import torch
import math
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
# from tensorboardX import SummaryWriter
from PIL import Image
import torch.utils.data as data
from model.ResNet50Layer import Res50PoseRess
from model.Discriminator import Discriminator
# from utils.loss import CrossEntropy2d
# from dataset.gta5_dataset import GTA5DataSet
# from dataset.cityscapes_dataset import cityscapesDataSet

# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

# MODEL = 'DeepLab'
BATCH_SIZE = 10
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/media/qing/DATA/BIM/10930457/RecurrentBIMPoseNetDataset/Synthetic dataset/Syn-pho-real/'
DATA_LIST_PATH = '/media/qing/DATA/BIM/10930457/RecurrentBIMPoseNetDataset/Synthetic dataset/Syn-pho-real/groundtruth_SynPhoReal.txt'
# IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
DATA_DIRECTORY_TARGET = '/media/qing/DATA/BIM/10930457/RecurrentBIMPoseNetDataset/Realdataset/Real/'
DATA_LIST_PATH_TARGET = '/media/qing/DATA/BIM/10930457/RecurrentBIMPoseNetDataset/Realdataset/Real/groundtruth_all_real_images.txt'
# INPUT_SIZE_TARGET = '1024,512'
LEARNING_RATE = 1e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 500
SNAPSHOT_DIR = '/media/qing/DATA/BIM/weights/Syn-pho-real/LSGAN/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log'




LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'LS'

epoches = 200

TARGET = 'cityscapes'
SET = 'train'

median_lst = []

def norm_q(x_q_base):

    Norm = torch.norm(x_q_base, 2, 1)
    norm_q_base = torch.div(torch.t(x_q_base), Norm)

    return torch.t(norm_q_base)



def median(lst):
    lst.sort()
    if len(lst) % 2 == 1:
        return lst[len(lst) // 2]
    else:
        return (lst[len(lst) // 2 - 1]+lst[len(lst) // 2]) / 2.0


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    # parser.add_argument("--model", type=str, default=MODEL,
    #                     help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    # parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
    #                     help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    # parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
    #                     help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    # parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
    #                     help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--cpu", action='store_true', help="choose to use cpu device.")
    parser.add_argument("--tensorboard", action='store_true', help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def default_loader(path):
    return Image.open(path).convert('RGB')

imgtransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


class myImageSourceloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fh = open(label)

        imgs = []
        class_names = []

        for line in fh.readlines():
            Label = []
            cls = line.split()
            fn_base = cls.pop(0)
            # fn_ref = cls.pop(0)
            if os.path.isfile(os.path.join(root, fn_base)):
                for ind,v in enumerate(cls):
                    Label.append(float(v))

                imgs.append((fn_base, torch.Tensor(Label[0:3]), torch.Tensor(Label[3:7])))
                #imgs.append((fn, tuple([float(v) for v in cls])))

        self.root = root
        self.imgs = imgs
        # self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn_base, base_t, base_q = self.imgs[index]
        img_base = self.loader(os.path.join(self.root, fn_base))
        # img_ref = self.loader(os.path.join(self.root, fn_ref))
        # sample_base = {'image':img_base,'label': base_q}
        # sample_ref = {'image': img_ref, 'label': ref_q}
        if self.transform is not None:
            img_base = self.transform(img_base)
            # img_ref, ref_q = self.transform(sample_ref)
        return img_base,torch.Tensor(base_t), torch.Tensor(base_q)

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes


class myImageTargetloder(data.Dataset):
    def __init__(self, root, label, transform=None, target_transform=None, loader=default_loader):
        fh = open(label)

        imgs = []
        class_names = []

        for line in fh.readlines():
            Label = []
            cls = line.split()
            fn_base = cls.pop(0)

            imgs.append(fn_base)
                #imgs.append((fn, tuple([float(v) for v in cls])))

        self.root = root
        self.imgs = imgs
        # self.classes = class_names
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn_base= self.imgs[index]
        img_base = self.loader(os.path.join(self.root, fn_base))
        # img_ref = self.loader(os.path.join(self.root, fn_ref))
        # sample_base = {'image':img_base,'label': base_q}
        # sample_ref = {'image': img_ref, 'label': ref_q}
        if self.transform is not None:
            img_base = self.transform(img_base)
            # img_ref, ref_q = self.transform(sample_ref)
        return img_base

    def __len__(self):
        return len(self.imgs)

    def getName(self):
        return self.classes

def main():
    """Create the model and start the training."""

    device = torch.device("cuda" if not args.cpu else "cpu")

    # w, h = map(int, args.input_size.split(','))
    # input_size = (w, h)
    #
    # w, h = map(int, args.input_size_target.split(','))
    # input_size_target = (w, h)

    cudnn.enabled = True

    # Create network
    model = Res50PoseRess()


    model.train()
    model.to(device)

    cudnn.benchmark = True

    # init D
    model_D1 = Discriminator(2048).to(device)
    # model_D1 = FCDiscriminator().to(device)
    # model_D2 = FCDiscriminator().to(device)

    model_D1.train()
    model_D1.to(device)
    #
    # model_D2.train()
    # model_D2.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    trainloader = data.DataLoader(
        myImageSourceloder(args.data_dir, args.data_list,transform = imgtransform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # trainloader_iter = data.DataLoader.__iter__(trainloader)

    targetloader = data.DataLoader(myImageTargetloder(args.data_dir_target, args.data_list_target,transform = imgtransform ),
                                   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    validloader = data.DataLoader( myImageSourceloder(args.data_dir_target, args.data_list_target,transform = imgtransform ),
                                   batch_size=1, shuffle=False, num_workers=args.num_workers,
                                   pin_memory=True)



    # implement model.optim_parameters(args) to handle different models' lr setting

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D1 = optim.Adam(model_D1.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D1.zero_grad()
    #
    # optimizer_D2 = optim.Adam(model_D2.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    # optimizer_D2.zero_grad()

    if args.gan == 'Vanilla':
        bce_loss = torch.nn.BCEWithLogitsLoss()
    elif args.gan == 'LS':
        bce_loss = torch.nn.MSELoss()
    pos_loss = torch.nn.MSELoss()

    # interp = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear', align_corners=True)
    # interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    # labels for adversarial training
    source_label = 0
    target_label = 1

    # # set up tensor board
    # if args.tensorboard:
    #     if not os.path.exists(args.log_dir):
    #         os.makedirs(args.log_dir)
    #
    #     writer = SummaryWriter(args.log_dir)

    pdist = nn.PairwiseDistance(2)
    for epoch in range(epoches):
        model.train()
        targetloader_iter = data.DataLoader.__iter__(targetloader)

        for i_iter, batch in enumerate(trainloader):

            loss_seg_value1 = 0
            loss_adv_target_value1 = 0
            loss_D_value1 = 0

            loss_seg_value2 = 0
            loss_adv_target_value2 = 0
            loss_D_value2 = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            optimizer_D1.zero_grad()
            # optimizer_D2.zero_grad()
            adjust_learning_rate_D(optimizer_D1, i_iter)
            # adjust_learning_rate_D(optimizer_D2, i_iter)

            # for sub_i in range(args.iter_size):

            # train G

            # don't accumulate grads in D
            for param in model_D1.parameters():
                param.requires_grad = False
            #
            # for param in model_D2.parameters():
            #     param.requires_grad = False

            # train with source

            # batch = trainloader_iter.next()

            images,t_real,q_real  = batch
            images = images.to(device)
            q_real = q_real.to(device)
            t_real = t_real.to(device)

            # t_pred_source, q_pred_source, feat_fc_source, feat_layer4_source,feat_layer3_source,feat_layer2_source,feat_layer1_source = model(images)
            x_layer1_source, x_layer2_source, x_layer3_source, x_layer4_source, x_1_q, x_1_t, x_2_q, x_2_t, x_3_q, x_3_t, x_4_q, x_4_t = model(
                images)

            loss_q_2 = pos_loss(x_2_q, q_real)
            loss_t_2 = pos_loss(x_2_t, t_real)
            loss_q_3 = pos_loss(x_3_q, q_real)
            loss_t_3 = pos_loss(x_3_t, t_real)
            loss_q_4 = pos_loss(x_4_q, q_real)
            loss_t_4 = pos_loss(x_4_t, t_real)
            loss_t = 0.15 * loss_t_2 + 0.15 * loss_t_3 + 0.7 * loss_t_4
            loss_q = 0.15 * loss_q_2 + 0.15 * loss_q_3 + 0.7 * loss_q_4
            loss = loss_t + 250 * loss_q

            # proper normalization
            # loss = loss / args.iter_size
            loss.backward()
            loss_seg_value1 += loss_q.item()
            loss_seg_value2 += loss_t.item()

            # train with target
            try:

                batch = targetloader_iter.next()
            except StopIteration:
                targetloader_iter = data.DataLoader.__iter__(targetloader)
                batch = targetloader_iter.next()


            images = batch
            images = images.to(device)

            # t_pred_source, q_pred_source, feat_fc_target, feat_layer4_target,feat_layer3_target,feat_layer2_target,feat_layer1_target = model(images)
            x_layer1_target, x_layer2_target, x_layer3_target, x_layer4_target, x_1_q, x_1_t, x_2_q, x_2_t, x_3_q, x_3_t, x_4_q, x_4_t = model(
                images)
            # pred_target1 = interp_target(pred_target1)
            # pred_target2 = interp_target(pred_target2)

            D_out1 = model_D1(x_layer4_target)
            # D_out2 = model_D2(F.softmax(pred_target2))

            loss_adv_target1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            # loss_adv_target2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss = loss_adv_target1
            # loss = loss / args.iter_size
            loss.backward()
            loss_adv_target_value1 += loss_adv_target1.item()
            # loss_adv_target_value2 += loss_adv_target2.item() / args.iter_size

            # train D

            # bring back requires_grad
            for param in model_D1.parameters():
                param.requires_grad = True

            # for param in model_D2.parameters():
            #     param.requires_grad = True

            # train with source
            feat_source = x_layer4_source.detach()
            # pred2 = pred2.detach()

            D_out1 = model_D1(feat_source)
            # D_out2 = model_D2(F.softmax(pred2))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_label).to(device))

            # loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_label).to(device))

            loss_D1 = loss_D1  / 2
            # loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            # loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            # loss_D_value2 += loss_D2.item()

            # train with target
            feat_target = x_layer4_target.detach()
            # pred_target2 = pred_target2.detach()

            D_out1 = model_D1(feat_target)
            # D_out2 = model_D2(F.softmax(pred_target2))

            loss_D1 = bce_loss(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(target_label).to(device))

            # loss_D2 = bce_loss(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(target_label).to(device))

            loss_D1 = loss_D1  / 2
            # loss_D2 = loss_D2 / args.iter_size / 2

            loss_D1.backward()
            # loss_D2.backward()

            loss_D_value1 += loss_D1.item()
            # loss_D_value2 += loss_D2.item()

            optimizer.step()
            optimizer_D1.step()
            # optimizer_D2.step()

            if args.tensorboard:
                scalar_info = {
                    'loss_seg1': loss_seg_value1,
                    'loss_seg2': loss_seg_value2,
                    'loss_adv_target1': loss_adv_target_value1,
                    'loss_adv_target2': loss_adv_target_value2,
                    'loss_D1': loss_D_value1,
                    'loss_D2': loss_D_value2,
                }

                # if i_iter % 10 == 0:
                #     for key, val in scalar_info.items():
                #         writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.snapshot_dir))
            print(
            'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.3f} loss_seg2 = {3:.3f} loss_adv1 = {4:.3f}, loss_adv2 = {5:.3f} loss_D1 = {6:.3f} loss_D2 = {7:.3f}'.format(
                i_iter, args.num_steps, loss_seg_value1, loss_seg_value2, loss_adv_target_value1, loss_adv_target_value2, loss_D_value1, loss_D_value2))

            if i_iter >= args.num_steps_stop - 1:
                print('save model ...')
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
                # torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D1.pth'))
                # torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps_stop) + '_D2.pth'))
                break

        if epoch % 1== 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(epoch) + '.pth'))
            # torch.save(model_D1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(epoch) + '_D1.pth'))
                # torch.save(model_D2.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D2.pth'))
        #evalue
        model.eval()
        with torch.no_grad():

            dis_Err_Count = []
            ort2_Err_count = []

            loss_counter = 0.

            for i, batch in enumerate(validloader):
                images, t_real, q_real = batch
                images = images.to(device)
                q_real = q_real.to(device)
                t_real = t_real.to(device)

                x_layer1_target, x_layer2_target, x_layer3_target, x_layer4_target, x_1_q, x_1_t, x_2_q, x_2_t, x_3_q, x_3_t, x_4_q, x_4_t = model(
                    images)


                dis_Err = pdist(x_4_t, t_real)
                dis_Err_Count.append(float(dis_Err))

                q_pred_source = norm_q(x_4_q)

                Ort_Err2 = float(2 * torch.acos(torch.abs(torch.sum(q_real * q_pred_source, 1))) * 180.0 / math.pi)

                ort2_Err_count.append(Ort_Err2)
                # result.append([dis_Err,Ort_Err2])

            dis_Err_i = median(dis_Err_Count)
            ort2_Err_i = median(ort2_Err_count)
            median_lst.append([dis_Err_i, ort2_Err_i])

            # print('average Distance err  = {} ,average orientation error = {} average Error = {}'.format(loss_counter / j,sum(dis_Err_Count)/j, sum(ort_Err_count)/j))
            print('Media distance error  = {}, median orientation error2 = {}'.format(dis_Err_i, ort2_Err_i))
            print(median_lst)

    #
        # if args.tensorboard:
        #     writer.close()


if __name__ == '__main__':
    main()
