# -*- coding: utf-8 -*
'''
配置说明：
1. 原位深遥感图像，通过逐像素点除以2047（即2^11)的方式归一化，制作成网络输入，将网络输出保存成图像前，再乘以2047，
复原到原位深图像
2. GF：trainset: 19976 ×（128×128），testset: 90 ×（512×512）
3. QB：trainset: 18123 ×（128×128），testset: 24 ×（512×512）

5. SGD，momentum=0.9， base-lr: 0.001，每200epoch衰减一次
6. MSE loss
7. total epochs： 500， batchsize：32，
'''

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.nn as nn
from torch.nn import init
import time
import scipy.io as sio
import gdal, ogr, os, osr
from os.path import join
import math
import cv2
from visdom import Visdom

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
## 超参数设置
version = 1  # 版本号
mav_value = 1023  # GF：1023，QB：2047
satellite = 'gf'  #  gf，qb
method = 'pannet'
train_batch_size = 32  # 32
test_batch_size = 1
total_epochs = 500
lr = 0.001
lr_decay_freq = 200  # lr衰减 1/10
test_freq = 20
model_backup_freq = 20
num_workers = 1

## 文件夹设置
traindata_dir = '../TIF/train/'
testdata_dir = '../TIF/test/'
testsample_dir = '../pannet-results/test-samples-v{}/'.format(version)
evalsample_dir = '../pannet-results/eval-samples-v{}/'.format(version)
record_dir = '../pannet-results/record-v{}/'.format(version)
model_dir = '../pannet-results/models-v{}/'.format(version)
backup_model_dir = join(model_dir, 'backup_model/')
checkpoint_model = join(model_dir, '{}-{}-model.pth'.format(satellite, method))

if not os.path.exists(evalsample_dir):
    os.makedirs(evalsample_dir)
if not os.path.exists(testsample_dir):
    os.makedirs(testsample_dir)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(backup_model_dir):
    os.makedirs(backup_model_dir)

## Device configuration
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('==> gpu or cpu:', device, ', how many gpus available:', torch.cuda.device_count())

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ["mul.tif"])

def load_image(filepath):
    img = gdal.Open(filepath)  # 原始数据
    img = img.ReadAsArray()  # [C,W,H]
    if filepath.split('_')[1] != 'pan.tif':
        img = img.transpose(1, 2, 0)  # [W,H,C]
    img = img.astype(np.float32) / mav_value    # 归一化处理
    return img

class DatasetFromFolder(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_filenames = [join(img_dir, x.split('_')[0]) for x in os.listdir(img_dir) if is_image_file(x)]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):   # idx的范围是从0到len（self）
        input_pan = load_image('%s_pan.tif'%self.image_filenames[index])
        input_lr = load_image('%s_lr.tif'%self.image_filenames[index])
        input_lr_u = load_image('%s_lr_u.tif'%self.image_filenames[index])
        target = load_image('%s_mul.tif'%self.image_filenames[index])

        if self.transform:
            input_pan = self.transform(input_pan)
            input_lr = self.transform(input_lr)
            input_lr_u = self.transform(input_lr_u)
            target = self.transform(target)
        return input_pan, input_lr, input_lr_u, target

## 变换ToTensor
class ToTensor(object):
    def __call__(self, input):
        # 因为torch.Tensor的高维表示是 C*H*W,所以在下面执行from_numpy之前，要先做shape变换
        # 把H*W*C转换成 C*H*W  4*128*128
        if input.ndim == 3:
            input = np.transpose(input, (2, 0, 1))
            input = torch.from_numpy(input).type(torch.FloatTensor)
        else:
            input = torch.from_numpy(input).unsqueeze(0).type(torch.FloatTensor)
        return input

def get_train_set(traindata_dir):
    return DatasetFromFolder(traindata_dir,
                             transform=transforms.Compose([ToTensor()]))

def get_test_set(testdata_dir):
    return DatasetFromFolder(testdata_dir,
                             transform=transforms.Compose([ToTensor()]))

transformed_trainset = get_train_set(traindata_dir)
transformed_testset = get_test_set(testdata_dir)
print('train:', len(transformed_trainset), 'test:', len(transformed_testset))

## 训练集  ## 验证集  ## 测试集
trainset_dataloader = DataLoader(dataset=transformed_trainset, batch_size=train_batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True, drop_last=True)
testset_dataloader = DataLoader(dataset=transformed_testset, batch_size=test_batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True, drop_last=True)

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        # self.opt = opt
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

class Residual_Block(nn.Module):
    def __init__(self, channels):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        fea = self.relu(self.bn1(self.conv1(x)))
        fea = self.relu(self.bn2(self.conv2(fea)))
        result = fea + x
        return result

class PanNet(nn.Module):
    def __init__(self):
        super(PanNet, self).__init__()
        self.layer_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=4, kernel_size=8, stride=4, padding=2, output_padding=0)
        )
        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.layer_2 = nn.Sequential(
            self.make_layer(Residual_Block, 4, 32),
            nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)
        )

    def make_layer(self, block, num_of_layer, channels):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(channels))
        return nn.Sequential(*layers)

    def forward(self, pan_hp, lr_hp):
        lr_u_hp = self.layer_0(lr_hp)
        ms = torch.cat([pan_hp, lr_u_hp], dim=1)
        fea = self.layer_1(ms)
        output = self.layer_2(fea)
        return output

def array2raster(newRasterfn, rasterOrigin, pixelWidth, pixelHeight, array, bandSize):
    if (bandSize == 4):
        cols = array.shape[2]
        rows = array.shape[1]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')  # #存的数据格式

        outRaster = driver.Create(newRasterfn, cols, rows, 4, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        for i in range(1, 5):
            outband = outRaster.GetRasterBand(i)
            outband.WriteArray(array[i - 1, :, :])
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()
    elif (bandSize == 1):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')

        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_UInt16)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))

        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)

def denorm(x):
    x = (x * mav_value).astype(np.uint16)
    return x

def eval_img_save(x,name,k):
    x = x.numpy()
    x = np.transpose(x, (0, 2, 3, 1))   # [batch_size,512,512,4]
    if name == 'real_images':
        array2raster(join(evalsample_dir, 'real_images_{}_epoch{}.tif'.format(k + 1,total_epochs)),
                     [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
    else:
        array2raster(join(evalsample_dir, '{}_v{}_eval_fused_images_{}_epoch{}.tif'.format(method, version, k + 1,total_epochs)),
                     [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)

def test_img_save(x,name,epoch):
    x = np.transpose(x, (0, 2, 3, 1))
    x = x.numpy()  # [batch_size,512,512,4]
    if name == 'test_fused_images':
        array2raster(join(testsample_dir, 'test_fused_images_9_epoch{}.tif'.format(epoch)),
                            [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
    elif name == 'real_images':
        array2raster(join(testsample_dir, 'real_images_9_epoch{}.tif'.format(epoch)),
                            [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)
    elif name == 'test_pan_images':
        array2raster(join(testsample_dir, 'test_pan_images_9_epoch{}.tif'.format(epoch)),
                     [0, 0], 8, 8, denorm(x[0].reshape(x.shape[1], x.shape[2])), 1)
    else:
        array2raster(join(testsample_dir, 'test_lrms_images_9_epoch{}.tif'.format(epoch)),
                            [0, 0], 8, 8, denorm(x[0].transpose(2, 0, 1)), 4)

# get high-frequency
def get_edge(data):
    data = data.numpy()
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5),
                                                        normalize=True)  # 第二个参数的-1表示输出图像使用的深度与输入图像相同
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5), normalize=True)
    return torch.from_numpy(rs)

def adjust_learning_rate(lr, epoch, freq):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = lr * (0.1 ** (epoch // freq))
    return lr

# Device setting
criterion = nn.MSELoss().to(device)
model = PanNet()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)

# 配置GPU并行
if (torch.cuda.device_count() > 1):
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

# 模型训练
def train(model, trainset_dataloader, start_epoch):
    print('===>Begin Training!')
    model.train()
    steps_per_epoch = len(trainset_dataloader)
    total_iterations = total_epochs * steps_per_epoch
    print('total_iterations:{}'.format(total_iterations))

    train_loss_record = open('%s/train_loss_record.txt' % record_dir, "w")
    epoch_time_record = open('%s/epoch_time_record.txt' % record_dir, "w")
    time_sum = 0

    viz = Visdom()
    viz.line(np.array([0.]), np.array([0.]), win='pannet_train_loss', opts=dict(title='pannet_train loss'))

    for epoch in range(start_epoch + 1, total_epochs + 1):
        start = time.time()  # 记录每轮训练的开始时刻
        lr = 0.001
        # 配置更新的学习率
        lr = adjust_learning_rate(lr, epoch - 1, lr_decay_freq)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("=>epoch =", epoch, "lr =", optimizer.param_groups[0]["lr"])

        prefetcher = DataPrefetcher(trainset_dataloader)
        data = prefetcher.next()
        i = 0
        while data is not None:
            # run_step()
            i += 1
            if i >= steps_per_epoch:
                break
            img_pan, img_lr, img_lr_u, target = data[0], data[1], data[2], data[3]  # 此时数据类型是张量 [batchsize,C,W,H]
            lr_hp = get_edge(img_lr)
            pan_hp = get_edge(img_pan)
            # 把数据集中提取出来的数据放到device上去
            lr_hp = lr_hp.to(device)
            pan_hp = pan_hp.to(device)
            target = target.to(device)
            img_lr_u = img_lr_u.to(device)

            mrs = model(pan_hp, lr_hp)  # 网络输出
            train_fused_images = mrs + img_lr_u
            train_loss = criterion(train_fused_images, target)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        data = prefetcher.next()

        print('=> {}-{}-Epoch[{}/{}]: train_loss: {:.15f}'.format(satellite, method, epoch, total_epochs, train_loss.item()))
        train_loss_record.write("Epoch[{}/{}]: train_loss: {:.15f}\n".format(epoch, total_epochs, train_loss.item()))

        viz.line(np.array([train_loss.item()]), np.array([epoch]), win='pannet_train_loss', update='append')

        # Save the model checkpoints and backup
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, checkpoint_model)

        # backup a model every epoch
        if epoch % model_backup_freq == 0:
            torch.save(model.state_dict(), join(backup_model_dir, '{}-{}-model-epochs{}.pth'.format(satellite, method, epoch)))

        if epoch % test_freq == 0:
            checkpoint = torch.load(checkpoint_model)
            model.load_state_dict(checkpoint['model'])
            print('==>Testing the model after training {} epochs'.format(epoch))
            test(model,testset_dataloader, epoch)

        # 输出每轮训练花费时间
        time_epoch = (time.time() - start)
        time_sum += time_epoch
        print('==>No:{} epoch training costs {:.4f}min'.format(epoch, time_epoch / 60))
        epoch_time_record.write(
            "No:{} epoch training costs {:.4f}min\n".format(epoch, time_epoch / 60))

# 模型验证
def test(model, testset_dataloader, epoch):
    avg_test_loss = 0
    model.eval()
    test_loss_record = open('%s/test_loss_record.txt' % record_dir, "a")
    with torch.no_grad():
        for k, data in enumerate(testset_dataloader):
            img_pan, img_lr, img_lr_u, target = data[0], data[1], data[2], data[3]  # 此时数据类型是张量
            lr_hp = get_edge(img_lr)
            pan_hp = get_edge(img_pan)
            # 把数据集中提取出来的数据放到device上去
            lr_hp = lr_hp.to(device)
            pan_hp = pan_hp.to(device)
            target = target.to(device)
            img_lr_u = img_lr_u.to(device)

            mrs = model(pan_hp, lr_hp)  # 网络输出
            test_fused_images = torch.add(mrs, img_lr_u)
            test_loss = criterion(test_fused_images, target)
            avg_test_loss += test_loss.item()

            # 保存融合图像
            if k == 8:
                print('==>Save the test_fused_images')
                test_fused_images = test_fused_images.cpu()
                test_img_save(test_fused_images, 'test_fused_images', epoch)

                if epoch == test_freq:
                    print('==>Save the reference_images')
                    real_images, img_lr_u, img_pan = target.cpu(), img_lr_u.cpu(), img_pan.cpu()
                    test_img_save(real_images, 'real_images', epoch)
                    test_img_save(img_lr_u, 'test_lrms_images', epoch)
                    test_img_save(img_pan, 'test_pan_images', epoch)

        print("===>Epoch{} Avg.test.loss: {:.10f} ".format(epoch, avg_test_loss / len(testset_dataloader)))
        test_loss_record.write(
            "Epoch{} Avg.test.loss: {:.10f}\n".format(epoch, avg_test_loss / len(testset_dataloader)))
        test_loss_record.close()

def eval(model, testset_dataloader):
    model.eval()
    eval_loss_record = open('%s/eval_loss_record.txt' % record_dir, "w")
    with torch.no_grad():
        for k, data in enumerate(testset_dataloader):
            img_pan, img_lr, img_lr_u, target = data[0], data[1], data[2], data[3]
            lr_hp = get_edge(img_lr)
            pan_hp = get_edge(img_pan)
            # 把数据集中提取出来的数据放到device上去
            lr_hp = lr_hp.to(device)
            pan_hp = pan_hp.to(device)
            target = target.to(device)
            img_lr_u = img_lr_u.to(device)

            mrs = model(pan_hp, lr_hp)  # 网络输出
            eval_fused_images = torch.add(mrs, img_lr_u)
            eval_loss = criterion(eval_fused_images, target)

            # 损失函数
            print("===>Batch:{} Eval.loss: {:.10f} ".format(k+1, eval_loss.item()))
            eval_loss_record.write("Batch:{} Eval.loss: {:.10f}\n".format(k+1, eval_loss.item()))

            # 保存融合图像
            print('==>Save the fused_images')
            eval_fused_images, real_images = eval_fused_images.cpu(), target.cpu()
            eval_img_save(eval_fused_images, 'eval_fused_images', k)
            eval_img_save(real_images, 'real_images', k)

    eval_loss_record.close()

def main():
    # 如果有保存的模型，则加载模型，并在其基础上继续训练
    if os.path.exists(checkpoint_model):
        print("==> loading checkpoint '{}'".format(checkpoint_model))
        checkpoint = torch.load(checkpoint_model)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        print('==> 加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('==> 无保存模型，将从头开始训练！')

    train(model, trainset_dataloader, start_epoch)

    eval(model, testset_dataloader)

if __name__ == '__main__':
    main()

