import torch.nn as nn
import torch
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.checkpoint import checkpoint_sequential
import time
import cv2
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.metrics import jaccard_similarity_score as jsc
import sklearn

def display_image(out, label):
      _,pred = torch.max(out.data,1)
      pred = pred.to(torch.device("cpu"))
      pred = pred.detach()
      img_grid = pred[0]*255
      #img_grid = torchvision.utils.make_grid(out)      
      plt.imshow(img_grid)
      plt.show()
      label = label.to(torch.device("cpu"))
      label = label.detach()
      label = label[0]*255
      plt.imshow(label)
      plt.show()

def iou(pred, target, n_classes =5):
      _,pred =torch.max(pred.data, 1) 
      pred = pred.to(torch.device("cpu"))
      ious = []
      pred = pred.view(-1)
      target = target.view(-1)

  # Ignore IoU for background class ("0")
      for cls in range(n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        pred_inds = pred_inds.numpy()
        target_inds = target_inds.numpy()
        intersection = (pred_inds[target_inds]).long().sum().data.cpu()[0]  # Cast to long to prevent overflows
        union = pred_inds.long().sum().data.cpu()[0] + target_inds.long().sum().data.cpu()[0] - intersection
        if union == 0:
          ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
          ious.append(float(intersection) / float(max(union, 1)))
      return np.array(ious)
      

def confusion(prediction, truth):
      _,prediction =torch.max(prediction.data, 1)
      prediction = prediction.to(torch.device("cpu"))
      truth = truth.to(torch.device("cpu"))          
      truth = truth.type(torch.DoubleTensor)
      prediction = prediction.type(torch.DoubleTensor)  
      confusion_vector = prediction / truth
      #print(confusion_vector)
      true_positives = torch.sum(confusion_vector == 1).item()
      false_positives = torch.sum(confusion_vector == float('inf')).item()
      true_negatives = torch.sum(torch.isnan(confusion_vector)).item()
      false_negatives = torch.sum(confusion_vector == 0).item()
      
      if (true_positives + false_negatives)!= 0: 
        recal = true_positives /(true_positives + false_negatives)
      else:
        recal = 0  
      if (true_positives + false_positives)!= 0:
        precision = true_positives /(true_positives + false_positives)
      else:
        precision = 0 
      if (precision + recal)!= 0:
        F_one = (2 * precision *recal) / (precision + recal)
      else:
        F_one = 0 

      if (true_positives + false_negatives + false_positives) != 0: 
        IOU = true_positives / (true_positives + false_negatives + false_positives)
      else : 
        IOU = 0 
      #print(true_positives, false_positives, true_negatives, false_negatives)
      return recal, precision, F_one, IOU

def accuracy(output, target, total_train = 0, correct_train = 0):
      _, predicted = torch.max(output.data, 1)
      total_train = predicted.nelement()
      correct_train += predicted.eq(target.data).sum().item() 
      return(correct_train*100/ total_train)


images_list = []
labels_list = []
images = []
labels = []

images_list_val =[]
labels_list_val = []
images_val = []
labels_val = []

images_list_test = []
labels_list_test = []
images_test = []
labels_test = []

def getFiles(path):
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img = os.path.join(path,file)
            #img = cv2.imread(os.path.join(path,file))
            if img is not None:
                images_list.append(img)


def getFile(path):
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = os.path.join(path,file)
            #img = cv2.imread(os.path.join(path,file))
            if img is not None:
                labels_list.append(img)
                
filesPath = "/content/drive/My Drive/Crack500/traincrop"
filepath = "/content/drive/My Drive/Crack500/traincrop_label"

getFiles(filesPath)
getFile(filepath)

print(len(images_list))
print(len(labels_list))


images_list_final = []
labels_list_final = [] 
for i in range(1896):
    image_finder = images_list[i][43:-3]
    for k in range(1896):
        label_finder = labels_list[k][49:-3]
        if image_finder == label_finder:
            labels_list_final.append(labels_list[k])
            images_list_final.append(images_list[i])
count = 0 
for i in range(1896):
    image_new = images_list_final[i][43:-3]
    label_new = labels_list_final[i][49:-3]
    if image_new == label_new:
        count = count +1
print(count)        
for i in range(1896):
    images.append(cv2.imread(images_list_final[i]))
    labels.append(cv2.imread(labels_list_final[i]))
    print(i)


def getFiles_val(path):
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img = os.path.join(path,file)
            #img = cv2.imread(os.path.join(path,file))
            if img is not None:
                images_list_val.append(img)


def getFile_val(path):
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = os.path.join(path,file)
            #img = cv2.imread(os.path.join(path,file))
            if img is not None:
                labels_list_val.append(img)
                
filesPath = "/content/drive/My Drive/Crack500/valcrop"
filepath = "/content/drive/My Drive/Crack500/valcrop_label"

getFiles_val(filesPath)
getFile_val(filepath)

print(len(images_list_val))
print(len(labels_list_val))


images_list_final_val = []
labels_list_final_val = [] 

for i in range(348):
    image_finder = images_list_val[i][41:-3]
    for k in range(348):
        label_finder = labels_list_val[k][47:-3]
        if image_finder == label_finder:
            labels_list_final_val.append(labels_list_val[k])
            images_list_final_val.append(images_list_val[i])
count = 0 
for i in range(348):
    image_new = images_list_final_val[i][41:-3]
    label_new = labels_list_final_val[i][47:-3]
    if image_new == label_new:
        count = count +1
print(count)        
for i in range(348):
    images_val.append(cv2.imread(images_list_final_val[i]))
    labels_val.append(cv2.imread(labels_list_final_val[i]))
    print(i)


def getFiles_test(path):
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            img = os.path.join(path,file)
            #img = cv2.imread(os.path.join(path,file))
            if img is not None:
                images_list_test.append(img)


def getFile_test(path):
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = os.path.join(path,file)
            #img = cv2.imread(os.path.join(path,file))
            if img is not None:
                labels_list_test.append(img)
                
filesPath = "/content/drive/My Drive/Crack500/testcrop"
filepath = "/content/drive/My Drive/Crack500/testcrop_label"

getFiles_test(filesPath)
getFile_test(filepath)

print(len(images_list_test))
print(len(labels_list_test))


images_list_final_test = []
labels_list_final_test = [] 

for i in range(1124):
    image_finder = images_list_test[i][42:-3]
    for k in range(1124):
        label_finder = labels_list_test[k][48:-3]
        if image_finder == label_finder:
            labels_list_final_test.append(labels_list_test[k])
            images_list_final_test.append(images_list_test[i])
count = 0 
for i in range(1124):
    image_new = images_list_final_test[i][42:-3]
    label_new = labels_list_final_test[i][48:-3]
    if image_new == label_new:
        count = count +1
print(count)        
for i in range(1124):
    images_test.append(cv2.imread(images_list_final_test[i]))
    labels_test.append(cv2.imread(labels_list_final_test[i]))
    

class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'MobileNetv2_DeepLabv3'
        self.dataset = 'cityscapes'
        self.s = [2, 1, 2, 2, 2, 1, 1]  # stride of each conv stage
        self.t = [1, 1, 6, 6, 6, 6, 6]  # expansion factor t
        self.n = [1, 1, 2, 3, 4, 3, 3]  # number of repeat time
        self.c = [32, 16, 24, 32, 64, 96, 160]  # output channel of each conv stage
        self.output_stride = 16
        self.multi_grid = (1, 2, 4)
        self.aspp = (6, 12, 18)
        self.down_sample_rate = 32  # classic down sample rate

        # dataset parameters
        self.rescale_size = 600
        self.image_size = 512
        self.num_class = 5  # 20 classes for training
        self.dataset_root = '/content/drive/My Drive/Crack500'
        self.dataloader_workers = 0
        self.shuffle = True
        self.train_batch = 8
        self.val_batch = 2
        self.test_batch = 1

        # train parameters
        self.num_epoch = 150
        self.base_lr = 0.0002
        self.power = 0.9
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.should_val = True
        self.val_every = 2
        self.display = 1  # show train result every display epoch
        self.should_split = True  # should split training procedure into several parts
        self.split = 2  # number of split

        # model restore parameters
        self.resume_from = None  # None for train from scratch
        self.pre_trained_from = None  # None for train from scratch
        self.should_save = True
        self.save_every = 10

        # create training dir
        self.summary_dir, self.ckpt_dir = create_train_dir(self)

def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (np.std(dataset + ep, axis=axis) / 255.0).tolist()


def create_train_dir(params):
    """
    Create folder used in training, folder hierarchy:
    current folder--exp_folder
                   |
                   --summaries
                   --checkpoints
    """
    experiment = params.model + '_' + params.dataset
    exp_dir = os.path.join(os.getcwd(), experiment)
    summary_dir = os.path.join(exp_dir, 'summaries/')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')

    dir = [exp_dir, summary_dir, checkpoint_dir]
    for dir_ in dir:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    return summary_dir, checkpoint_dir





class Cityscapes(Dataset):
    def __init__(self, dataset_dir, mode='train', transforms=None):
        """
        Create Dataset subclass on cityscapes dataset
        :param dataset_dir: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
        :param mode: phase, 'train', 'test' or 'eval'
        :param transforms: transformation
        """
        self.dataset = dataset_dir
        self.transforms = transforms
        require_file = ['trainImages.txt', 'trainLabels.txt',
                        'valImages.txt',   'valLabels.txt',
                        'testImages.txt',  'testLabels.txt']

        # check requirement
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Unsupported mode %s' % mode)

        if not os.path.exists(self.dataset):
            raise ValueError('Dataset not exists at %s' % self.dataset)

        
                #generate_txt(self.dataset, file)

        # create image and label list
#        self.image_list = []
#        self.label_list = []
        if mode == 'train':
          self.images = images
          self.labels = labels
          self.images_list_final = images_list_final
          self.labels_list_final = labels_list_final
        
        if mode == 'val':
          self.images = images_val
          self.labels = labels_val
          self.images_list_final = images_list_final_val
          self.labels_list_final = labels_list_final_val
        
        if mode == 'test': 
          self.images = images_test
          self.labels = labels_test
          self.images_list_final = images_list_final_test
          self.labels_list_final = labels_list_final_test

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Overrides default method
        tips: 3 channels of label image are the same
        """
        image = self.images[index]
        label = self.labels[index]  # label.size (1024, 2048, 3)
        image_name = self.images_list_final[index]
        label_name = self.labels_list_final[index]
        if label.min() == -1:
            raise ValueError

        sample = {'image': image, 'label': label[:, :, 0],
                  'image_name': image_name, 'label_name': label_name}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


class Rescale(object):
    """
    Rescale the image in a sample to a given size.
    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        sample['image'], sample['label'] = image, label

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, output_stride=16):
        self.output_stride = output_stride

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        # reset label shape
        # w, h = label.shape[0]//self.output_stride, label.shape[1]//self.output_stride
        # label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        # label[label == 255] = 19
        label = label
        label = label.astype(np.int64)

        # normalize image
        image /= 255
        sample['image'], sample['label'] = torch.from_numpy(image), torch.from_numpy(label)

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __call__(self, sample, p=0.5):
        image, label = sample['image'], sample['label']
        if np.random.uniform(0, 1) < p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        sample['image'], sample['label'] = image, label

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w, :]

        label = label[top: top + new_h, left: left + new_w]

        sample['image'], sample['label'] = image, label

        return sample


def print_config(params):
    for name, value in sorted(vars(params).items()):
        print('\t%-20s:%s' % (name, str(value)))
    print('')




class bar(object):
    def __init__(self):
        self.start_time = None
        self.iter_per_sec = 0
        self.time = None

    def click(self, current_idx, max_idx, total_length=40):
        """
        Each click is a draw procedure of progressbar
        :param current_idx: range from 0 to max_idx-1
        :param max_idx: maximum iteration
        :param total_length: length of progressbar
        """
        if self.start_time is None:
            self.start_time = time.time()
        else:
            self.time = time.time()-self.start_time
            self.iter_per_sec = 1/self.time
            perc = current_idx * total_length // max_idx
            # print progress bar
            print('\r|'+'='*perc+'>'+' '*(total_length-1-perc)+'| %d/%d (%.2f iter/s)' % (current_idx+1,
                                                                                          max_idx,
                                                                                          self.iter_per_sec), end='')
            self.start_time = time.time()

    def close(self):
        self.__init__()
        print('')



class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, s=1, dilation=1):
        """
        Initialization of inverted residual block
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param t: the expansion factor of block
        :param s: stride of the first convolution
        :param dilation: dilation rate of 3*3 depthwise conv
        """
        super(InvertedResidual, self).__init__()

        self.in_ = in_channels
        self.out_ = out_channels
        self.t = t
        self.s = s
        self.dilation = dilation
        self.inverted_residual_block()

    def inverted_residual_block(self):
        """
        Build Inverted Residual Block and residual connection
        """
        block = []
        # pad = 1 if self.s == 3 else 0
        # conv 1*1
        block.append(nn.Conv2d(self.in_, self.in_*self.t, 1, bias=False))
        block.append(nn.BatchNorm2d(self.in_*self.t))
        block.append(nn.ReLU6())

        # conv 3*3 depthwise
        block.append(nn.Conv2d(self.in_*self.t, self.in_*self.t, 3,
                               stride=self.s, padding=self.dilation, groups=self.in_*self.t, dilation=self.dilation,
                               bias=False))
        block.append(nn.BatchNorm2d(self.in_*self.t))
        block.append(nn.ReLU6())

        # conv 1*1 linear
        block.append(nn.Conv2d(self.in_*self.t, self.out_, 1, bias=False))
        block.append(nn.BatchNorm2d(self.out_))

        self.block = nn.Sequential(*block)

        # if use conv residual connection
        if self.in_ != self.out_ and self.s != 2:
            self.res_conv = nn.Sequential(nn.Conv2d(self.in_, self.out_, 1, bias=False),
                                          nn.BatchNorm2d(self.out_))
        else:
            self.res_conv = None

    def forward(self, x):
        if self.s == 1:
            # use residual connection
            if self.res_conv is None:
                out = x + self.block(x)
            else:
                out = self.res_conv(x) + self.block(x)
        else:
            # plain block
            out = self.block(x)

        return out


def get_inverted_residual_block_arr(in_, out_, t=6, s=1, n=1):
    block = []
    block.append(InvertedResidual(in_, out_, t, s=s))
    for i in range(n-1):
        block.append(InvertedResidual(out_, out_, t, 1))
    return block


class ASPP_plus(nn.Module):
    def __init__(self, params):
        super(ASPP_plus, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 1, bias=False),
                                     nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[0], dilation=params.aspp[0], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[1], dilation=params.aspp[1], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[2], dilation=params.aspp[2], bias=False),
                                      nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(256*5, 256, 1, bias=False),
                                      nn.BatchNorm2d(256))
        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)
    def forward(self, x):
        conv11 = self.conv11(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)

        # image pool and upsample
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        image_pool = self.conv11(image_pool)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = upsample(image_pool)

        # concate
        concate = torch.cat([conv11, conv33_1, conv33_2, conv33_3, upsample], dim=1)

        return self.concate_conv(concate)
    
WARNING = lambda x: print('\033[1;31;2mWARNING: ' + x + '\033[0m')
LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')

# create model
class MobileNetv2_DeepLabv3(nn.Module):
    """
    A Convolutional Neural Network with MobileNet v2 backbone and DeepLab v3 head
        used for Semantic Segmentation on Cityscapes dataset
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, params):
        super(MobileNetv2_DeepLabv3, self).__init__()
        self.params = params
        #self.datasets = datasets
        self.pb = bar()  # hand-made progressbar
        self.epoch = 0
        self.init_epoch = 0
        self.ckpt_flag = False
        self.train_loss = []
        self.val_loss = []
        self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)

        # build network
        block = []

        # conv layer 1
        block.append(nn.Sequential(nn.Conv2d(3, self.params.c[0], 3, stride=self.params.s[0], padding=1, bias=False),
                                   nn.BatchNorm2d(self.params.c[0]),
                                   # nn.Dropout2d(self.params.dropout_prob, inplace=True),
                                   nn.ReLU6()))

        # conv layer 2-7
        for i in range(6):
            block.extend(get_inverted_residual_block_arr(self.params.c[i], self.params.c[i+1],
                                                                t=self.params.t[i+1], s=self.params.s[i+1],
                                                                n=self.params.n[i+1]))

        # dilated conv layer 1-4
        # first dilation=rate, follows dilation=multi_grid*rate
        rate = self.params.down_sample_rate // self.params.output_stride
        block.append(InvertedResidual(self.params.c[6], self.params.c[6],
                                             t=self.params.t[6], s=1, dilation=rate))
        for i in range(3):
            block.append(InvertedResidual(self.params.c[6], self.params.c[6],
                                                 t=self.params.t[6], s=1, dilation=rate*self.params.multi_grid[i]))

        # ASPP layer
        block.append(ASPP_plus(self.params))

        # final conv layer
        block.append(nn.Conv2d(256, self.params.num_class, 1))

        # bilinear upsample
        block.append(nn.Upsample(scale_factor=self.params.output_stride, mode='bilinear', align_corners=False))

        self.network = nn.Sequential(*block).cuda()
        # print(self.network)

        # build loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # optimizer
        self.opt = torch.optim.RMSprop(self.network.parameters(),
                                       lr=self.params.base_lr,
                                       momentum=self.params.momentum,
                                       weight_decay=self.params.weight_decay)

        # initialize
        self.initialize()

        # load data
        self.load_checkpoint()
        self.load_model()

    """######################"""
    """# Train and Validate #"""
    """######################"""
    
    
    def train_one_epoch(self):
        """
        Train network in one epoch
        """
        print('Training......')

        # set mode train
        self.network.train()

        # prepare data
        train_loss = 0
        transform = transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              RandomHorizontalFlip(),
                                              ToTensor()
                                              ])



        dataset = Cityscapes(params.dataset_root, mode='train', transforms = transform)

        train_loader = DataLoader(dataset,
                                  batch_size=params.train_batch,
                                  shuffle=params.shuffle,
                                  num_workers=params.dataloader_workers)
        
        train_size = 1896
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
        else:
            total_batch = train_size // self.params.train_batch
        recal = 0
        precision = 0
        F_one = 0
        IOU = 0
        accuracy_new = 0 
        # train through dataset
        for batch_idx, batch in enumerate(train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.network(image_cuda)


            loss = self.loss_fn(out, label_cuda)
            
            #display_image(out, label_cuda)
            TP, FP, TN, FN = confusion(out, label_cuda)
            recal = recal+TP
            precision = precision+FP
            F_one = F_one + TN
            IOU = IOU+ FN 
            accuracy_final = accuracy(out, label_cuda)
            accuracy_new = accuracy_new + accuracy_final

            # optimize
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # accumulate
            train_loss += loss.item()

            # record first loss
            if self.train_loss == []:
                self.train_loss.append(train_loss)
                self.summary_writer.add_scalar('loss/train_loss', train_loss, 0)
        
        print("\t")
        print(recal/total_batch, precision/ total_batch, F_one/ total_batch, IOU/ total_batch)
        print(accuracy_new/total_batch)
        
        self.pb.close()
        train_loss /= total_batch
        self.train_loss.append(train_loss)

        # add to summary
        self.summary_writer.add_scalar('loss/train_loss', train_loss, self.epoch)

    

    def train_one_epoch_Image_display(self):
        """
        Train network in one epoch
        """
        print('Training......')

        # set mode train
        self.network.train()

        # prepare data
        train_loss = 0
        transform = transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              RandomHorizontalFlip(),
                                              ToTensor()
                                              ])



        dataset = Cityscapes(params.dataset_root, mode='train', transforms = transform)

        train_loader = DataLoader(dataset,
                                  batch_size=params.train_batch,
                                  shuffle=params.shuffle,
                                  num_workers=params.dataloader_workers)
        
        train_size = 1896
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
        else:
            total_batch = train_size // self.params.train_batch
        recal = 0
        precision = 0
        F_one = 0
        IOU = 0
        accuracy_new = 0 
        # train through dataset
        for batch_idx, batch in enumerate(train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.network(image_cuda)
            
            pred = image_cuda
            pred = pred.to(torch.device("cpu"))
            pred = pred.detach()
            img_grid = pred[0]
            #img_grid = torchvision.utils.make_grid(out)      
            img_grid = img_grid.numpy().transpose(1, 2, 0)*255
            cv2.imwrite("/content/drive/My Drive/Train_images/original%d.jpg" % batch_idx, img_grid)

            
            loss = self.loss_fn(out, label_cuda)
            
            #display_image(out, label_cuda)
            TP, FP, TN, FN = confusion(out, label_cuda)
            recal = recal+TP
            precision = precision+FP
            F_one = F_one + TN
            IOU = IOU+ FN 
            _,predict = torch.max(out.data,1)
            predict = predict.to(torch.device("cpu"))
            predict = predict.detach()
            img = predict[0]
            img = img.numpy()*255
            #img_grid = torchvision.utils.make_grid(out)      
            cv2.imwrite("/content/drive/My Drive/Train_images/predict_label%d.png" % batch_idx, img)
            label = label_cuda.to(torch.device("cpu"))
            label = label.detach()
            label = label[0].numpy()*255
            cv2.imwrite("/content/drive/My Drive/Train_images/original_label%d.png" % batch_idx, label)

            
            

            accuracy_final = accuracy(out, label_cuda)
            accuracy_new = accuracy_new + accuracy_final

        
        print("\t")
        print(recal/total_batch, precision/ total_batch, F_one/ total_batch, IOU/ total_batch)
        print(accuracy_new/total_batch)

        
    def Train(self):
        """
        Train network in n epochs, n is defined in params.num_epoch
        """
        self.init_epoch = self.epoch
        if self.epoch >= self.params.num_epoch:
            WARNING('Num_epoch should be smaller than current epoch. Skip training......\n')
        else:
            for _ in range(self.epoch, self.params.num_epoch):
                self.epoch += 1
                print('-' * 20 + 'Epoch.' + str(self.epoch) + '-' * 20)

                # train one epoch
                self.train_one_epoch()

                # should display
                if self.epoch % self.params.display == 0:
                    print('\tTrain loss: %.4f' % self.train_loss[-1])

                # should save
                if self.params.should_save:
                    if self.epoch % self.params.save_every == 0:
                        self.save_checkpoint()

                # test every params.test_every epoch
                if self.params.should_val:
                    if self.epoch % self.params.val_every == 0:
                        self.val_one_epoch()
                        print('\tVal loss: %.4f' % self.val_loss[-1])

                # adjust learning rate
                self.adjust_lr()
            self.train_one_epoch_Image_display()    
                   
            # save the last network state
            if self.params.should_save:
                self.save_checkpoint()

            # train visualization
            self.plot_curve()


    def val_one_epoch(self):
        """
        Validate network in one epoch every m training epochs,
            m is defined in params.val_every
        """
        # TODO: add IoU compute function
        print('Validating:')

        # set mode eval
        self.network.eval()

        # prepare data
        val_loss = 0
        transform = transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              
                                              ToTensor()
                                              ])
        dataset = Cityscapes(params.dataset_root, mode='val', transforms = transform)
        val_loader = DataLoader(dataset,
                                batch_size=params.val_batch,
                                shuffle=params.shuffle,
                                num_workers=params.dataloader_workers)
        val_size = 348
        if val_size % self.params.val_batch != 0:
            total_batch = val_size // self.params.val_batch + 1
        else:
            total_batch = val_size // self.params.val_batch
        recal = 0
        precision = 0
        F_one = 0
        IOU = 0
        accuracy_new = 0  
        # validate through dataset
        for batch_idx, batch in enumerate(val_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.network(image_cuda)
            
            TP, FP, TN, FN = confusion(out, label_cuda)
            recal = recal+TP
            precision = precision+FP
            F_one = F_one +TN
            IOU = IOU+ FN 
            accuracy_final = accuracy(out, label_cuda)
            accuracy_new = accuracy_new +accuracy_final
            loss = self.loss_fn(out, label_cuda)
            val_loss += loss.item()

            # record first loss
            if self.val_loss == []:
                self.val_loss.append(val_loss)
                self.summary_writer.add_scalar('loss/val_loss', val_loss, 0)
        
        print(accuracy_new/total_batch)
        print("\t")
        print(recal/total_batch, precision/ total_batch, F_one/ total_batch, IOU/ total_batch)
        self.pb.close()
        val_loss /= total_batch
        self.val_loss.append(val_loss)

        # add to summary
        self.summary_writer.add_scalar('loss/val_loss', val_loss, self.epoch)
    
    def Test(self):
        """
        Test network on test set
        """
        print('Testing:')
        # set mode eval
        torch.cuda.empty_cache()
        self.network.eval()
        transform = transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              
                                              ToTensor()
                                              ])
        dataset = Cityscapes(params.dataset_root, mode='test', transforms = transform)
        test_loader = DataLoader(dataset,
                                batch_size=params.test_batch,
                                shuffle=params.shuffle,
                                num_workers=params.dataloader_workers)
        # prepare test data
        recal = 0
        precision = 0
        F_one = 0
        IOU = 0
        accuracy_new = 0
        test_size = 1124
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch

        # test for one epoch
        for batch_idx, batch in enumerate(test_loader):
            self.pb.click(batch_idx, total_batch)
            image, label, name = batch['image'], batch['label'], batch['label_name']
            image_cuda, label_cuda = image.cuda(), label.cuda()
            pred = image_cuda
            pred = pred.to(torch.device("cpu"))
            pred = pred.detach()
            img_grid = pred[0]
            #img_grid = torchvision.utils.make_grid(out)      
            img_grid = img_grid.numpy().transpose(1, 2, 0)*255
            cv2.imwrite("/content/drive/My Drive/Train_images/original%d.jpg" % batch_idx, img_grid)
            if self.params.should_split:
                image_cuda.requires_grad_()
                out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            else:
                out = self.network(image_cuda)
            TP, FP, TN, FN = confusion(out, label_cuda)
            recal = recal+TP
            precision = precision+FP
            F_one = F_one +TN
            IOU = IOU+ FN 
            _,predict = torch.max(out.data,1)
            predict = predict.to(torch.device("cpu"))
            predict = predict.detach()
            img = predict[0]
            img = img.numpy()*255
            #img_grid = torchvision.utils.make_grid(out)      
            cv2.imwrite("/content/drive/My Drive/Test_images/predict_label%d.png" % batch_idx, img)
            label = label_cuda.to(torch.device("cpu"))
            label = label.detach()
            label = label[0].numpy()*255
            cv2.imwrite("/content/drive/My Drive/Test_images/original_label%d.png" % batch_idx, label)

            accuracy_final = accuracy(out, label_cuda)
            accuracy_new = accuracy_new + accuracy_final
        print("\t")
        print(recal/total_batch, precision/ total_batch, F_one/ total_batch, IOU/ total_batch)
        print("\t")
        print(accuracy_new/total_batch)
    def save_checkpoint(self):
        
        save_dict = {'epoch'        :  self.epoch,
                     'train_loss'   :  self.train_loss,
                     'val_loss'     :  self.val_loss,
                     'state_dict'   :  self.network.state_dict(),
                     'optimizer'    :  self.opt.state_dict()}
        torch.save(save_dict, self.params.ckpt_dir+'Checkpoint_epoch_%d.pth.tar' % self.epoch)
        print('Checkpoint saved')

    def load_checkpoint(self):
        """
        Load checkpoint from given path
        """
        if self.params.resume_from is not None and os.path.exists(self.params.resume_from):
            try:
                LOG('Loading Checkpoint at %s' % self.params.resume_from)
                ckpt = torch.load(self.params.resume_from)
                self.epoch = ckpt['epoch']
                try:
                    self.train_loss = ckpt['train_loss']
                    self.val_loss = ckpt['val_loss']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.network.load_state_dict(ckpt['state_dict'])
                self.opt.load_state_dict(ckpt['optimizer'])
                LOG('Checkpoint Loaded!')
                LOG('Current Epoch: %d' % self.epoch)
                self.ckpt_flag = True
            except:
                WARNING('Cannot load checkpoint from %s. Start loading pre-trained model......' % self.params.resume_from)
        else:
            WARNING('Checkpoint do not exists. Start loading pre-trained model......')

    def load_model(self):
        """
        Load ImageNet pre-trained model into MobileNetv2 backbone, only happen when
            no checkpoint is loaded
        """
        if self.ckpt_flag:
            LOG('Skip Loading Pre-trained Model......')
        else:
            if self.params.pre_trained_from is not None and os.path.exists(self.params.pre_trained_from):
                try:
                    LOG('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                    pretrain = torch.load(self.params.pre_trained_from)
                    self.network.load_state_dict(pretrain)
                    LOG('Pre-trained Model Loaded!')
                except:
                    WARNING('Cannot load pre-trained model. Start training......')
            else:
                WARNING('Pre-trained model do not exits. Start training......')

    """#############"""
    """# Utilities #"""
    """#############"""

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * (1 - float(self.epoch) / self.params.num_epoch) ** self.params.power
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        print('Change learning rate into %f' % (learning_rate))
        self.summary_writer.add_scalar('learning_rate', learning_rate, self.epoch)

    def plot_curve(self):
        """
        Plot train/val loss curve
        """
        x1 = np.arange(self.init_epoch, self.params.num_epoch+1, dtype=np.int).tolist()
        x2 = np.linspace(self.init_epoch, self.epoch,
                         num=(self.epoch-self.init_epoch)//self.params.val_every+1, dtype=np.int64)
        plt.plot(x1, self.train_loss, label='train_loss')
        plt.plot(x2, self.val_loss, label='val_loss')
        plt.legend(loc='best')
        plt.title('Train/Val loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


params = Params()
print('Network parameters:')
print_config(params)
print('Initializing MobileNet and DeepLab......')
net = MobileNetv2_DeepLabv3(params)
print('Model Built.\n')
print("Start Training")
net.Train()
print("Start testing")
net.Test()