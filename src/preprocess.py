import argparse
import math
import os
import pickle
import random
import sys

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision import transforms, datasets

from src.model.CrossEntropy import CrossEntropy_L2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Unified parameter adjustment
def give_parser():
    parser = argparse.ArgumentParser(description='MIA Evaluation Platform')
    # Static variable
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='Decide which dataset to use')
    parser.add_argument('--split_size', type=float, default=0.25, help='Data segmentation')
    parser.add_argument(
        '--datapath', type=str, default='./dataset', help='Determine where the dataset is stored'
    )
    parser.add_argument(
        '--dataloadpath', type=str, default='./dataset/Preprocessed',
        help='Determine where the preprocessed dataset is stored'
    )
    parser.add_argument(
        '--keepmodelpath', type=str, default='model/ModelStore/',
        help='Determine where the preprocessed dataset is stored'
    )
    parser.add_argument('--model', type=str, default='CNN', help='Determine the target model to use')
    # parser.add_argument('--model', type=str, default='ResNet18', help='Determine the target model to use')
    # parser.add_argument('--model', type=str, default='DenseNet169', help='Determine the target model to use')
    # parser.add_argument('--model', type=str, default='VGG11', help='Determine the target model to use')
    # parser.add_argument('--model', type=str, default='AlexNet', help='Determine the target model to use')
    parser.add_argument(
        '--attack_model', type=str, default='Softmax',
        help='Determine the attack model to use'
    )
    # init of model
    parser.add_argument(
        '--channel', type=int, default=3
    )
    parser.add_argument(
        '--in_features', type=int, default=1152
    )  # cifar 10
    parser.add_argument(
        '--out_features', type=int, default=128
    )  # cifar 10
    # parser.add_argument(
    #     '--in_features', type=int, default=800
    # )  # MNIST
    # parser.add_argument(
    #     '--out_features', type=int, default=32
    # )  # MNIST
    parser.add_argument(
        '--n_out', type=int, default=10
    )

    # init of dataset
    parser.add_argument(
        '--cluster', type=int, default=10520
    )

    # Training needs
    parser.add_argument(
        '--device', default='cuda', type=str,
        help='Determine the operation core of data execution'
    )
    parser.add_argument(
        '--batch_size', default=100, type=int,
        help='Determine training batch size'
    )
    parser.add_argument(
        '--epochs', default=50, type=int,
        help='Determine training epochs'
    )
    parser.add_argument(
        '--learning_rate', default=0.001, type=float,
        help='Determine training learning_rate'
    )
    parser.add_argument(
        '--l2_ratio', default=1e-7
    )
    # MIA need
    parser.add_argument(
        '--at_batch_size', default=10, type=int,
        help='Determine training batch size'
    )
    parser.add_argument(
        '--at_epochs', default=50, type=int,
        help='Determine training epochs'
    )
    parser.add_argument(
        '--at_learning_rate', default=0.01, type=float,
        help='Determine training learning_rate'
    )
    parser.add_argument(
        '--at_l2_ratio', default=1e-7
    )

    opt = parser.parse_args()
    return opt


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


# 高斯模糊
def give_gaussian(img, p=0.1):
    do_it = random.random() <= p
    if do_it:
        return img
    else:
        # 高斯增强
        def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
            # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
            x_coord = torch.arange(kernel_size)
            x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
            y_grid = x_grid.t()
            xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

            mean = (kernel_size - 1) / 2.
            variance = sigma ** 2.

            # Calculate the 2-dimensional gaussian kernel which is
            # the product of two gaussian distributions for two different
            # variables (in this case called x and y)
            gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                              torch.exp(
                                  -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                                  (2 * variance)
                              )

            # Make sure sum of values in gaussian kernel equals 1.
            gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

            # Reshape to 2d depthwise convolutional weight
            gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
            gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

            gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                        groups=channels,
                                        bias=False, padding=kernel_size // 2)

            gaussian_filter.weight.data = gaussian_kernel
            gaussian_filter.weight.requires_grad = False

            return gaussian_filter

        blur_layer = get_gaussian_kernel().cuda()
        return blur_layer(img)


# 随机增加对比度
def give_random_contrast(image, p=0.3, lower=0.7, upper=1.3):
    if random.random() < p:
        alpha = random.uniform(lower, upper)
        image *= alpha
        image = image.clip(min=0, max=255)
    return image


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + '/log.txt', 'w')
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


class GiveTestData():
    def __init__(self, opt):
        # Initialize static data
        self.opt = opt  # init flag

    # Read data generation, input and label
    def readData(self, dataName, dataPath):
        '''
        :param dataName: str, name of dataset
        :param dataPath: str, path of dataset
        :return: ndarrary
        '''
        if dataName == 'MINST':
            data_path = dataPath + '/processed'
            train_data_dict = torch.load(data_path + '/training.pt')
            X, y = train_data_dict[0].numpy(), train_data_dict[1].numpy()

            for index in range(len(X)):
                X[index] = X[index].transpose((1, 0))

            X = X.reshape(X.shape[0], -1)

            test_data_dict = torch.load(data_path + '/test.pt')
            XTest, yTest = test_data_dict[0].numpy(), test_data_dict[1].numpy()

            for index in range(len(XTest)):
                XTest[index] = XTest[index].transpose((1, 0))

            XTest = XTest.reshape(XTest.shape[0], -1)

            return X, y, XTest, yTest

        elif dataName == 'CIFAR10':
            for i in range(5):
                f = open(dataPath + '/data_batch_' + str(i + 1), 'rb')
                train_data_dict = pickle.load(f, encoding='iso-8859-1')

                f.close()
                if i == 0:
                    X = train_data_dict["data"]
                    y = train_data_dict["labels"]
                    continue

                X = np.concatenate((X, train_data_dict["data"]), axis=0)
                y = np.concatenate((y, train_data_dict["labels"]), axis=0)

            f = open(dataPath + '/test_batch', 'rb')
            test_data_dict = pickle.load(f, encoding='iso-8859-1')
            f.close()

            XTest = np.array(test_data_dict["data"])
            yTest = np.array(test_data_dict["labels"])
            return X, y, XTest, yTest

        elif dataName == 'CIFAR100':
            f = open(dataPath + '/train', 'rb')
            train_data_dict = pickle.load(f, encoding='iso-8859-1')
            f.close()

            X = train_data_dict['data']
            y = train_data_dict['fine_labels']

            f = open(dataPath + '/test', 'rb')
            test_data_dict = pickle.load(f, encoding='iso-8859-1')
            f.close()

            XTest = np.array(test_data_dict['data'])
            yTest = np.array(test_data_dict['fine_labels'])

            return X, y, XTest, yTest
        elif dataName == 'News':
            train = fetch_20newsgroups(
                data_home=dataPath,
                subset='train',
                remove=('headers', 'footers', 'quotes')
            )

            test = fetch_20newsgroups(
                data_home=dataPath,
                subset='test',
                remove=('headers', 'footers', 'quotes')
            )

            X = np.concatenate((train.data, test.data), axis=0)
            y = np.concatenate((train.target, test.target), axis=0)

            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(X)
            X = X.toarray()

            return X, y
        else:
            pass

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=True):
        '''
        :param inputs: ndarray,such as (100,1,28,28)
        :param targets: ndarray,such as (100,10)
        :param batch_size: int, batch size of train data
        :param shuffle: decide whether to disrupt the data
        :return:
        '''
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        start_idx = None
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

        if start_idx is not None and start_idx + batch_size < len(inputs):
            excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
            yield inputs[excerpt], targets[excerpt]

    # Randomly scramble data
    def shuffleAndSplitData(self, dataX, dataY, train_cluster, test_cluster, attack_cluster=0, cross=False):
        '''
        :param dataX: ndarrary
        :param dataY: ndarrary
        :param cluster: int
        :return:
        '''
        # Bind the dataset first and then sort it randomly
        c = list(zip(dataX, dataY))
        random.shuffle(c)
        dataX, dataY = zip(*c)
        # Target model training set
        toTrainData = np.array(dataX[:train_cluster])
        toTrainLabel = np.array(dataY[:train_cluster])
        # Target model test set
        toTestData = np.array(dataX[train_cluster:train_cluster + test_cluster])
        toTestLabel = np.array(dataY[train_cluster:train_cluster + test_cluster])
        if cross == False:
            # Shadow model training set
            shadowData = np.array(dataX[train_cluster + test_cluster:2 * train_cluster + test_cluster])
            shadowLabel = np.array(dataY[train_cluster + test_cluster:2 * train_cluster + test_cluster])
            # Shadow model testing set
            shadowTestData = np.array(dataX[2 * train_cluster + test_cluster:2 * train_cluster + 2 * test_cluster])
            shadowTestLabel = np.array(dataY[2 * train_cluster + test_cluster:2 * train_cluster + 2 * test_cluster])
        else:
            # Shadow model training set
            shadowData = np.array(toTrainData[:attack_cluster])
            shadowLabel = np.array(toTrainLabel[:attack_cluster])
            # Shadow model testing set
            shadowTestData = np.array(dataX[train_cluster + test_cluster:])
            shadowTestLabel = np.array(dataY[train_cluster + test_cluster:])

        return toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel

    def shuffleAndSplitData_for_GanNoise(self, dataX, dataY, train_cluster, test_cluster, attack_cluster=0,
                                         TrainTest_radio=0.2, cross=False):
        """
        :param dataX: ndarrary
        :param dataY: ndarrary
        :param cluster: int
        :return:
        """
        # Bind the dataset first and then sort it randomly
        c = list(zip(dataX, dataY))
        random.shuffle(c)
        dataX, dataY = zip(*c)
        # Target model training set
        toTrainData = np.array(dataX[:train_cluster])
        toTrainLabel = np.array(dataY[:train_cluster])
        # Target model test set
        toTestData = np.array(dataX[train_cluster:train_cluster + test_cluster])
        toTestLabel = np.array(dataY[train_cluster:train_cluster + test_cluster])
        if cross == False:
            # Shadow model training set
            shadowData = np.array(dataX[train_cluster + test_cluster:2 * train_cluster + test_cluster])
            shadowLabel = np.array(dataY[train_cluster + test_cluster:2 * train_cluster + test_cluster])
            # Shadow model testing set
            shadowTestData = np.array(dataX[2 * train_cluster + test_cluster:2 * train_cluster + 2 * test_cluster])
            shadowTestLabel = np.array(dataY[2 * train_cluster + test_cluster:2 * train_cluster + 2 * test_cluster])
        else:
            # Shadow model training set
            shadowData = np.array(toTrainData[:attack_cluster])
            shadowLabel = np.array(toTrainLabel[:attack_cluster])
            # Shadow model testing set
            shadowTestData = np.array(dataX[train_cluster + test_cluster:])
            shadowTestLabel = np.array(dataY[train_cluster + test_cluster:])

        # 从训练集中抛出GanNoise训练集中使用的非成员数据
        TrainCluster = int(len(toTrainData) * (1 - TrainTest_radio))
        toTrainData_Train = np.array(toTrainData[:TrainCluster])
        toTrainData_Test = np.array(toTrainData[TrainCluster:])
        toTrainLabel_Train = np.array(toTrainLabel[:TrainCluster])
        toTrainLabel_Test = np.array(toTrainLabel[TrainCluster:])
        TrainCluster = int(len(shadowData) * (1 - TrainTest_radio))
        shadowData_Train = np.array(shadowData[:TrainCluster])
        shadowData_Test = np.array(shadowData[TrainCluster:])
        shadowLabel_Train = np.array(shadowLabel[:TrainCluster])
        shadowLabel_Test = np.array(shadowLabel[TrainCluster:])
        return toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel

    # Data preprocessing
    def Preprocessing(self, dataName, toTrainData, toTestData):
        #:param toTrainData: ndarrary,train data
        #:param toTestData: ndarrary,test data
        #:return: ndarrary,Preprocessed data
        # 处理MNIST数据集
        if dataName == 'MINST':
            def reshape_for_save(raw_data):
                raw_data = raw_data.reshape(
                    (raw_data.shape[0], 28, 28, 1)
                ).transpose(0, 3, 1, 2)
                return raw_data.astype(np.float32)

            offset = np.mean(reshape_for_save(toTrainData), 0)
            scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

            def rescale(raw_data):
                return (reshape_for_save(raw_data) - offset) / scale

            return rescale(toTrainData), rescale(toTestData)
        # 主要用于处理cifar10 和 cifar100
        elif dataName == 'CIFAR':
            def reshape_for_save(raw_data):
                raw_data = np.dstack(
                    (raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:])
                )
                raw_data = raw_data.reshape(
                    (raw_data.shape[0], 32, 32, 3)
                ).transpose(0, 3, 1, 2)
                return raw_data.astype(np.float32)

            offset = np.mean(reshape_for_save(toTrainData), 0)
            scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

            def rescale(raw_data):
                return (reshape_for_save(raw_data) - offset) / scale

            return rescale(toTrainData), rescale(toTestData)
        # 处理News数据
        elif dataName == 'News':
            def normalizeData(X):
                offset = np.mean(X, 0)
                scale = np.std(X, 0).clip(min=1)
                X = (X - offset) / scale
                X = X.astype(np.float32)
                return X

            return normalizeData(toTrainData), normalizeData(toTestData)
        elif dataName == 'Purchase100' or dataName =='Texas100':
            return toTrainData,  toTestData
        else:
            print('There are not preprocessing of this dataset!')
            return toTrainData, toTestData

    def Preprocessing_for_GanNoise(self, dataName, toTrainData, toTrainDataT, toTestData):
        """
        :param toTrainData: ndarrary,train data
        :param toTestData: ndarrary,test data
        :return: ndarrary,Preprocessed data
        """
        # 处理MNIST数据集
        if dataName == 'MINST':
            def reshape_for_save(raw_data):
                raw_data = raw_data.reshape(
                    (raw_data.shape[0], 28, 28, 1)
                ).transpose(0, 3, 1, 2)
                return raw_data.astype(np.float32)

            offset = np.mean(reshape_for_save(toTrainData), 0)
            scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

            def rescale(raw_data):
                return (reshape_for_save(raw_data) - offset) / scale

            return rescale(toTrainData), rescale(toTrainDataT), rescale(toTestData)
        # 主要用于处理cifar10 和 cifar100
        elif dataName == 'CIFAR':
            def reshape_for_save(raw_data):
                raw_data = np.dstack(
                    (raw_data[:, :1024], raw_data[:, 1024:2048], raw_data[:, 2048:])
                )
                raw_data = raw_data.reshape(
                    (raw_data.shape[0], 32, 32, 3)
                ).transpose(0, 3, 1, 2)
                return raw_data.astype(np.float32)

            offset = np.mean(reshape_for_save(toTrainData), 0)
            scale = np.std(reshape_for_save(toTrainData), 0).clip(min=1)

            def rescale(raw_data):
                return (reshape_for_save(raw_data) - offset) / scale

            return rescale(toTrainData), rescale(toTrainDataT), rescale(toTestData)
        # 处理News数据
        elif dataName == 'News':
            def normalizeData(X):
                offset = np.mean(X, 0)
                scale = np.std(X, 0).clip(min=1)
                X = (X - offset) / scale
                X = X.astype(np.float32)
                return X

            return normalizeData(toTrainData), normalizeData(toTrainDataT), normalizeData(toTestData)
        elif dataName == 'Purchase100' or dataName =='Texas100':
            return toTrainData, toTrainDataT, toTestData
        else:
            raise ValueError('There are not preprocessing of this dataset!')

    # init data
    def initializeData(self, dataSet, orginialDatasetPath, trainCluster=10520, testCluster=10520, attackCluster=0,
                       dataFolderPath='./dataset/', cross=False):
        if dataSet == 'CIFAR10':
            print("Loading data")
            dataX, dataY, testX, testY = self.readData('CIFAR10', orginialDatasetPath)
            print("Preprocessing data")
            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                X = np.concatenate([dataX, testX], axis=0)
                Y = np.concatenate([dataY, testY], axis=0)
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, cross)
            else:
                X = np.concatenate([dataX, testX], axis=0)
                Y = np.concatenate([dataY, testY], axis=0)
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, attackCluster, cross)
            # 对数据进行预处理
            toTrainDataSave, toTestDataSave = self.Preprocessing('CIFAR', toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = self.Preprocessing('CIFAR', shadowData, shadowTestData)
        elif dataSet == 'CIFAR100':
            print("Loading data")
            dataX, dataY, testX, testY = self.readData('CIFAR100', orginialDatasetPath)
            X = np.concatenate([dataX, testX], axis=0)
            Y = np.concatenate([dataY, testY], axis=0)
            print("Preprocessing data")
            dataPath = dataFolderPath + '/Preprocessed'
            # 得到划分后的目标模型和影子模型训练集和测试集
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, cross)
            else:
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, attackCluster, cross)
            # 对数据进行预处理
            toTrainDataSave, toTestDataSave = self.Preprocessing('CIFAR', toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = self.Preprocessing('CIFAR', shadowData, shadowTestData)
        elif dataSet == 'MNIST':
            print("Loading data")
            dataX, dataY, testX, testY = self.readData('MINST', orginialDatasetPath)
            X = np.concatenate([dataX, testX], axis=0)
            Y = np.concatenate([dataY, testY], axis=0)
            print("Preprocessing data")
            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, cross)
            else:
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, attackCluster, cross)
            toTrainDataSave, toTestDataSave = self.Preprocessing('MINST', toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = self.Preprocessing('MINST', shadowData, shadowTestData)

        elif dataSet == 'Texas100':
            print("Loading data")
            data = np.load(dataFolderPath + '/texas100.npz')
            X = data['features']
            Y = torch.from_numpy(data['labels']).to(torch.from_numpy(data['labels']).device, dtype=torch.float).argmax(dim=-1).numpy()

            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, cross)
            else:
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, attackCluster, cross)
            # 对数据进行预处理
            toTrainDataSave, toTestDataSave = self.Preprocessing('Texas100', toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = self.Preprocessing('Texas100', shadowData, shadowTestData)
        elif dataSet == 'Purchase100':
            print("Loading Purchase100 data")
            data = np.load(dataFolderPath + '/purchase100.npz')
            X = data['features']
            Y = data['labels']
            # 转化标签
            Y = np.asarray([np.argmax(i == 1) for i in Y[:, :, 1]])

            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, cross)
            else:
                toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData(X, Y, trainCluster, testCluster, attackCluster, cross)
            # 对数据进行预处理
            toTrainDataSave, toTestDataSave = self.Preprocessing('Purchase100', toTrainData, toTestData)
            shadowDataSave, shadowTestDataSave = self.Preprocessing('Purchase100', shadowData, shadowTestData)
        else:
            raise ValueError('Unsupported dataset')

        try:
            os.makedirs(dataPath)
        except OSError:
            pass
        # 将划分后的数据保存为npz格式，方便下次直接读取
        np.savez(dataPath + '/targetTrain_{}.npz'.format(self.opt.dataset), toTrainDataSave, toTrainLabel)
        np.savez(dataPath + '/targetTest_{}.npz'.format(self.opt.dataset), toTestDataSave, toTestLabel)
        np.savez(dataPath + '/shadowTrain_{}.npz'.format(self.opt.dataset), shadowDataSave, shadowLabel)
        np.savez(dataPath + '/shadowTest_{}.npz'.format(self.opt.dataset), shadowTestDataSave, shadowTestLabel)
        print("Preprocessing finished\n\n")

    def initializeData_for_GanNoise(self, dataSet, orginialDatasetPath, trainCluster=10520, testCluster=10520,
                                    attackCluster=0, dataFolderPath='./dataset/', TrainTest_radio=0.2, cross=False):
        if dataSet == 'CIFAR10':
            print("Loading data")
            dataX, dataY, testX, testY = self.readData('CIFAR10', orginialDatasetPath)
            print("Preprocessing data")
            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                X = np.concatenate([dataX, testX], axis=0)
                Y = np.concatenate([dataY, testY], axis=0)
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            else:
                X = np.concatenate([dataX, testX], axis=0)
                Y = np.concatenate([dataY, testY], axis=0)
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, attackCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            # 对数据进行预处理
            toTrainDataSave, toTrainDataTSave, toTestDataSave = self.Preprocessing_for_GanNoise('CIFAR',
                                                                                                toTrainData_Train,
                                                                                                toTrainData_Test,
                                                                                                toTestData)
            shadowDataSave, toTrainDataTSave, shadowTestDataSave = self.Preprocessing_for_GanNoise('CIFAR',
                                                                                                   shadowData_Train,
                                                                                                   shadowData_Test,
                                                                                                   shadowTestData)
        elif dataSet == 'CIFAR100':
            print("Loading data")
            dataX, dataY, testX, testY = self.readData('CIFAR100', orginialDatasetPath)
            X = np.concatenate([dataX, testX], axis=0)
            Y = np.concatenate([dataY, testY], axis=0)
            print("Preprocessing data")
            dataPath = dataFolderPath + '/Preprocessed'
            # 得到划分后的目标模型和影子模型训练集和测试集
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            else:
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, attackCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            # 对数据进行预处理
            toTrainDataSave, toTrainDataTSave, toTestDataSave = self.Preprocessing_for_GanNoise('CIFAR',
                                                                                                toTrainData_Train,
                                                                                                toTrainData_Test,
                                                                                                toTestData)
            shadowDataSave, toTrainDataTSave, shadowTestDataSave = self.Preprocessing_for_GanNoise('CIFAR',
                                                                                                   shadowData_Train,
                                                                                                   shadowData_Test,
                                                                                                   shadowTestData)
        elif dataSet == 'MNIST':
            print("Loading data")
            dataX, dataY, testX, testY = self.readData('MINST', orginialDatasetPath)
            X = np.concatenate([dataX, testX], axis=0)
            Y = np.concatenate([dataY, testY], axis=0)
            print("Preprocessing data")
            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            else:
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, attackCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)

            # 对数据进行预处理
            toTrainDataSave, toTrainDataTSave, toTestDataSave = self.Preprocessing_for_GanNoise('MINST',
                                                                                                toTrainData_Train,
                                                                                                toTrainData_Test,
                                                                                                toTestData)
            shadowDataSave, toTrainDataTSave, shadowTestDataSave = self.Preprocessing_for_GanNoise('MINST',
                                                                                                   shadowData_Train,
                                                                                                   shadowData_Test,
                                                                                                   shadowTestData)
        elif dataSet == 'Texas100':
            print("Loading data")
            data = np.load(dataFolderPath + '/texas100.npz')
            X = data['features']
            Y = torch.from_numpy(data['labels']).to(torch.from_numpy(data['labels']).device, dtype=torch.float).argmax(dim=-1).numpy()

            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            else:
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, attackCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            # 对数据进行预处理
            toTrainDataSave, toTrainDataTSave, toTestDataSave = self.Preprocessing_for_GanNoise('Texas100',
                                                                                                toTrainData_Train,
                                                                                                toTrainData_Test,
                                                                                                toTestData)
            shadowDataSave, toTrainDataTSave, shadowTestDataSave = self.Preprocessing_for_GanNoise('Texas100',
                                                                                                   shadowData_Train,
                                                                                                   shadowData_Test,
                                                                                                   shadowTestData)
        elif dataSet == 'Purchase100':
            print("Loading Purchase100 data")
            data = np.load(dataFolderPath + '/purchase100.npz')
            X = data['features']
            Y = data['labels']
            # 转化标签
            Y = np.asarray([np.argmax(i == 1) for i in Y[:, :, 1]])

            dataPath = dataFolderPath + '/Preprocessed'
            if cross == False:
                # 得到划分后的目标模型和影子模型训练集和测试集
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            else:
                toTrainData_Train, toTrainLabel_Train, toTrainData_Test, toTrainLabel_Test, shadowData_Train, shadowLabel_Train, shadowData_Test, shadowLabel_Test, toTestData, toTestLabel, shadowTestData, shadowTestLabel = \
                    self.shuffleAndSplitData_for_GanNoise(X, Y, trainCluster, testCluster, attackCluster, cross=cross,
                                                          TrainTest_radio=TrainTest_radio)
            # 对数据进行预处理
            toTrainDataSave, toTrainDataTSave, toTestDataSave = self.Preprocessing_for_GanNoise('Purchase100',
                                                                                                toTrainData_Train, toTrainData_Test, toTestData)
            shadowDataSave, toTrainDataTSave, shadowTestDataSave = self.Preprocessing_for_GanNoise('Purchase100',
                                                                                                   shadowData_Train, shadowData_Test, shadowTestData)
        else:
            raise ValueError('Unsupported dataset')

        try:
            os.makedirs(dataPath)
        except OSError:
            pass
        # 将划分后的数据保存为npz格式，方便下次直接读取
        np.savez(dataPath + '/targetTrain_{}.npz'.format(self.opt.dataset), toTrainDataSave, toTrainLabel_Train)
        np.savez(dataPath + '/targetTrainT_{}.npz'.format(self.opt.dataset), toTrainDataTSave, toTrainLabel_Test)
        np.savez(dataPath + '/targetTest_{}.npz'.format(self.opt.dataset), toTestDataSave, toTestLabel)
        np.savez(dataPath + '/shadowTrain_{}.npz'.format(self.opt.dataset), shadowDataSave, shadowLabel_Train)
        np.savez(dataPath + '/shadowTrainT_{}.npz'.format(self.opt.dataset), toTrainDataTSave, shadowLabel_Test)
        np.savez(dataPath + '/shadowTest_{}.npz'.format(self.opt.dataset), shadowTestDataSave, shadowTestLabel)
        print("Preprocessing finished\n\n")

    def preprocess_data(self):
        if self.opt.dataset == 'MNIST':
            _ = datasets.MNIST(
                self.opt.datapath, True, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            _ = datasets.MNIST(
                self.opt.datapath, False, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            datapath = self.opt.datapath + '/MNIST'
        elif self.opt.dataset == 'CIFAR10':
            _ = datasets.CIFAR10(
                self.opt.datapath, True, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            _ = datasets.CIFAR10(
                self.opt.datapath, False, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            datapath = self.opt.datapath + '/cifar-10-batches-py'
        elif self.opt.dataset == 'CIFAR100':
            _ = datasets.CIFAR100(
                self.opt.datapath, True, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            _ = datasets.CIFAR100(
                self.opt.datapath, False, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            datapath = self.opt.datapath + '/cifar-100-python'
        elif self.opt.dataset == 'Texas100':
            datapath = self.opt.datapath
        elif self.opt.dataset == 'Purchase100':
            datapath = self.opt.datapath
        else:
            raise ValueError('Unsupported data set')

        self.initializeData(
            dataSet=self.opt.dataset,
            orginialDatasetPath=datapath,
            trainCluster=self.opt.train_cluster,
            testCluster=self.opt.test_cluster,
            attackCluster=self.opt.attack_cluster,
            dataFolderPath=self.opt.datapath,
            cross=self.opt.cross
        )

    def preprocess_data_for_GanNoise(self):
        if self.opt.dataset == 'MNIST':
            _ = datasets.MNIST(
                self.opt.datapath, True, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            _ = datasets.MNIST(
                self.opt.datapath, False, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            datapath = self.opt.datapath + '/MNIST'
        elif self.opt.dataset == 'CIFAR10':
            _ = datasets.CIFAR10(
                self.opt.datapath, True, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            _ = datasets.CIFAR10(
                self.opt.datapath, False, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            datapath = self.opt.datapath + '/cifar-10-batches-py'
        elif self.opt.dataset == 'CIFAR100':
            _ = datasets.CIFAR100(
                self.opt.datapath, True, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            _ = datasets.CIFAR100(
                self.opt.datapath, False, transform=transforms.Compose(
                    [
                        transforms.ToTensor()
                    ]
                ), download=True
            )
            datapath = self.opt.datapath + '/cifar-100-python'
        elif self.opt.dataset == 'Texas100':
            datapath = self.opt.datapath
        elif self.opt.dataset == 'Purchase100':
            datapath = self.opt.datapath
        else:
            raise ValueError('Unsupported data set')
        self.initializeData_for_GanNoise(
            dataSet=self.opt.dataset,
            orginialDatasetPath=datapath,
            trainCluster=self.opt.train_cluster,
            testCluster=self.opt.test_cluster,
            attackCluster=self.opt.attack_cluster,
            dataFolderPath=self.opt.datapath,
            TrainTest_radio=self.opt.traintest_radio,
            cross=self.opt.cross
        )

    # give test data
    def giveData(self, ofTarget=True, preprocess=False):
        '''
        :param ofTarget: give target data or not
        :return: ndarrary
        '''
        if preprocess == True:
            self.preprocess_data()

        # load data
        def loadData(data_name):
            with np.load(data_name) as f:
                train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
            return train_x, train_y

        if ofTarget == True:
            data_train_name = '/targetTrain_{}.npz'.format(self.opt.dataset)
            data_test_name = '/targetTest_{}.npz'.format(self.opt.dataset)
        else:
            data_train_name = '/shadowTrain_{}.npz'.format(self.opt.dataset)
            data_test_name = '/shadowTest_{}.npz'.format(self.opt.dataset)

        try:
            targetTrain, targetTrainLabel = loadData(self.opt.dataloadpath + data_train_name)
            tTest, tTestLabel = loadData(self.opt.dataloadpath + data_test_name)
        except:
            self.preprocess_data()
            targetTrain, targetTrainLabel = loadData(self.opt.dataloadpath + data_train_name)
            tTest, tTestLabel = loadData(self.opt.dataloadpath + data_test_name)

        dataset = (targetTrain.astype(np.float32),
                   targetTrainLabel.astype(np.int32),
                   tTest.astype(np.float32),
                   tTestLabel.astype(np.int32),
                   )
        # dataset = (targetTrain.astype(np.float32),
        #            targetTrainLabel.astype(np.int32),
        #            targetTest.astype(np.float32),
        #            targetTestLabel.astype(np.int32),
        #            t_targetTest.astype(np.float32),
        #            t_targetTestLabel.astype(np.int32),
        #            )
        return dataset

    def giveGanNoiseData(self, ofTarget=True, preprocess=False):
        """
        :param ofTarget: give target data or not
        :return: ndarrary
        """
        if preprocess == True:
            self.preprocess_data_for_GanNoise()

        # load data
        def loadData(data_name):
            with np.load(data_name) as f:
                train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
            return train_x, train_y

        if ofTarget == True:
            data_train_name = '/targetTrain_{}.npz'.format(self.opt.dataset)
            data_traint_name = '/targetTrainT_{}.npz'.format(self.opt.dataset)
            data_test_name = '/targetTest_{}.npz'.format(self.opt.dataset)
        else:
            data_train_name = '/shadowTrain_{}.npz'.format(self.opt.dataset)
            data_traint_name = '/shadowTrainT_{}.npz'.format(self.opt.dataset)
            data_test_name = '/shadowTest_{}.npz'.format(self.opt.dataset)

        try:
            targetTrain, targetTrainLabel = loadData(self.opt.dataloadpath + data_train_name)
            targetTrainT, targetTrainTLabel = loadData(self.opt.dataloadpath + data_traint_name)
            tTest, tTestLabel = loadData(self.opt.dataloadpath + data_test_name)
        except:
            self.preprocess_data_for_GanNoise()
            targetTrain, targetTrainLabel = loadData(self.opt.dataloadpath + data_train_name)
            targetTrainT, targetTrainTLabel = loadData(self.opt.dataloadpath + data_traint_name)
            tTest, tTestLabel = loadData(self.opt.dataloadpath + data_test_name)

        dataset = (targetTrain.astype(np.float32),
                   targetTrainLabel.astype(np.int32),
                   targetTrainT.astype(np.float32),
                   targetTrainTLabel.astype(np.int32),
                   tTest.astype(np.float32),
                   tTestLabel.astype(np.int32),
                   )

        return dataset


class Train():
    def __init__(self, opt):
        # Initialize static data
        self.opt = opt

    # Batch process the training data
    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=True):
        '''
        :param inputs: ndarray,such as (100,1,28,28)
        :param targets: ndarray,such as (100,10)
        :param batch_size: int, batch size of train data
        :param shuffle: decide whether to disrupt the data
        :return:
        '''
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)

        start_idx = None
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield inputs[excerpt], targets[excerpt]

        if start_idx is not None and start_idx + batch_size < len(inputs):
            excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
            yield inputs[excerpt], targets[excerpt]

    def give_same_batch(self, inputs, targets, batch_size):
        for input_batch, target_batch in self.iterate_minibatches(inputs, targets, batch_size, shuffle=True):
            return input_batch, target_batch

    # Core function of training model
    def trainModel(self, model, dataset, keep_path='.model/ModelStore/', keep_name='model', batch_size=None,
                   epochs=None, learning_rate=None, l2_ratio=None, l2_flag=False, keep_model_flag=False,
                   SGD_nor_not=False, eval_flag=False):
        '''
        :param model: nn.Module
        :param dataset: ndarrary
        :param batch_size: int
        :param epochs: int
        :param learning_rate: float
        :param l2_ratio: float
        :return: nn.Module
        '''
        train_x, train_y, test_x, test_y = dataset
        n_in = train_x.shape
        n_out = len(np.unique(train_y))
        if batch_size == None:
            batch_size = self.opt.batch_size
        if epochs == None:
            epochs = self.opt.epochs
        if learning_rate == None:
            learning_rate = self.opt.learning_rate
        if l2_ratio == None:
            l2_ratio = self.opt.l2_ratio

        if batch_size > len(train_y):
            batch_size = len(train_y)
        print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))

        # create loss function
        m = n_in[0]
        if l2_flag == True:
            criterion = CrossEntropy_L2(model, m, l2_ratio).to(self.opt.device)
        else:
            criterion = nn.CrossEntropyLoss()

        model.to(self.opt.device)

        # create optimizer
        if SGD_nor_not:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # count loss in an epoch
        temp_loss = 0.0

        print('Training...')
        model.train()
        for epoch in range(epochs):
            for input_batch, target_batch in self.iterate_minibatches(train_x, train_y, batch_size):
                input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
                input_batch, target_batch = input_batch.to(self.opt.device), target_batch.to(self.opt.device)
                optimizer.zero_grad()
                outputs = model(input_batch)
                loss = criterion(outputs, target_batch)
                loss.backward()
                optimizer.step()
                temp_loss += loss.item()

            temp_loss = round(temp_loss, 3)
            if epoch % 5 == 0:
                print('Epoch {}, train loss {}'.format(epoch, temp_loss))

            temp_loss = 0.0
        if keep_model_flag:
            torch.save(model.state_dict(), keep_path + keep_name + '.pth')
            if eval_flag:
                torch.save(model.state_dict(), keep_path + keep_name + '_eval' + '.pth')
        model.eval()  # 把网络设定为测试状态
        pred_y = []
        with torch.no_grad():
            for input_batch, _ in self.iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
                input_batch = torch.tensor(input_batch)
                input_batch = input_batch.to(self.opt.device)
                outputs = model(input_batch)
                pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
            pred_y = np.concatenate(pred_y)
        print('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))

        model.eval()  # 把网络设定为测试状态
        pred_y = []
        if test_x is not None:
            print('Testing...')
            if batch_size > len(test_y):
                batch_size = len(test_y)
            with torch.no_grad():
                for input_batch, _ in self.iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
                    input_batch = torch.tensor(input_batch)
                    input_batch = input_batch.to(self.opt.device)
                    outputs = model(input_batch)
                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
                pred_y = np.concatenate(pred_y)

            print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
        # print('More detailed results: \n {}'.format(classification_report(test_y, pred_y)))

        mcm = multilabel_confusion_matrix(test_y, pred_y, labels=[0, 1])
        print(mcm)

        return model, accuracy_score(test_y, pred_y)

        # Core function of training model

    # Print label by Training model
    def giveLabelPrinting(self, model, train_dataset, test_dataset, batch_size=None):
        train_x, train_y = train_dataset
        test_x, test_y = test_dataset
        attack_x, attack_y = [], []
        if batch_size == None:
            batch_size = self.opt.batch_size
        # data used in training, label is 1
        for batch, _ in self.iterate_minibatches(train_x, train_y, batch_size, False):
            batch = torch.tensor(batch)
            batch = batch.to(self.opt.device)
            output = model(batch)
            preds_tensor = nn.functional.softmax(output, dim=1)
            attack_x.append(preds_tensor.detach().cpu().numpy())
            attack_y.append(np.ones(len(batch)))
        # data not used in training, label is 0
        for batch, _ in self.iterate_minibatches(test_x, test_y, batch_size, False):
            batch = torch.tensor(batch)
            batch = batch.to(self.opt.device)
            output = model(batch)
            preds_tensor = nn.functional.softmax(output, dim=1)
            attack_x.append(preds_tensor.detach().cpu().numpy())
            attack_y.append(np.zeros(len(batch)))

        attack_x = np.vstack(attack_x)
        attack_y = np.concatenate(attack_y)
        attack_x = attack_x.astype('float32')
        attack_y = attack_y.astype('int32')
        return attack_x, attack_y

    # Print acc
    def giveTrainingInfo(self, model, train_dataset, test_dataset, epoch, device=torch.device('cuda'), batch_size=None,
                         test_only=False):
        model.to(device)
        model.eval()  # 把网络设定为测试状态
        if batch_size == None:
            batch_size = self.opt.batch_size
        train_x, train_y = train_dataset
        test_x, test_y = test_dataset
        pred_y = []
        trainAcc = 0
        # print('epoch {} acc: '.format(epoch))
        if test_only == False:
            with torch.no_grad():
                for input_batch, _ in self.iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
                    input_batch = torch.tensor(input_batch)
                    input_batch = input_batch.to(self.opt.device)
                    outputs = model(input_batch)
                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
                pred_y = np.concatenate(pred_y)
            trainAcc = accuracy_score(train_y, pred_y)
            # print('Training Accuracy: {}'.format(trainAcc))

        model.eval()  # 把网络设定为测试状态
        pred_y = []
        if test_x is not None:
            # print('Testing...')
            if batch_size > len(test_y):
                batch_size = len(test_y)
            with torch.no_grad():
                for input_batch, _ in self.iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
                    input_batch = torch.tensor(input_batch)
                    input_batch = input_batch.to(self.opt.device)
                    outputs = model(input_batch)
                    pred_y.append(torch.max(outputs, 1)[1].data.cpu().numpy())
                pred_y = np.concatenate(pred_y)
        testAcc = accuracy_score(test_y, pred_y)
        # print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
        # print('more info: {}'.format(classification_report(test_y, pred_y)))
        return trainAcc, testAcc

    def trainGanFirst(self, netG, netD, dataset, epochs=100, nz=100, keep_path='model/ModelStore/'):
        # 拆解数据集
        train_x, train_y, test_x, test_y = dataset
        # 确定损失函数
        criterion = torch.nn.BCELoss()
        # 优化器选择
        optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
        real_label = 1.0
        fake_label = 0.0
        netG.to(self.opt.device)
        netD.to(self.opt.device)
        # 训练开始
        for epoch in range(0, epochs):
            batch = 0
            for input_batch, target_batch in self.iterate_minibatches(train_x, train_y, self.opt.at_batch_size):
                input_batch, target_batch = torch.tensor(input_batch).to(self.opt.device), torch.tensor(
                    target_batch).type(torch.long).to(self.opt.device)
                batch_size = input_batch.size(0)
                label = torch.full((batch_size, 1), real_label).to(self.opt.device)
                # （1）训练判别器
                # training real data
                netD.zero_grad()
                real_data = input_batch.to(self.opt.device)
                output = netD(real_data)
                loss_D1 = criterion(output, label)
                loss_D1.backward()
                # training fake data
                noise_z = torch.randn(batch_size, nz, 1, 1, device=self.opt.device)
                fake_data = netG(noise_z)
                label = torch.full((batch_size, 1), fake_label).to(self.opt.device)
                output = netD(fake_data.detach())
                loss_D2 = criterion(output, label)
                loss_D2.backward()

                # 更新判别器
                optimizerD.step()

                # （2）训练生成器
                netG.zero_grad()
                label = torch.full((batch_size, 1), real_label).to(self.opt.device)
                output = netD(fake_data)
                lossG = criterion(output, label)
                lossG.backward()

                # 更新生成器
                optimizerG.step()

                if batch % 100 == 0:
                    print('epoch: %4d, batch: %4d, discriminator loss: %.4f, generator loss: %.4f'
                          % (epoch, batch, loss_D1.item() + loss_D2.item(), lossG.item()))
                batch += 1

        torch.save(netG.state_dict(), keep_path + 'GANG_%s_best.pth' % (self.opt.dataset))
        torch.save(netD.state_dict(), keep_path + 'GAND_%s_best.pth' % (self.opt.dataset))
        return netG, netD

    # Train Gan model
    def trainGanModel(
            self, generator, discriminator, dataset, batch_size=None, epochs=None, learning_rate=None, l2_ratio=None,
            l2_flag=False
    ):
        if batch_size == None:
            batch_size = self.opt.batch_size
        if epochs == None:
            epochs = self.opt.epochs
        if learning_rate == None:
            learning_rate = self.opt.learning_rate
        if l2_ratio == None:
            l2_ratio = self.opt.l2_ratio
        train_x, train_y, test_x, test_y = dataset
        n_in = train_x.shape
        n_out = len(np.unique(train_y))
        m = n_in[0]

        if l2_flag == True:
            criterion = CrossEntropy_L2(generator, m, l2_ratio).to(self.opt.device)
        else:
            criterion = nn.CrossEntropyLoss()
        # create optimizer
        gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=self.opt.at_learning_rate)
        # count loss in an epoch
        temp_loss = 0.0

        def give_weight(accLoss, discriminatorLoss):
            if accLoss > 1:  # max function
                b = 0
                a = 1
            elif accLoss < 1 and discriminatorLoss < 0.55:
                a = 0.95
                b = 0.05
            elif accLoss < 1 and discriminatorLoss > 0.6:
                a = 0.2
                b = 0.8
            elif accLoss < 0.1 and discriminatorLoss > 0.7:
                a = 0.1
                b = 0.9
            else:
                a = 0.5
                b = 0.5
            return a, b

        epoch = 0
        # gan training
        while True:
            epoch += 1
            generator.to(self.opt.device).train()
            discriminator.to(self.opt.device).train()
            # test data for discriminator
            attack_x, attack_y = self.giveLabelPrinting(generator, (train_x, train_y), (test_x, test_y),
                                                        self.opt.at_batch_size)
            # Take the top three with the highest output distribution probability of target model and shadow model
            attackX = np.array([sorted(s, reverse=True)[0:3] for s in attack_x])
            attackY = attack_y
            x_train, x_test, y_train, y_test = train_test_split(attackX, attackY, test_size=0.25)
            num = 0
            for input_batch, target_batch in self.iterate_minibatches(x_train, y_train, self.opt.at_batch_size):
                num += 1
                input_batch, target_batch = torch.tensor(input_batch).to(self.opt.device), torch.tensor(
                    target_batch).type(torch.long).to(self.opt.device)
                dis_optimizer.zero_grad()
                outputs = discriminator(input_batch)
                dLoss = criterion(outputs, target_batch)
                dLoss.backward()
                temp_loss += dLoss.item()
                dis_optimizer.step()
            print('epoch {} training attack model loss:{} '.format(epoch, temp_loss / num))
            temp_loss = 0
            # give discriminator acc
            _, discriminatorLoss = self.giveTrainingInfo(model=discriminator, train_dataset=(x_train, y_train),
                                                         test_dataset=(x_test, y_test), epoch=epoch,
                                                         batch_size=self.opt.at_batch_size, device=self.opt.device,
                                                         test_only=True)
            num = 0
            # generator_model,generator_accuracy = self.trainModel(generator,dataset, batch_size, epochs, learning_rate, l2_ratio)
            for input_batch, target_batch in self.iterate_minibatches(train_x, train_y, batch_size):
                num += 1
                input_batch, target_batch = torch.tensor(input_batch).to(self.opt.device), torch.tensor(
                    target_batch).type(torch.long).to(self.opt.device)
                gen_optimizer.zero_grad()
                outputs = generator(input_batch)
                accLoss = criterion(outputs, target_batch).item()
                a, b = give_weight(accLoss, discriminatorLoss)
                loss = a * criterion(outputs, target_batch) + b * discriminatorLoss
                loss.backward()
                gen_optimizer.step()
                temp_loss += loss.item()
            print('epoch {} training target model loss:{} '.format(epoch, temp_loss / num))
            temp_loss = 0
            # give generator acc
            _, testAcc = self.giveTrainingInfo(model=generator, train_dataset=(train_x, train_y),
                                               test_dataset=(test_x, test_y), epoch=epoch, batch_size=batch_size,
                                               device=self.opt.device, test_only=True)

            if (testAcc > 0.8 and discriminatorLoss <= 0.55) or epoch >= epochs:
                print("==================================== Training Gan end! =====================================")
                print('testAcc is {}'.format(testAcc))
                print('discriminatorLoss is {}'.format(discriminatorLoss))
                break
        torch.save(generator.state_dict(),
                   self.opt.keepmodelpath + '{}_{}_gandefence.pth'.format(self.opt.model, self.opt.dataset))
        return generator

    def trainGanTest(
            self, generator, discriminator, dataset, batch_size=None, epochs=None, learning_rate=None, l2_ratio=None,
            l2_flag=False
    ):
        if batch_size == None:
            batch_size = self.opt.batch_size
        if epochs == None:
            epochs = self.opt.epochs
        if learning_rate == None:
            learning_rate = self.opt.learning_rate
        if l2_ratio == None:
            l2_ratio = self.opt.l2_ratio
        train_x, train_y, test_x, test_y = dataset
        n_in = train_x.shape
        n_out = len(np.unique(train_y))
        m = n_in[0]

        if l2_flag == True:
            criterion = CrossEntropy_L2(generator, m, l2_ratio).to(self.opt.device)
        else:
            criterion = nn.CrossEntropyLoss()

        def give_weight(accLoss, discriminatorLoss):
            if accLoss > 1:  # max function
                b = 0
                a = 1
            elif accLoss < 0.1 and discriminatorLoss > 0.7:
                a = 0.2
                b = 0.8
            elif accLoss < 0.5 and discriminatorLoss > 0.6:
                a = 0.4
                b = 0.6
            elif accLoss < 1 and discriminatorLoss < 0.55:
                a = 0.95
                b = 0.05
            else:
                a = 0.5
                b = 0.5
            return a, b

        # create optimizer
        gen_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=self.opt.at_learning_rate)
        # count loss in an epoch
        temp_loss = 0.0
        epoch = 0
        generator.to(self.opt.device).train()
        discriminator.to(self.opt.device).train()
        # gan training
        while True:
            epoch += 1
            num = 1
            this_num = 0
            this_testAcc = 0
            this_discriminatorLoss = 0
            attack_train_flag = False
            # generator_model,generator_accuracy = self.trainModel(generator,dataset, batch_size, epochs, learning_rate, l2_ratio)
            for input_batch, target_batch in self.iterate_minibatches(train_x, train_y, batch_size):
                this_num += 1
                input_batch, target_batch = torch.tensor(input_batch).to(self.opt.device), torch.tensor(
                    target_batch).type(torch.long).to(self.opt.device)
                gen_optimizer.zero_grad()
                outputs = generator(input_batch)
                # give generator acc
                _, testAcc = self.giveTrainingInfo(model=generator, train_dataset=(train_x, train_y),
                                                   test_dataset=(test_x, test_y), epoch=epoch, batch_size=batch_size,
                                                   device=self.opt.device, test_only=True)
                # test data for discriminator
                attack_x, attack_y = self.giveLabelPrinting(generator, (train_x, train_y), (test_x, test_y),
                                                            self.opt.at_batch_size)
                # Take the top three with the highest output distribution probability of target model and shadow model
                attackX = np.array([sorted(s, reverse=True)[0:3] for s in attack_x])
                attackY = attack_y
                x_train, x_test, y_train, y_test = train_test_split(attackX, attackY, test_size=0.25)
                temp_loss = 0
                # 计算攻击损失值
                _, discriminatorLoss = self.giveTrainingInfo(model=discriminator, train_dataset=(x_train, y_train),
                                                             test_dataset=(x_test, y_test), epoch=epoch,
                                                             batch_size=self.opt.at_batch_size, device=self.opt.device,
                                                             test_only=True)
                if discriminatorLoss <= 0.6 and attack_train_flag != True:
                    # 计算MIA置信度
                    for d_input_batch, d_target_batch in self.iterate_minibatches(x_train, y_train,
                                                                                  self.opt.at_batch_size):
                        if num != 1:
                            num += 1
                        d_input_batch, d_target_batch = torch.tensor(d_input_batch).to(self.opt.device), torch.tensor(
                            d_target_batch).type(torch.long).to(self.opt.device)
                        dis_optimizer.zero_grad()
                        d_outputs = discriminator(d_input_batch)
                        dLoss = criterion(d_outputs, d_target_batch)
                        dLoss.backward()
                        temp_loss += dLoss.item()
                        dis_optimizer.step()
                    # give discriminator acc
                    this_discriminatorLoss += discriminatorLoss
                    attack_train_flag = True
                    continue
                attack_train_flag = False
                this_testAcc += testAcc
                accLoss = criterion(outputs, target_batch).item()
                a, b = give_weight(accLoss, discriminatorLoss)
                # loss = a * criterion(outputs, target_batch) + b * (discriminatorLoss - 0.5)
                loss = torch.add(a * torch.mean(criterion(outputs, target_batch)), b * (discriminatorLoss - 0.5))
                # loss =   a * criterion(outputs, target_batch) + b *(discriminatorLoss)
                loss.backward()
                gen_optimizer.step()
                temp_loss += loss.item()
            print('epoch {} training target model loss:{} '.format(epoch, this_testAcc / this_num))
            print('epoch {} training attack model loss:{} '.format(epoch, discriminatorLoss))
            if (this_testAcc / this_num > 0.8 and this_discriminatorLoss / this_num <= 0.55) or epoch >= epochs:
                print("==================================== Training Gan end! =====================================")
                print('testAcc is {}'.format(this_testAcc / this_num))
                print('discriminatorLoss is {}'.format(this_discriminatorLoss / num))
                break
        torch.save(generator.state_dict(), self.opt.keepmodelpath + 'g.pth')
        return generator


if __name__ == '__main__':
    pass
