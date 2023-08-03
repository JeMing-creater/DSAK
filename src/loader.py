import os
import pickle
import random
from typing import Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
from easydict import EasyDict
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torchvision import transforms, datasets

from src.model import DesNet_utilis, VGG_utilis, RestNet_utilis, CNN, AlexNet
from src.model.CrossEntropy import CrossEntropy_L2


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super(NumpyDataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回一个样本
        :param index: 样本下标
        :return: 图片Tensor， 图片标签
        """
        return self.x[index], self.y[index].astype(np.long)

    def __len__(self) -> int:
        return self.y.shape[0]


def get_transforms(dataset_name: str, config: EasyDict) -> Tuple[torchvision.transforms.Compose, torchvision.transforms.Compose]:
    if dataset_name == 'CIFAR100':
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomGrayscale(),
        ])
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomGrayscale(),
        ])
        return train_transform, val_transform
    elif dataset_name == 'Purchase100':
        return torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.RandomGrayscale(),
        ])
    else:
        raise ValueError()


# ========================== 功能类 ===========================
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
        else:
            print('There are not preprocessing of this dataset!')
            return toTrainData, toTestData

    # init data
    def initializeData(self, dataSet, orginialDatasetPath, trainCluster=10520, testCluster=10520, attackCluster=0,
                       dataFolderPath='./dataset/', cross=False):
        '''
        param dataName: str, name of dataset
        :param orginialDatasetPath: path of this dataset
        :param testCluster: int
        :param dataFolderPath: path of all dataset
        :return:
        '''
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

    # give model
    def giveModel(self):
        if self.opt.model == 'CNN':
            target_model = CNN.CNN_Model(self.opt.channel, self.opt.in_features, self.opt.out_features, self.opt.n_out)
            shadow_model = CNN.CNN_Model(self.opt.channel, self.opt.in_features, self.opt.out_features, self.opt.n_out)
        elif self.opt.model == 'ResNet18':
            target_model = RestNet_utilis.ResNet18(in_channels=self.opt.channel, num_classes=self.opt.n_out)
            shadow_model = RestNet_utilis.ResNet18(in_channels=self.opt.channel, num_classes=self.opt.n_out)
        elif self.opt.model == 'DenseNet169':
            target_model = DesNet_utilis.densenet169(in_channels=self.opt.channel, num_class=self.opt.n_out)
            shadow_model = DesNet_utilis.densenet169(in_channels=self.opt.channel, num_class=self.opt.n_out)
        elif self.opt.model == 'VGG11':
            target_model = VGG_utilis.VGG(vgg_name='vgg11', in_channels=self.opt.channel, num_classes=self.opt.n_out)
            shadow_model = VGG_utilis.VGG(vgg_name='vgg11', in_channels=self.opt.channel, num_classes=self.opt.n_out)
        elif self.opt.model == 'VGG16':
            target_model = VGG_utilis.VGG(vgg_name='vgg16', in_channels=self.opt.channel, num_classes=self.opt.n_out)
            shadow_model = VGG_utilis.VGG(vgg_name='vgg16', in_channels=self.opt.channel, num_classes=self.opt.n_out)
        elif self.opt.model == 'AlexNet':
            target_model = AlexNet.AlexNet(in_channels=self.opt.channel, num_classes=self.opt.n_out)
            shadow_model = AlexNet.AlexNet(in_channels=self.opt.channel, num_classes=self.opt.n_out)
        return target_model, shadow_model

    # give test data
    def giveData(self, ofTarget=True, preprocess=False):
        '''
        :param ofTarget: give target data or not
        :return: ndarrary
        '''
        if preprocess == True:
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
            if self.opt.dataset == 'CIFAR10':
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
            if self.opt.dataset == 'CIFAR100':
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
            self.initializeData(
                dataSet=self.opt.dataset,
                orginialDatasetPath=datapath,
                trainCluster=self.opt.train_cluster,
                testCluster=self.opt.test_cluster,
                attackCluster=self.opt.attack_cluster,
                dataFolderPath=self.opt.datapath,
                cross=self.opt.cross
            )

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

        targetTrain, targetTrainLabel = loadData(self.opt.dataloadpath + data_train_name)
        # use for gan test
        tTest, tTestLabel = loadData(self.opt.dataloadpath + data_test_name)
        # targetTest, t_targetTest, targetTestLabel, t_targetTestLabel = train_test_split(tTest, tTestLabel,
        #                                                                                 test_size=split_size)
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
