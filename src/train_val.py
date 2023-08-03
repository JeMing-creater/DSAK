import os
import sys
from datetime import datetime
import torch.nn as nn
import yaml
import torch
import numpy as np
from accelerate import Accelerator
from easydict import EasyDict
from objprint import objstr
from timm.optim import optim_factory
from src.preprocess import GiveTestData, same_seeds, Logger
from src.utils import LinearWarmupCosineAnnealingLR, load_pretrain_model, give_model
import src.model_dic as model_dic
from src.MIA import FourAttack
from sklearn.metrics import accuracy_score
import math
import random
# from pyvacy import optim, analysis
from functools import partial
import torch.nn.functional as F


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
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


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)


def clipDataTopX(dataToClip, top=3):
    res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
    return np.array(res)


def train_one_epoch(config, model, batch_size, loss_functions, data,
                    optimizer, accelerator, scheduler, epoch, step):
    # 训练
    model.train()
    targetTrain, targetTrainLabel, targetTest, targetTestLabel = data
    i = 0
    for input_batch, target_batch in iterate_minibatches(targetTrain, targetTrainLabel, batch_size,
                                                         shuffle=True):
        input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
        input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
        logits = model(input_batch)
        label = target_batch

        total_loss = 0
        log = ''
        for name in loss_functions:
            loss = loss_functions[name](logits, label)
            accelerator.log({'Train/' + name: float(loss)}, step=step)
            log += f' {name} {float(loss):1.5f} '
            total_loss += loss
        accelerator.backward(total_loss)
        optimizer.step()
        optimizer.zero_grad()
        accelerator.log({
            'Train/Total Loss': float(total_loss),
        }, step=step)
        # accelerator.print(
        #     f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training [{i + 1}/{int(targetTrain.shape[0] / batch_size) + 1}] Loss: {total_loss:1.5f} {log}',
        #     flush=True)
        step += 1
        i += 1
    scheduler.step(epoch)
    model.eval()
    pred_y = []
    with torch.no_grad():
        correct = 0
        total = 0
        for input_batch, target_batch in iterate_minibatches(targetTrain, targetTrainLabel, batch_size,
                                                             shuffle=False):
            input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
            outputs = model(input_batch)
            log = ''
            total_loss = 0
            for name in loss_functions:
                loss = loss_functions[name](outputs, target_batch)
                accelerator.log({'Train/' + name: float(loss)}, step=step)
                log += f' {name} {float(loss):1.5f} '
                total_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum()

    metric = {}
    metric.update({
        f'Train/mean acc': correct / total,
    })

    accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training metric {metric}')
    accelerator.log(metric, step=epoch)

    return step


def val_one_epoch(config, model, loss_functions, data, accelerator, epoch, val_step):
    model.eval()
    targetTrain, targetTrainLabel, targetTest, targetTestLabel = data
    pred_y = []
    metric = {}
    batch_size = config.trainer.batch_size
    with torch.no_grad():
        correct = 0
        total = 0
        for input_batch, target_batch in iterate_minibatches(targetTest, targetTestLabel, batch_size,
                                                             shuffle=False):
            input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
            outputs = model(input_batch)
            log = ''
            total_loss = 0
            for name in loss_functions:
                loss = loss_functions[name](outputs, target_batch)
                accelerator.log({'Train/' + name: float(loss)}, step=val_step)
                log += f' {name} {float(loss):1.5f} '
                total_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum()
    metric.update({
        f'Test/mean acc': correct / total,
    })
    with torch.no_grad():
        correct = 0
        total = 0
        for input_batch, target_batch in iterate_minibatches(targetTrain, targetTrainLabel, batch_size,
                                                             shuffle=False):
            input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
            outputs = model(input_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum()
    metric.update({
        f'Train val/mean acc': correct / total,
    })
    accelerator.print(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Testing metric {metric}')
    accelerator.log(metric, step=epoch)
    return torch.Tensor([metric['Test/mean acc']]).to(accelerator.device), torch.Tensor(
        [metric['Train val/mean acc']]).to(accelerator.device), val_step


def val_Accuracy(target_model, batch_size, data, accelerator):
    target_model.eval()
    target_model.to(accelerator.device)
    targetTrain, targetTrainLabel, targetTest, targetTestLabel = data
    # targetTrain, targetTrainLabel, _, _, targetTest, targetTestLabel = data
    metric = {}
    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for input_batch, target_batch in iterate_minibatches(targetTrain, targetTrainLabel, batch_size,
                                                             shuffle=False):
            input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
            outputs = target_model(input_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum()
            # accelerator.print(
            #     f'Training data [{i + 1}/{int(targetTrain.shape[0] / batch_size) + 1}] Testing!',
            #     flush=True)
            i += 1
    metric.update({
        f'Train val/mean acc': correct / total,
    })
    with torch.no_grad():
        correct = 0
        total = 0
        i = 0
        for input_batch, target_batch in iterate_minibatches(targetTest, targetTestLabel, batch_size,
                                                             shuffle=False):
            input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
            input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
            outputs = target_model(input_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum()
            # accelerator.print(
            #     f'Testing data [{i + 1}/{int(targetTest.shape[0] / batch_size) + 1}] Testing!',
            #     flush=True)
            i += 1
    metric.update({
        f'Test/mean acc': correct / total,
    })

    # accelerator.print(f'Testing metric {metric}')
    return torch.Tensor([metric['Test/mean acc']]).to(accelerator.device), torch.Tensor(
        [metric['Train val/mean acc']]).to(accelerator.device)


def val_Attack_Accuracy(attack_model, target_model, batch_size, tdata, sdata, accelerator, generator_model=None):
    attack_model.eval()
    target_model.eval()
    # generator_model.eval()
    train_x, train_y, test_x, test_y = tdata
    strain_x, strain_y, _, _ = sdata
    r = np.arange(len(strain_x))
    np.random.shuffle(r)
    strain_x = strain_x[r]
    strain_y = strain_y[r]
    train_data = torch.from_numpy(strain_x).type(torch.FloatTensor)
    labels = torch.from_numpy(strain_y).type(torch.LongTensor)

    # 随机选择数据
    attack_data = torch.from_numpy(train_x).type(torch.FloatTensor)
    attack_labels = torch.from_numpy(train_y).type(torch.LongTensor)
    all_acc = 0
    i = 0
    num = 0
    with torch.no_grad():
        for inputs, targets in iterate_minibatches(attack_data, attack_labels, batch_size, True):
            inputs_attack, targets_attack = inputs.to(accelerator.device), targets.to(accelerator.device)
            # 取出对应标签
            if (num + 1) * batch_size >= len(train_data):
                r = np.arange(len(strain_x))
                np.random.shuffle(r)
                strain_x = strain_x[r]
                strain_y = strain_y[r]
                num = 0
            inputs = train_data[num * batch_size:(num + 1) * batch_size].to(accelerator.device)
            targets = labels[num * batch_size:(num + 1) * batch_size].to(accelerator.device)

            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            inputs_attack, targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(
                targets_attack)

            # 作目标模型输出
            if generator_model != None:
                img = (generator_model(inputs_attack) + inputs_attack).detach()
                img = torch.cat((img, inputs_attack))
                inputs = torch.cat((inputs, inputs))
                targets = torch.cat((targets, targets))
                targets_attack = torch.cat((targets_attack, targets_attack))
            else:
                img = inputs_attack

            # inputs = inputs
            # img = inputs_attack

            outputs = target_model(inputs)
            outputs_non = target_model(img)

            # 拼接攻击模型输入与标签
            classifier_input = torch.cat((inputs, img))
            comb_inputs = torch.cat((outputs, outputs_non))
            comb_targets = torch.cat((targets, targets_attack)).view([-1, 1]).type(torch.FloatTensor)

            # attack_input = comb_inputs
            one_hot_tr = torch.from_numpy((np.zeros((comb_inputs.size()[0], outputs.size(1))))).to(
                accelerator.device).type(
                torch.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets, targets_attack)).type(
                torch.LongTensor).view(
                [-1, 1]).data, 1).to(accelerator.device)
            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr).to(accelerator.device)
            attack_output = attack_model(comb_inputs, infer_input_one_hot).view([-1])
            att_labels = np.zeros((inputs.size()[0] + img.size()[0]))
            att_labels[:inputs.size()[0]] = 0.0
            att_labels[inputs.size()[0]:] = 1.0
            is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor).to(accelerator.device)
            v_is_member_labels = torch.autograd.Variable(is_member_labels)
            classifier_targets = comb_targets.clone().view([-1]).type(torch.LongTensor)
            attack_acc = np.mean(
                np.equal((attack_output.data.cpu().numpy() > 0.5), (v_is_member_labels.data.cpu().numpy() > 0.5)))
            all_acc += attack_acc
            i += 1
            num += 1
        all_acc = all_acc / i
    return all_acc


# 生成模型预训练
def advance_train_generator(config, generator_model, optimizer, data, accelerator, epochs=30,
                            train=False):
    same_seeds(config.trainer.seed)
    train_x, train_y, test_x, test_y = data
    if train == True:
        best_loss = 10000
        for epoch in range(0, epochs):
            generator_model.train()
            step = 0
            total_loss = 0
            for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, True):
                input_batch, target_batch = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
                input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
                optimizer.zero_grad()
                img = generator_model(input_batch)

                # loss = criterion(outputs, target)
                # loss = torch.cosine_similarity(outputs, target).mean()
                # loss = torch.abs(torch.cosine_similarity(img.unsqueeze(1), input_batch.unsqueeze(0))).mean()
                if 'CIFAR' in config.preprocess.dataset:
                    # loss = torch.cosine_similarity(img, input_batch).mean()
                    loss = torch.abs(torch.cosine_similarity(img.unsqueeze(1), input_batch.unsqueeze(0))).mean()
                else:
                    loss = torch.abs(torch.cosine_similarity(img.unsqueeze(1), input_batch.unsqueeze(0))).mean()
                accelerator.backward(loss)
                optimizer.step()
                accelerator.log({
                    'Generator Train/Total Loss': float(loss),
                }, step=step)
                total_loss += loss
                step += 1

            accelerator.print(
                f'Advance [{epoch + 1}/{epochs}] generator loss: {total_loss / (int(test_x.shape[0] / config.trainer.batch_size))}',
                flush=True)
            if total_loss / (int(test_x.shape[0] / config.trainer.batch_size)) < best_loss:
                accelerator.save_state(
                    output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}_Generator/best")

    generator_model = load_pretrain_model(
        f"{os.getcwd()}/model_store/{config.finetune.checkpoint}_Generator/best/pytorch_model.bin", generator_model,
        accelerator)
    return generator_model


# ============================================ other ============================================================
def advance_train_target(config, target_model, optimizer, criterion, data, accelerator, epochs=5):
    # 拆出模型相关工具
    target_model.train()
    # 拆分数据
    train_x, train_y, test_x, test_y = data
    r = np.arange(len(train_x))  # 随机选择数据
    np.random.shuffle(r)
    train_x = train_x[r]
    train_y = train_y[r]
    train_data = torch.from_numpy(train_x).type(torch.FloatTensor)
    labels = torch.from_numpy(train_y).type(torch.LongTensor)

    batch_size = config.trainer.batch_size
    i = 0
    all_loss = 0
    for epoch in range(0, epochs):
        target_model.train()
        for input_batch, target_batch in iterate_minibatches(train_data, labels, batch_size, shuffle=False):
            input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
            inputs, targets = torch.autograd.Variable(input_batch), torch.autograd.Variable(target_batch)

            # 作模型输出
            logits = target_model(inputs)

            # loss = criterion(logits, label) + config.defence.attack_radio *attack_guide
            loss = criterion(logits, targets)

            all_loss += loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            i += 1
        all_loss = all_loss / i
        val_acc, train_acc = val_Accuracy(target_model, config.trainer.batch_size, data, accelerator)
        accelerator.print(
            f'Advance [{epoch + 1}/{epochs}] Training Target Model Loss: {all_loss} acc: {val_acc}',
            flush=True)
    return target_model


# 攻击模型预训练
def advance_train_attack(config, target_model, attack_model, optimizer, criterion, tdata, sdata, accelerator,
                         epochs=5):
    # 拆分数据
    train_x, train_y, test_x, test_y = tdata
    strain_x, strain_y, _, _ = sdata
    r = np.arange(len(strain_x))
    np.random.shuffle(r)
    strain_x = strain_x[r]
    strain_y = strain_y[r]
    train_data = torch.from_numpy(strain_x).type(torch.FloatTensor)
    labels = torch.from_numpy(strain_y).type(torch.LongTensor)

    # 随机选择数据
    r = np.arange(len(train_x))
    np.random.shuffle(r)
    train_x = train_x[r]
    train_y = train_y[r]
    attack_data = torch.from_numpy(train_x).type(torch.FloatTensor)
    attack_labels = torch.from_numpy(train_y).type(torch.LongTensor)

    target_model.eval()
    attack_model.train()
    batch_size = config.trainer.batch_size
    for epoch in range(0, epochs):
        i = 0
        all_acc = 0
        all_loss = 0
        for inputs, targets in iterate_minibatches(train_data, labels, batch_size, False):
            # inputs, targets = torch.tensor(inputs), torch.tensor(targets).type(torch.long)
            inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
            # 取出对应标签
            inputs_attack = attack_data[i * batch_size:(i + 1) * batch_size].to(accelerator.device)
            targets_attack = attack_labels[i * batch_size:(i + 1) * batch_size].to(accelerator.device)
            # inputs_attack, targets_attack = torch.tensor(inputs_attack), torch.tensor(targets_attack).type(torch.long)
            inputs_attack, targets_attack = inputs_attack.to(accelerator.device), targets_attack.to(accelerator.device)
            inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            inputs_attack, targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(
                targets_attack)

            # 作目标模型输出
            optimizer.zero_grad()
            outputs = target_model(inputs)
            outputs_non = target_model(inputs_attack)

            # 拼接攻击模型输入与标签
            classifier_input = torch.cat((inputs, inputs_attack))
            comb_inputs = torch.cat((outputs, outputs_non))
            comb_targets = torch.cat((targets, targets_attack)).view([-1, 1]).type(torch.cuda.FloatTensor)

            # attack_input = comb_inputs
            one_hot_tr = torch.from_numpy((np.zeros((comb_inputs.size()[0], outputs.size(1))))).cuda().type(
                torch.cuda.FloatTensor)
            target_one_hot_tr = one_hot_tr.scatter_(1,
                                                    torch.cat((targets, targets_attack)).type(
                                                        torch.cuda.LongTensor).view(
                                                        [-1, 1]).data, 1)
            infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr)
            attack_output = attack_model(comb_inputs, infer_input_one_hot).view([-1])
            att_labels = np.zeros((inputs.size()[0] + inputs_attack.size()[0]))
            att_labels[:inputs.size()[0]] = 0.0
            att_labels[inputs.size()[0]:] = 1.0
            is_member_labels = torch.from_numpy(att_labels).type(torch.cuda.FloatTensor).to(accelerator.device)
            v_is_member_labels = torch.autograd.Variable(is_member_labels)
            classifier_targets = comb_targets.clone().view([-1]).type(torch.cuda.LongTensor)

            # 计算攻击损失
            loss_attack = criterion(attack_output, v_is_member_labels)
            attack_acc = np.mean(
                np.equal((attack_output.data.cpu().numpy() > 0.5), (v_is_member_labels.data.cpu().numpy() > 0.5)))
            all_acc += attack_acc
            all_loss += loss_attack
            loss_attack.backward()
            optimizer.step()
            i += 1
        accelerator.print(
            f'Advance [{epoch + 1}/{epochs}] Training Attack Model Loss: {all_loss / i} acc: {all_acc / i}', flush=True)
    return attack_model


def train_adv_target(config, model, optimizer, criterion, data, accelerator, epoch, scheduler,
                     shadow_log='target'):
    # 拆出模型相关工具
    target_model, attack_model = model
    target_model.train()
    attack_model.eval()

    # 拆分数据
    train_x, train_y, test_x, test_y = data
    train_data = torch.from_numpy(train_x).type(torch.FloatTensor)
    labels = torch.from_numpy(train_y).type(torch.LongTensor)

    batch_size = config.trainer.batch_size
    i = 0
    all_loss = 0
    target_model.train()

    for input_batch, target_batch in iterate_minibatches(train_data, labels, batch_size, shuffle=True):
        input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
        inputs, targets = torch.autograd.Variable(input_batch), torch.autograd.Variable(target_batch)

        # 作模型输出
        img = inputs
        logits = target_model(img)
        label = targets

        # 计算攻击模型指导量
        one_hot_tr = torch.from_numpy((np.zeros((logits.size()[0], logits.size(1))))).to(accelerator.device).type(
            torch.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, label.type(torch.LongTensor).view([-1, 1]).data, 1)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr).to(accelerator.device)
        attack_output = attack_model(logits, infer_input_one_hot)
        if torch.mean((attack_output)) - 0.5 > 0:
            attack_guide = abs(torch.log(1 - torch.mean((attack_output))) - torch.log(torch.tensor(0.5)))
            # attack_guide = torch.mean((attack_output)) - 0.5
        else:
            attack_guide = 0

        loss = criterion(logits, label) + 1 * attack_guide

        all_loss += loss
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        i += 1
    scheduler.step(epoch)
    all_loss = all_loss / i
    val_acc, train_acc = val_Accuracy(target_model, config.trainer.batch_size, data, accelerator)
    accelerator.print(
        f'Epoch [{epoch + 1}/{config.defence.defence_epochs}]  {shadow_log} Target Model Loss: {all_loss} val_acc: {val_acc} train_acc: {train_acc}',
        flush=True)
    return target_model, attack_model, float(val_acc)


def train_adv_attack(config, model, optimizer, criterion, tdata, sdata, accelerator, epoch, shadow_log='Target'):
    # 拆出模型相关工具
    target_model, attack_model = model
    target_model.eval()
    attack_model.train()
    # 拆分数据
    train_x, train_y, test_x, test_y = tdata
    strain_x, strain_y, _, _ = sdata
    r = np.arange(len(strain_x))
    np.random.shuffle(r)
    strain_x = strain_x[r]
    strain_y = strain_y[r]
    train_data = torch.from_numpy(strain_x).type(torch.FloatTensor)
    labels = torch.from_numpy(strain_y).type(torch.LongTensor)

    # 随机选择数据
    attack_data = torch.from_numpy(train_x).type(torch.FloatTensor)
    attack_labels = torch.from_numpy(train_y).type(torch.LongTensor)

    batch_size = config.trainer.batch_size
    i = 0
    all_acc = 0
    all_loss = 0
    num = 0
    for inputs, targets in iterate_minibatches(attack_data, attack_labels, batch_size, True):
        inputs_attack, targets_attack = inputs.to(accelerator.device), targets.to(accelerator.device)
        # 取出对应标签
        if (num + 1) * batch_size >= len(train_data):
            r = np.arange(len(strain_x))
            np.random.shuffle(r)
            strain_x = strain_x[r]
            strain_y = strain_y[r]
            num = 0
        inputs = train_data[num * batch_size:(num + 1) * batch_size].to(accelerator.device)
        targets = labels[num * batch_size:(num + 1) * batch_size].to(accelerator.device)

        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        inputs_attack, targets_attack = torch.autograd.Variable(inputs_attack), torch.autograd.Variable(
            targets_attack)

        # 作目标模型输出
        optimizer.zero_grad()
        img = inputs_attack
        inputs = inputs
        # inputs = torch.cat((generator_model(inputs)+inputs,inputs))
        targets = targets
        targets_attack = targets_attack

        outputs = target_model(inputs)
        outputs_non = target_model(img)

        # 拼接攻击模型输入与标签
        classifier_input = torch.cat((inputs, img))
        comb_inputs = torch.cat((outputs, outputs_non))
        comb_targets = torch.cat((targets, targets_attack)).view([-1, 1]).type(torch.FloatTensor)

        # attack_input = comb_inputs
        one_hot_tr = torch.from_numpy((np.zeros((comb_inputs.size()[0], outputs.size(1))))).to(accelerator.device).type(
            torch.FloatTensor)
        target_one_hot_tr = one_hot_tr.scatter_(1, torch.cat((targets, targets_attack)).type(
            torch.LongTensor).view(
            [-1, 1]).data, 1).to(accelerator.device)
        infer_input_one_hot = torch.autograd.Variable(target_one_hot_tr).to(accelerator.device)
        attack_output = attack_model(comb_inputs, infer_input_one_hot).view([-1])
        att_labels = np.zeros((inputs.size()[0] + img.size()[0]))
        att_labels[:inputs.size()[0]] = 0.0
        att_labels[inputs.size()[0]:] = 1.0
        is_member_labels = torch.from_numpy(att_labels).type(torch.FloatTensor).to(accelerator.device)
        v_is_member_labels = torch.autograd.Variable(is_member_labels)
        classifier_targets = comb_targets.clone().view([-1]).type(torch.LongTensor)

        # 计算攻击损失
        loss_attack = criterion(attack_output, v_is_member_labels)
        attack_acc = np.mean(
            np.equal((attack_output.data.cpu().numpy() > 0.5), (v_is_member_labels.data.cpu().numpy() > 0.5)))
        all_acc += attack_acc
        all_loss += loss_attack
        accelerator.backward(loss_attack)
        optimizer.step()
        num += 1
        i += 1
    all_acc = all_acc / i
    all_loss = all_loss / i
    accelerator.print(
        f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} Attack Model Loss: {all_loss} attack_acc:{all_acc}',
        flush=True)
    return target_model, attack_model


def train_dp_target(config, model, criterion, optimizer, data, accelerator, epoch, scheduler, shadow_log='target'):
    train_x, train_y, test_x, test_y = data
    train_x = train_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    test_x = test_x.astype(np.float32)
    test_y = test_y.astype(np.int32)

    n_in = train_x.shape
    n_out = len(np.unique(train_y))
    batch_size = config.trainer.batch_size
    if batch_size > len(train_y):
        batch_size = len(train_y)

    temp_loss = 0.0
    i = 0
    for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size, shuffle=True):
        input_batch, target_batch = torch.from_numpy(input_batch), torch.from_numpy(target_batch).type(torch.long)
        input_batch, target_batch = input_batch.to(accelerator.device), target_batch.to(accelerator.device)
        inputs, targets = torch.autograd.Variable(input_batch), torch.autograd.Variable(target_batch)
        # empty parameters in optimizer
        optimizer.zero_grad()
        outputs = model(inputs)
        # calculate loss value
        loss = criterion(outputs, targets)
        # back propagation
        loss.backward()
        # update paraeters in optimizer(update weight)
        optimizer.step()
        # 计算损失值
        temp_loss += loss.item()
        i += 1
    scheduler.step(epoch)
    # all_acc = all_acc / i
    all_loss = temp_loss / i
    val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
    accelerator.print(
        f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} Target Model Loss: {all_loss} val_acc: {val_acc} train_acc: {train_acc}',
        flush=True)
    return model, float(val_acc)


def train_relax_model(config, model, optimizer, criterion, data, accelerator, epoch, scheduler, alpha=0, upper=1,
                      shadow_log='target'):
    train_x, train_y, test_x, test_y = data
    data = (train_x, train_y, test_x, test_y)
    softmax = nn.Softmax(dim=1)
    crossentropy_soft = partial(CrossEntropy_soft, reduction='none')
    all_loss = 0
    i = 1
    model.train()
    for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, shuffle=True):
        inputs, targets = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
        inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
        outputs = model(inputs)
        loss_ce_full = criterion(outputs, targets)
        loss_ce = torch.mean(loss_ce_full)
        if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
            loss = (loss_ce - alpha).abs()
            # loss = -loss_ce
        else:
            pred = torch.argmax(outputs, dim=1)
            correct = torch.eq(pred, targets).float()
            confidence_target = softmax(outputs)[torch.arange(targets.size(0)), targets]
            confidence_target = torch.clamp(confidence_target, min=0., max=upper)
            confidence_else = (1.0 - confidence_target) / (config.model.num_classes - 1)
            onehot = one_hot_embedding(targets, num_classes=config.model.num_classes)
            soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, config.model.num_classes) \
                           + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, config.model.num_classes)
            loss = (1 - correct) * crossentropy_soft(outputs, soft_targets) - 1. * loss_ce_full
            loss = torch.mean(loss).abs()

            # with torch.no_grad():
            #     prob_gt = F.softmax(outputs, dim=1)[torch.arange(targets.size(0)), targets]
            #     prob_ngt = (1.0 - prob_gt) / (config.model.num_classes - 1)
            #     onehot = F.one_hot(targets, num_classes=config.model.num_classes)
            #     soft_labels = onehot * prob_gt.unsqueeze(-1).repeat(1, config.model.num_classes) \
            #                   + (1 - onehot) * prob_ngt.unsqueeze(-1).repeat(1, config.model.num_classes)
            # loss = criterion(outputs, soft_labels)
            # loss = torch.mean(loss)

        all_loss += loss.item()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        i += 1
    scheduler.step(epoch)
    all_loss = all_loss / i
    val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
    accelerator.print(
        f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} Targrt Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
        flush=True)
    return model, val_acc


def train_pub(config, model, optimizer, data, distill_labels, accelerator, epoch, t_softmax, alpha=1):
    train_x, train_y, test_x, test_y = data
    data = (train_x, train_y, test_x, test_y)
    all_loss = 0
    batch_size = config.trainer.batch_size
    true_criterion = nn.CrossEntropyLoss()
    i = 1
    model.train()
    len_t = len(train_x) // batch_size
    if len(train_x) % batch_size:
        len_t += 1

    for ind in range(len_t):

        inputs = train_x[ind * batch_size:(ind + 1) * batch_size]
        targets = distill_labels[ind * batch_size:(ind + 1) * batch_size]
        true_targets = train_y[ind * batch_size:(ind + 1) * batch_size]

        inputs, targets, true_targets = torch.tensor(inputs), torch.tensor(targets), torch.tensor(true_targets).type(torch.long)

        inputs, targets, true_targets = inputs.to(accelerator.device), targets.to(accelerator.device), true_targets.to(accelerator.device)

        inputs, targets, true_targets = torch.autograd.Variable(inputs), torch.autograd.Variable(
            targets), torch.autograd.Variable(true_targets)
        # inputs=inputs.to(accelerator.device)

        # compute output
        model.to(accelerator.device)
        outputs = model(inputs)

        loss = alpha * F.kl_div(F.log_softmax(outputs / t_softmax, dim=1), F.softmax(targets / t_softmax, dim=1)) + (1 - alpha) * true_criterion(outputs, true_targets)

        # measure loss
        all_loss += loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
    # scheduler.step(epoch)
    all_loss = all_loss / i
    val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
    accelerator.print(
        f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
        flush=True)
    return model, val_acc


# def distill_train(config, model, optimizer, criterion, data, accelerator, epoch, scheduler, uniform_reg=False):
#     train_x, train_y, _, _, test_x, test_y = data
#     data = (train_x, train_y, test_x, test_y)
#     all_loss = 0
#     i = 1
#     model.train()
#     for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, shuffle=True):
#         inputs, targets = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
#         inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
#
#         outputs = model(inputs)
#
#         if uniform_reg == True:
#             uniform_ = (torch.ones(config.trainer.batch_size, outputs.shape[1])).cuda()
#             t_softmax = 1
#             loss = criterion(outputs, targets) + 100 * F.kl_div(F.log_softmax(outputs / t_softmax, dim=1), F.softmax(uniform_ / t_softmax, dim=1))
#         else:
#             loss = criterion(outputs, targets)
#
#         all_loss += loss.item()
#         optimizer.zero_grad()
#         accelerator.backward(loss)
#         optimizer.step()
#     scheduler.step(epoch)
#     all_loss = all_loss / i
#     val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
#     accelerator.print(
#         f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] distill Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
#         flush=True)
#     return model, val_acc

# def train_DMP_model(config, model, optimizer, criterion, data, distill_label, accelerator, scheduler, epoch, shadow_log='Target'):
#
#     train_x, train_y, test_x, test_y = data
#     model.train()
#     distil_test_criterion=nn.CrossEntropyLoss()
#     distil_schedule=[60, 90, 150]
#     distil_lr=.1
#     distil_epochs=200
#     distil_best_acc=0
#     best_distil_test_acc=0
#     gamma=.1
#     t_softmax=1
#     for epoch in range(distil_epochs):
#         if epoch in distil_schedule:
#             distil_lr *= gamma
#         distil_optimizer = optimizer.SGD(model.parameters(), lr=distil_lr, momentum=0.99, weight_decay=1e-5)
#
#         distil_model, distil_tr_loss = train_pub(config, model, optimizer, data, distill_label, accelerator, epoch, scheduler, t_softmax=t_softmax, alpha=1)



# # 保存最佳模型组
# def split_save_checkpoint(state, is_best, acc, split_name, checkpoint):
#     if not os.path.isdir(os.path.join(checkpoint, split_name)):
#         os.makedirs(os.path.join(checkpoint, split_name))
#     if is_best:
#         filepath = os.path.join(checkpoint, split_name, 'split_model_best.pth.tar')
#         if os.path.exists(filepath):
#             tmp_ckpt = torch.load(filepath)
#             best_acc = tmp_ckpt['best_acc']
#             if best_acc > acc:
#                 return
#         torch.save(state, filepath)
#
# def split_train(config, model, optimizer, criterion, data, accelerator, epoch, scheduler, shadow_log='target'):
#     train_x, train_y, test_x, test_y = data
#     data = (train_x, train_y, test_x, test_y)
#     all_loss = 0
#     i = 1
#     model.train()
#     for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, shuffle=True):
#         inputs, targets = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
#         inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         all_loss += loss.item()
#         optimizer.zero_grad()
#         accelerator.backward(loss)
#         optimizer.step()
#     scheduler.step(epoch)
#     all_loss = all_loss / i
#     val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
#     accelerator.print(
#         f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} Split Ai Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
#         flush=True)
#     return model, val_acc
#
#
# def distill_train(config, model, teacher_model, optimizer, criterion, data, accelerator, epoch, scheduler,
#                   shadow_log='target'):
#     train_x, train_y, _, _, test_x, test_y = data
#     data = (train_x, train_y, test_x, test_y)
#     all_loss = 0
#     i = 1
#     model.train()
#     teacher_model.eval()
#     for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, shuffle=True):
#         inputs, targets = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
#         inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
#         target_labels = F.softmax(teacher_model(inputs)).argmax(dim=1).detach().cpu().numpy()
#         outputs = model(inputs)
#         one_hot_tr = torch.zeros(inputs.size()[0], outputs.size()[1]).to(accelerator.device, torch.float)
#         target_one_hot = one_hot_tr.scatter_(1, target_labels.view([-1, 1]), 1)
#
#         # loss = criterion(outputs, target_labels)
#         loss = (-torch.sum(targets * torch.log(F.softmax(outputs, dim=1)))) / inputs.shape[0]
#
#         all_loss += loss.item()
#         optimizer.zero_grad()
#         accelerator.backward(loss)
#         optimizer.step()
#     scheduler.step(epoch)
#     all_loss = all_loss / i
#     val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
#     accelerator.print(
#         f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} distill Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
#         flush=True)
#     return model, val_acc


# def split_train(config, model, optimizer, criterion, data, accelerator, epoch, scheduler, shadow_log='target'):
#     train_x, train_y, _, _, test_x, test_y = data
#     data = (train_x, train_y, test_x, test_y)
#     all_loss = 0
#     i = 1
#     model.train()
#     for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, shuffle=True):
#         inputs, targets = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
#         inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         all_loss += loss.item()
#         optimizer.zero_grad()
#         accelerator.backward(loss)
#         optimizer.step()
#     scheduler.step(epoch)
#     all_loss = all_loss / i
#     val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
#     accelerator.print(
#         f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} Split Ai Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
#         flush=True)
#     return model, val_acc
#
#
# def distill_train(config, model, teacher_model, optimizer, criterion, data, accelerator, epoch, scheduler,
#                   shadow_log='target'):
#     train_x, train_y, _, _, test_x, test_y = data
#     data = (train_x, train_y, test_x, test_y)
#     all_loss = 0
#     i = 1
#     model.train()
#     teacher_model.eval()
#     for input_batch, target_batch in iterate_minibatches(train_x, train_y, config.trainer.batch_size, shuffle=True):
#         inputs, targets = torch.tensor(input_batch), torch.tensor(target_batch).type(torch.long)
#         inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)
#         target_labels = F.softmax(teacher_model(inputs)).argmax(dim=1).detach().cpu().numpy()
#         outputs = model(inputs)
#         one_hot_tr = torch.zeros(inputs.size()[0], outputs.size()[1]).to(accelerator.device, torch.float)
#         target_one_hot = one_hot_tr.scatter_(1, target_labels.view([-1, 1]), 1)
#
#         # loss = criterion(outputs, target_labels)
#         loss = (-torch.sum(targets * torch.log(F.softmax(outputs, dim=1)))) / inputs.shape[0]
#
#         all_loss += loss.item()
#         optimizer.zero_grad()
#         accelerator.backward(loss)
#         optimizer.step()
#     scheduler.step(epoch)
#     all_loss = all_loss / i
#     val_acc, train_acc = val_Accuracy(model, config.trainer.batch_size, data, accelerator)
#     accelerator.print(
#         f'Epoch [{epoch + 1}/{config.defence.defence_epochs}] {shadow_log} distill Model Loss: {all_loss} , Train acc: {train_acc},Val acc: {val_acc} ',
#         flush=True)
#     return model, val_acc
