import copy
import os
import sys
from datetime import datetime
import torch.nn as nn
import yaml
import torch
import random
from accelerate import Accelerator
from easydict import EasyDict
from objprint import objstr
from timm.optim import optim_factory
from src.preprocess import GiveTestData, same_seeds, Logger
from src.utils import LinearWarmupCosineAnnealingLR, load_pretrain_model, give_model, cosine_scheduler, common_params, give_mask
import src.model_dic as model_dic
from src.MIA import FourAttack
from src.train_val import val_Accuracy, train_one_epoch, val_one_epoch, iterate_minibatches, advance_train_generator


def train_model(config, target_model, data, accelerator, load=True, shadow=False):
    tdataset = data
    if shadow == True:
        shadow_log = 'shadow'
    else:
        shadow_log = 'target'
    try:
        if load == True:
            target_model.load_state_dict(torch.load(
                f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/{shadow_log}/best/pytorch_model.bin"))
            val_acc, train_acc = val_Accuracy(target_model, config.trainer.batch_size, tdataset, accelerator)
            accelerator.print(f'Train acc: {train_acc}, Test acc: {val_acc}')
            if val_acc > 0.2:
                return target_model
    except:
        pass
    # 第二步，构建损失函数和优化器
    loss_functions = {
        'CrossEntropyLoss': nn.CrossEntropyLoss(),
    }
    optimizer = optim_factory.create_optimizer_v2(target_model, opt=config.trainer.optimizer,
                                                  weight_decay=config.trainer.weight_decay,
                                                  lr=config.trainer.lr, betas=(0.9, 0.95))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.trainer.warmup,
                                              max_epochs=config.defence.defence_epochs * config.defence.defence_epochs)
    target_model, optimizer, scheduler = accelerator.prepare(target_model, optimizer, scheduler)
    starting_epoch = 0
    batch_size = config.trainer.batch_size
    step = 0
    val_step = 0
    best_acc = 0
    for epoch in range(starting_epoch, config.trainer.num_epochs):
        # 训练目标模型
        step = train_one_epoch(config, target_model, batch_size, loss_functions, tdataset,
                               optimizer, accelerator,scheduler, epoch, step)
        # 验证目标模型
        mean_acc, train_acc, val_step = val_one_epoch(config, target_model, loss_functions, tdataset, accelerator,
                                                      epoch, val_step)

        accelerator.print(
            f'{shadow_log} Epoch [{epoch + 1}/{config.trainer.num_epochs}] train acc: {train_acc} mean acc: {mean_acc}')

        # 保存模型
        if mean_acc > best_acc:
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/{shadow_log}/best")
            torch.save(target_model.state_dict(),
                       f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/{shadow_log}/best/pytorch_model.bin")
            best_acc = mean_acc

    target_model.load_state_dict(
        torch.load(f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/{shadow_log}/best/pytorch_model.bin"))
    val_acc, train_acc = val_Accuracy(target_model, batch_size, tdataset, accelerator)
    accelerator.print(f'Train acc: {train_acc}, Test acc: {val_acc}')
    return target_model


def self_defence(config, target_model, teacher_model, generator_model, data, accelerator, dataset, shadow=False):
    train_x, train_y, test_x, test_y = data
    # 第一步，构建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer_s = optim_factory.create_optimizer_v2(target_model, opt=config.trainer.optimizer,
                                                    weight_decay=config.trainer.weight_decay,
                                                    lr=config.defence.target_lr, betas=(0.9, 0.95))
    optimizer_t = optim_factory.create_optimizer_v2(teacher_model, opt=config.trainer.optimizer,
                                                    weight_decay=config.trainer.weight_decay,
                                                    lr=config.defence.teacher_lr, betas=(0.9, 0.95))
    optimizer_g = optim_factory.create_optimizer_v2(generator_model, opt=config.trainer.optimizer,
                                                    weight_decay=config.trainer.weight_decay,
                                                    lr=config.defence.generator_lr, betas=(0.9, 0.95))
    # 余弦退火
    scheduler_s = LinearWarmupCosineAnnealingLR(optimizer_s, warmup_epochs=config.trainer.warmup,
                                                max_epochs=config.defence.defence_epochs * config.defence.defence_epochs)
    # EMA
    momentum_schedule = cosine_scheduler(config.defence.momentum_teacher, 1, config.defence.defence_epochs,
                                         train_x.shape[0])
    # 多卡加载
    target_model, teacher_model, generator_model, optimizer_s, optimizer_t, optimizer_g, scheduler_s = accelerator.prepare(
        target_model,
        teacher_model, generator_model,
        optimizer_s, optimizer_t,
        optimizer_g, scheduler_s)
    best_acc = 0
    if shadow == True:
        shadow_log = 'shadow'
    else:
        shadow_log = 'target'
    # 拆分数据
    train_data = torch.from_numpy(train_x).type(torch.FloatTensor).to(accelerator.device)
    labels = torch.from_numpy(train_y).type(torch.LongTensor).to(accelerator.device)
    step = 0
    params_q, params_k = common_params(target_model, teacher_model, accelerator)
    # 生成模型预训练
    generator_model = advance_train_generator(train=True, config=config, generator_model=generator_model,
                                              data=data, optimizer=optimizer_g, accelerator=accelerator, epochs=3)

    is_copy = False
    for epoch in range(0, config.defence.defence_epochs):
        all_loss = 0
        t_loss = 0
        s_loss = 0
        g_loss = 0
        i = 0
        for inputs, targets in iterate_minibatches(train_data, labels, config.trainer.batch_size, True):
            # 教师模型训练
            optimizer_t.zero_grad()
            teacher_model.train()

            teacher_output = teacher_model(inputs)
            teacher_loss = criterion(teacher_output, targets)
            t_loss += teacher_loss.item()
            accelerator.backward(teacher_loss)
            optimizer_t.step()

            # 生成模型训练
            optimizer_g.zero_grad()
            generator_model.train()
            teacher_model.eval()
            generator_output = generator_model(inputs)
            generator_teacher_output = teacher_model(generator_output)

            loss1 = criterion(generator_teacher_output, targets)  # 接近主label
            loss2 = - torch.log(
                torch.abs(torch.cosine_similarity(generator_output.unsqueeze(1), inputs.unsqueeze(0))).mean())  # 远离训练图像
            generator_loss = config.defence.generator_altho1 * loss1 + config.defence.generator_altho2 * loss2

            g_loss += generator_loss.item()
            accelerator.backward(generator_loss)
            optimizer_g.step()

            # 目标模型训练
            if is_copy:
                target_model.load_state_dict(old_state_dict)
                is_copy = False
            target_model.train()
            generator_model.eval()
            optimizer_s.zero_grad()
            generator_model.eval()
            if epoch < config.trainer.warmup:
                target_label_output = target_model(inputs)
                student_loss = criterion(target_label_output, targets)
            else:
                _, teacher_label = torch.max(teacher_output.detach().data, 1)

                if 'CIFAR' in dataset:
                    n = random.uniform(0, 1)
                    if n < config.defence.alpka:
                        target_input = torch.cat((generator_output, inputs), dim=0)
                        target_label = torch.cat((teacher_label, targets), dim=0)
                    else:
                        target_input = torch.cat((generator_output, generator_output), dim=0)
                        target_label = torch.cat((teacher_label, teacher_label), dim=0)

                    target_generator_output = target_model(target_input.detach())
                    student_loss = config.defence.target_altho * criterion(target_generator_output, target_label)
                else:
                    # 基于生成数据训练
                    target_generator_output = target_model(generator_output.detach())
                    student_loss = config.defence.target_altho * criterion(target_generator_output, teacher_label)

            s_loss += student_loss.item()
            accelerator.backward(student_loss)
            optimizer_s.step()


            all_loss += student_loss.item() + generator_loss.item()
            # EMA
            if epoch > config.trainer.warmup:
                if (epoch - config.trainer.warmup) % config.defence.ema_epoch == 0:
                    if 'CIFAR' in dataset:
                        with torch.no_grad():
                            m = momentum_schedule[step]  # momentum parameter
                            for param_q, param_k in zip(params_q, params_k):
                                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
                    else:
                        t_model = copy.deepcopy(target_model)
                        old_state_dict = t_model.state_dict().copy()
                        is_copy = True
                        teacher_model.load_state_dict(target_model.state_dict())
            step += 1
            i += 1
        scheduler_s.step(epoch)

        val_acc, train_acc = val_Accuracy(target_model, config.trainer.batch_size, data, accelerator)
        t_val_acc, t_train_acc = val_Accuracy(teacher_model, config.trainer.batch_size, data, accelerator)
        accelerator.print(
            f'Epoch [{epoch}/{config.defence.defence_epochs}] TrainAcc:{train_acc}, TestAcc:{val_acc}, TotalLoss:{all_loss / i}, GLoss:{g_loss / i}, SLoss:{s_loss / i}')
        accelerator.print(
            f'Epoch [{epoch}/{config.defence.defence_epochs}] t_TrainAcc:{t_train_acc}, t_TestAcc:{t_val_acc}, TLoss:{t_loss / i}')

        if val_acc > best_acc:
            accelerator.print('Get Save!')
            best_acc = val_acc
            accelerator.save_state(
                output_dir=f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/Defence_{shadow_log}/best")
            torch.save(target_model.state_dict(),
                       f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/Defence_{shadow_log}/best/Target.pth")
            torch.save(teacher_model.state_dict(),
                       f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/Defence_{shadow_log}/best/Teacher.pth")
            torch.save(generator_model.state_dict(),
                       f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/Defence_{shadow_log}/best/generator.pth")
    defence_model = load_pretrain_model(
        f"{os.getcwd()}/model_store/{config.finetune.checkpoint}/Defence_{shadow_log}/best/Target.pth",
        target_model,
        accelerator)
    val_acc, train_acc = val_Accuracy(defence_model, config.trainer.batch_size, tdataset, accelerator)
    accelerator.print(f'Train acc: {train_acc}, Test acc: {val_acc}')
    return defence_model


if __name__ == '__main__':
    # 读取配置
    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    same_seeds(config.trainer.seed)
    logging_dir = os.getcwd() + '/logs/' + str(datetime.now())
    accelerator = Accelerator(cpu=False, log_with=["tensorboard"], project_dir=logging_dir)
    Logger(logging_dir if accelerator.is_local_main_process else None)
    accelerator.init_trackers(os.path.split(__file__)[-1].split(".")[0])
    accelerator.print(objstr(config))

    # 加载数据集
    accelerator.print('加载数据集...')
    gt = GiveTestData(config.preprocess)
    tdataset = gt.giveData(ofTarget=True, preprocess=config.preprocess.preprocess)
    sdataset = gt.giveData(ofTarget=False, preprocess=False)
    train_x, train_y, test_x, test_y = tdataset

    # 加载相关模型
    # 1. 目标模型
    accelerator.print('加载模型...')
    target_model, shadow_model = give_model(config)
    target_teacher_model, shadow_teacher_model = give_model(config)

    if 'CIFAR' in config.preprocess.dataset:
        generator_model = model_dic.Generator(in_channels=config.model.in_channels)
    else:
        generator_model = model_dic.LinearGenerator(in_features=config.model.in_features)

    defence_model = self_defence(config=config, target_model=target_model, teacher_model=target_teacher_model,
                                 generator_model=generator_model,
                                 data=tdataset,
                                 accelerator=accelerator,
                                 dataset=config.preprocess.dataset)

    # 非敌手防御验证
    target_model, shadow_model = give_model(config)
    target_model = train_model(config=config, target_model=target_model, data=tdataset, accelerator=accelerator,
                               shadow=False, load=True)  # 要先训练一次，然后改成True
    shadow_model = train_model(config=config, target_model=shadow_model, data=sdataset, accelerator=accelerator,
                               shadow=True, load=True) # 要先训练一次，然后改成True
    # 验证效果(查看原始风险)
    FourAttack(config, target_model , shadow_model, tdataset, sdataset, accelerator.device)
    # 验证效果(查看防御后风险)
    FourAttack(config, defence_model, shadow_model, tdataset, sdataset, accelerator.device)
