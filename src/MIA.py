import numpy as np
import torch
import torch.utils.data
from accelerate.logging import get_logger
from torch import nn
from torch.utils.data import DataLoader


def GiveMemguard(scores):
  """ Given confidence vectors, perform memguard post processing to protect from membership inference.

  Note that this defense assumes the strongest defender that can make arbitrary changes to the confidence vector
  so long as it does not change the label. We as well have the (weaker) constrained optimization that will be
  released at a future data.

  Args:
    scores: confidence vectors as 2d numpy array

  Returns: 2d scores protected by memguard.

  """
  scores = scores.cpu()
  n_classes = scores.shape[1]
  epsilon = 1e-5
  on_score = (1. / n_classes) + epsilon
  off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
  predicted_labels = np.argmax(scores.detach().numpy(), axis=-1)
  defended_scores = np.ones_like(scores.detach().numpy()) * off_score
  defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
  return torch.from_numpy(defended_scores)



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


class black_box_benchmarks(object):
    def __init__(self, shadow_train_performance, shadow_test_performance,
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        '''
        self.num_classes = num_classes

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)

        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    def clipDataTopX(self, dataToClip, top):
        res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
        return np.array(res)

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        print(
            'For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(
                acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc))
        return

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem += np.sum(t_tr_values >= thre)
        t_te_non_mem += np.sum(t_te_values < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name, acc=mem_inf_acc))
        return

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        # if (all_methods) or ('confidence' in benchmark_methods):
        #     self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('confidence' in benchmark_methods):
            self.s_tr_outputs = self.clipDataTopX(self.s_tr_outputs, 1)
            self.s_te_outputs = self.clipDataTopX(self.s_te_outputs, 1)
            self.t_tr_outputs = self.clipDataTopX(self.t_tr_outputs, 1)
            self.t_te_outputs = self.clipDataTopX(self.t_te_outputs, 1)
            self._mem_inf_thre('confidence', self.s_tr_outputs, self.s_te_outputs, self.t_tr_outputs, self.t_te_outputs)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr,
                               -self.t_te_m_entr)
        return


# all attack
def FourAttack(config, model, shadow_model, target_dataset, shadow_dataset, device, memguard=False):
    def Outpred(target_model, train_ta_x, train_ta_y, device,memguard=False):
        Preds = []
        batch_size = 10
        target_model.eval()
        for batch, _ in iterate_minibatches(train_ta_x, train_ta_y, batch_size, False):
            batch = torch.tensor(batch).type(torch.FloatTensor)
            batch = batch.to(device)
            output = target_model(batch)
            preds_tensor = nn.functional.softmax(output, dim=1)
            if memguard == True:
                preds_tensor = GiveMemguard(preds_tensor)
            Preds.append(preds_tensor.detach().cpu().numpy())
        Preds = np.vstack(Preds)
        Preds = Preds.astype('float32')

        target_train_performance = (Preds, train_ta_y)

        return target_train_performance

    # get use data
    train_ta_x, train_ta_y, test_ta_x, test_ta_y = target_dataset
    train_sh_x, train_sh_y, test_sh_x, test_sh_y = shadow_dataset
    s = len(test_ta_x)
    train_ta_x, train_ta_y = train_ta_x[:s], train_ta_y[:s]
    s = len(test_sh_x)
    train_sh_x, train_sh_y = train_sh_x[:s], train_sh_y[:s]
    ## 测试attack
    ####################################################################################################
    print('###################### 测试attack ######################')
    target_train_performance = Outpred(model, train_ta_x, train_ta_y, device,memguard=memguard)
    target_test_performance = Outpred(model, test_ta_x, test_ta_y, device,memguard=memguard)
    shadow_train_performance = Outpred(shadow_model, train_sh_x, train_sh_y, device,memguard=False)
    shadow_test_performance = Outpred(shadow_model, test_sh_x, test_sh_y, device,memguard=False)

    num_classes = config.model.num_classes
    MIA = black_box_benchmarks(shadow_train_performance, shadow_test_performance, target_train_performance,
                               target_test_performance, num_classes=num_classes)
    MIA._mem_inf_benchmarks()


class MIAFourAttack(object):
    def __init__(self, num_classes: int, defence_model: torch.nn.Module, defence_shadow_model: torch.nn.Module, t_train_loader: DataLoader, t_test_loader: DataLoader, s_train_loader: DataLoader, s_test_loader: DataLoader):
        self.logger = get_logger(MIAFourAttack.__name__)
        self.num_classes = num_classes
        self.logger.info('###################### 测试attack ######################')
        target_train_performance = self.out_pred(defence_model, t_train_loader)
        target_test_performance = self.out_pred(defence_model, t_test_loader)
        shadow_train_performance = self.out_pred(defence_shadow_model, s_train_loader)
        shadow_test_performance = self.out_pred(defence_shadow_model, s_test_loader)
        # compute attack performance
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels).astype(int)

        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def member_inference_benchmarks(self, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        # if (all_methods) or ('confidence' in benchmark_methods):
        #     self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('confidence' in benchmark_methods):
            self.s_tr_outputs = self.clipDataTopX(self.s_tr_outputs, 1)
            self.s_te_outputs = self.clipDataTopX(self.s_te_outputs, 1)
            self.t_tr_outputs = self.clipDataTopX(self.t_tr_outputs, 1)
            self.t_te_outputs = self.clipDataTopX(self.t_te_outputs, 1)
            self._mem_inf_thre('confidence', self.s_tr_outputs, self.s_te_outputs, self.t_tr_outputs, self.t_te_outputs)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)
        return

    @staticmethod
    def _thre_setting(tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + 1 - t_te_acc)
        self.logger.info(f'For membership inference attack via correctness, the attack acc is {mem_inf_acc:.3f}, with train acc {t_tr_acc:.3f} and test acc {t_te_acc:.3f}')

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem += np.sum(t_tr_values >= thre)
        t_te_non_mem += np.sum(t_te_values < thre)
        mem_inf_acc = 0.5 * (t_tr_mem / (len(self.t_tr_labels) + 0.0) + t_te_non_mem / (len(self.t_te_labels) + 0.0))
        self.logger.info(f'For membership inference attack via {v_name}, the attack acc is {mem_inf_acc:.3f}')

    @staticmethod
    def _log_value(probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))

    @staticmethod
    def clipDataTopX(dataToClip, top):
        res = [sorted(s, reverse=True)[0:top] for s in dataToClip]
        return np.array(res)

    @staticmethod
    def _entr_comp(probs):
        return np.sum(np.multiply(probs, MIAFourAttack._log_value(probs)), axis=1)

    @staticmethod
    def _m_entr_comp(probs, true_labels):
        log_probs = MIAFourAttack._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = MIAFourAttack._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    @staticmethod
    @torch.no_grad()
    def out_pred(target_model: torch.nn.Module, data_loader: DataLoader):
        target_model.eval()
        predictions = []
        references = []
        for i, (x, y) in enumerate(data_loader):
            output = target_model(x)
            predictions.append(nn.functional.softmax(output, dim=1).detach().cpu().numpy())
            references.append(y.cpu().numpy())
        predictions = np.vstack(predictions).astype('float32')
        references = np.concatenate(references)
        return predictions, references
