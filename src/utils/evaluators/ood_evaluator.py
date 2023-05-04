import csv
import os
from typing import Dict, List

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.postprocessors import BasePostprocessor
from utils.postprocessors.utils import get_postprocessor_abbrv

from .base_evaluator import BaseEvaluator
from .metrics import compute_all_metrics


class OODEvaluator(BaseEvaluator):
    def __init__(self, eval_args):
        """OOD Evaluator.

        Args:
            eval_args: args
        """
        super(OODEvaluator, self).__init__(eval_args)
        self.id_pred = None
        self.id_conf = None
        self.id_gt = None

    def eval_ood(self, net: nn.Module, id_data_loader: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, DataLoader],
                 postprocessor: BasePostprocessor):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        # load training in-distribution data
        # assert 'test' in id_data_loader, \
        #     'id_data_loaders should have the key: test!'
        dataset_name = list(id_data_loader.keys())[0]
        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(
            net, id_data_loader['test'])
        self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        # if self.config.postprocessor.APS_mode:
        #     self.hyperparam_search(net, [id_pred, id_conf, id_gt],
        #                            ood_data_loaders, postprocessor)

        # load nearood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='nearood')
        # load farood data and compute ood metrics
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders,
                       postprocessor,
                       ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, DataLoader],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, ood_dl)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        self._save_csv(metrics_mean, dataset_name=ood_split)

    def eval_ood_(self, net: nn.Module, data_loader: Dict[str, DataLoader],
                 postprocessor: BasePostprocessor):
        metrics_list = []
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()
        net_name = 'resnet18' if type(net).__name__ == 'ResNet18_32x32' else 'unknown_net'
        postprocessor_name = get_postprocessor_abbrv(type(postprocessor).__name__)
        if postprocessor_name in ['odin', 'mds']:
            if not getattr(postprocessor, 'preprocessing'):
                postprocessor_name = postprocessor_name + '_2'
        base_dataset_name = list(data_loader.keys())[0].split('-')[0]
        datasets = []
        for dataset_name, dataloader in data_loader.items():
            print(f'Performing inference on {dataset_name} dataset...',
                    flush=True)
            ood_pred, ood_conf, ood_gt = postprocessor.inference(net, dataloader)
            self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            print(f'Computing metrics on {dataset_name} dataset...')

            datasets.append(dataset_name.split('-')[1])
            ood_metrics, data = compute_all_metrics(ood_conf, ood_gt, ood_pred)
            base = f'{net_name}-{postprocessor_name}-{base_dataset_name}-{datasets[-1]}'
            self._save_csv(ood_metrics, base=base)
            metrics_list.append(ood_metrics)
            os.makedirs(f'./features/{net_name}-{postprocessor_name}-{base_dataset_name}', exist_ok=True)
            np.savez(f'./features/{net_name}-{postprocessor_name}-{base_dataset_name}/{datasets[-1]}', fpr=data['fpr'], tpr=data['tpr'],
                        precision_in=data['precision_in'], recall_in=data['recall_in'],
                        precision_out=data['precision_out'], recall_out=data['recall_out'])

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        base = f'{net_name}-{postprocessor_name}-{base_dataset_name}-{"-".join(datasets)}'
        self._save_csv(metrics=metrics_mean, base=base)

    def _save_csv(self, metrics, base):
        [fpr, auroc, aupr_in, aupr_out,
         ccr_4, ccr_3, ccr_2, ccr_1, accuracy, best_error, best_delta] \
         = metrics
        #  ccr_4, ccr_3, ccr_2, ccr_1, accuracy, precision, recall, f1, support, average_precision, best_error, best_delta] \

        write_content = {
            'base': base,
            'FPR@95': '{:.6f}'.format(100 * fpr),
            'AUROC': '{:.6f}'.format(100 * auroc),
            'AUPR_IN': '{:.6f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.6f}'.format(100 * aupr_out),
            'CCR_4': '{:.6f}'.format(100 * ccr_4),
            'CCR_3': '{:.6f}'.format(100 * ccr_3),
            'CCR_2': '{:.6f}'.format(100 * ccr_2),
            'CCR_1': '{:.6f}'.format(100 * ccr_1),
            'ACC': '{:.6f}'.format(100 * accuracy),
            # 'PREC': '{:.2f}'.format(100 * precision),
            # 'REC': '{:.2f}'.format(100 * recall),
            # 'F1': '{:.2f}'.format(100 * f1),
            # 'SUPP': '{:.2f}'.format(100),
            # 'AVGP': '{:.2f}'.format(100 * average_precision),
            'ERROR': '{:.6f}'.format(100 * best_error),
            'DELTA': '{:.6f}'.format(best_delta)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.6f}, AUROC: {:.6f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.6f}, AUPR_OUT: {:.6f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('CCR: {:.6f}, {:.6f}, {:.6f}, {:.6f},'.format(
            ccr_4 * 100, ccr_3 * 100, ccr_2 * 100, ccr_1 * 100),
              end=' ',
              flush=True)
        print('ACC: {:.6f},'.format(accuracy * 100), end=' ', flush=True)
        # print('PREC: {:.2f},'.format(precision * 100), end=' ', flush=True)
        # print('REC: {:.2f},'.format(recall * 100), flush=True)
        # print('F1: {:.2f},'.format(f1 * 100), end=' ', flush=True)
        # print('SUPP: {:.2f},'.format(support), end=' ', flush=True)
        # print('AVGP: {:.2f},'.format(average_precision * 100), end=' ', flush=True)
        print('ERROR: {:.6f}, DELTA: {:.6f}'.format(best_error * 100, best_delta), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join('features', 'ood.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)

    def _save_scores(self, pred, conf, gt, save_name):
        save_dir = os.path.join('features', 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1):
        """Returns the accuracy score of the labels and predictions.

        :return: float
        """
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()
        self.id_pred, self.id_conf, self.id_gt = postprocessor.inference(
            net, data_loader)
        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def report(self, test_metrics):
        print('Completed!', flush=True)

    def hyperparam_search(
        self,
        net: nn.Module,
        id_list: List[np.ndarray],
        val_data_loader,
        postprocessor: BasePostprocessor,
    ):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []
        count = 0
        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
            count += 1
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])
        hyperparam_combination = self.recursive_generator(
            hyperparam_list, count)
        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)
            [id_pred, id_conf, id_gt] = id_list

            ood_pred, ood_conf, ood_gt = postprocessor.inference(
                net, val_data_loader)
            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print('Hyperparam:{}, auroc:{}'.format(hyperparam,
                                                   aps_dict[index]))
            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])
        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))

    def recursive_generator(self, list, n):
        if n == 1:
            results = []
            for x in list[0]:
                k = []
                k.append(x)
                results.append(k)
            return results
        else:
            results = []
            temp = self.recursive_generator(list, n - 1)
            for x in list[n - 1]:
                for y in temp:
                    k = y.copy()
                    k.append(x)
                    results.append(k)
            return results
