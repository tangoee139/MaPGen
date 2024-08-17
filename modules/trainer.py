import os
from abc import abstractmethod

import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from numpy import inf
from .metrics_clinical import CheXbertMetrics
import copy
from .optims import LinearWarmupCosineLRScheduler
import re
import jieba
from medical_report_helper.report_parser import ReportParser
from medical_report_helper.util import post_label_str, encode_labels, part_supplement, decode_labels, score
from medical_report_helper.socre_ziyu import socre, parse_label


class BaseTrainer(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model = model
        self.device = device
        self.is_main_process = is_main_process
        if self.args.dataset_name != 'mri':
            self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: {}".format(num_parameters))
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = 0.999
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )
        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'val_' + args.monitor_metric

        self.mnt_best = 0 
        self.log_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                # for different shuffling
                self.train_dataloader.sampler.set_epoch(epoch)

            result = self._train_epoch_blip(epoch)
            dist.barrier()
            if epoch>=4:
                result = self.eval_blip(result,epoch)

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                path = '/mnt/nvme_share/XXX/KneeMRG_qwen2/results/' + 'model_mask_prompt640_50_' + str(epoch) + '.pth'
                torch.save(self.model.module.state_dict(), path)
            #     if log[self.mnt_metric] >= self.mnt_best:
            #         self.mnt_best = log[self.mnt_metric]
            #         self.log_best = copy.deepcopy(log)
            #         best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            #         torch.save(self.model.module.state_dict(), best_path)
            #         print("Saving current best to {}".format(best_path))
            #     else:
            #         current_path = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
            #         torch.save(self.model.module.state_dict(), current_path)
            #         print("Saving current model to {}".format(current_path))

            # print logged information 
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

        if self.is_main_process:
            print('Best results w.r.t {}:'.format(self.mnt_metric))
            for key, value in self.log_best.items():
                print('\t{:15s}: {}'.format(str(key), value))

class Trainer(BaseTrainer):
    def __init__(self, model, criterion_cls, metric_ftns, args, train_dataloader, val_dataloader, test_dataloader, device, is_main_process):
        super(Trainer, self).__init__(model, criterion_cls, metric_ftns, args, device, is_main_process)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer, 
            self.args.epochs,
            self.args.total_steps, 
            self.args.min_lr, 
            self.args.init_lr, 
            decay_rate=None, 
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )
        keywords_path = "/mnt/nvme_share/XXX/data/膝关节概念关键词表（新）.xlsx"
        self.reportparser = ReportParser(keywords_path)
        num_label = 300
        label_file = '/mnt/nvme_share/XXX/data/label_set_largerData.csv'
        self.label_set_list = pd.read_csv(label_file)['label'][:num_label].tolist()
        label_simple_file = '/mnt/nvme_share/XXX/data/label_set_largerData_removeKey.csv'
        self.label_set_list_simple = pd.read_csv(label_simple_file)['label_simple'][:num_label].tolist()
        # 列名
        self.column_origin = 'original'
        self.column_generate = 'generated'

    def _train_epoch_blip(self, epoch):
        train_loss = 0
        self.model.train()
        for batch_idx, (images, captions, cls_labels) in enumerate(self.train_dataloader): ####for condition
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            cur_step = batch_idx+1 + (epoch-1)*int(self.args.total_steps/self.args.epochs)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=cur_step)
            loss_lm, loss_cls = self.model(images, captions, cls_labels, self.criterion_cls, batch_idx)
            loss = loss_lm + self.args.cls_weight*loss_cls
            #loss = loss_lm
            if batch_idx%20 == 0:
                print("epoch: {}, {}/{} loss: {}, loss_lm: {}, loss_cls: {}".format(epoch, batch_idx, len(self.train_dataloader), loss.item(), loss_lm.item(), self.args.cls_weight*loss_cls.item()))
                print('learning rate: {:.15f}'.format(self.optimizer.param_groups[0]['lr']))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        return log

    def eval_blip(self, log, epoch):
        self.model.module.eval()

        with torch.no_grad():
            rows, val_gts, val_res =[], [], []
            file_name = self.args.save_dir + '/test_output_val{}.csv'.format(epoch)
            for batch_idx, (images, captions, cls_labels) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.to(self.device)
                ground_truths = captions
                reports = self.model.module.generate(images, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len, batch_idx=batch_idx)
  
                val_res.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt))) for i, gt  in enumerate(reports)])
                val_gts.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt))) for i, gt  in enumerate(ground_truths)])

                for tes, gts in zip(reports, ground_truths):
                    new_row = {'generated': tes.replace(';', '\n'), 'original': gts.replace(';', '\n')}
                    rows.append(new_row)

            df = pd.DataFrame(rows)
            df.to_csv(file_name)

            val_met = self.metric_ftns({i: [gt] for i, gt  in enumerate(val_gts)},
                                       {i: [res] for i, res in enumerate(val_res)})

            if self.args.dataset_name !='mri':
                val_ce = self.chexbert_metrics.compute(val_gts, val_res)
                log.update(**{'val_' + k: v for k, v in val_ce.items()})
            else:
                val_met.update(socre(df, self.column_origin, self.column_generate, self.label_set_list,self.label_set_list_simple))
            log.update(**{'val_' + k: v for k, v in val_met.items()})


        with torch.no_grad():
            rows, test_gts, test_res =[], [], []
            file_name = self.args.save_dir + '/test_output_test{}.csv'.format(epoch)

            for batch_idx, (images, captions, cls_labels) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                cls_labels = cls_labels.numpy().tolist()
                ground_truths = captions
                reports = self.model.module.generate(images, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len, batch_idx=batch_idx)

                test_res.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt)))  for i, gt  in enumerate(reports)])
                test_gts.extend([' '.join(jieba.cut(re.sub(r'[^\u4e00-\u9fa5]', '', gt)))  for i, gt  in enumerate(ground_truths)])

                for tes, gts in zip(reports, ground_truths):
                    new_row = {'generated': tes.replace(';', '\n'), 'original': gts.replace(';', '\n')}
                    rows.append(new_row)
            df = pd.DataFrame(rows)
            df.to_csv(file_name)
            test_met = self.metric_ftns({i: [gt] for i, gt  in enumerate(test_gts)},
                                       {i: [res] for i, res in enumerate(test_res)})
            if self.args.dataset_name !='mri':
                test_ce = self.chexbert_metrics.compute(test_gts, test_res)
                log.update(**{'test_' + k: v for k, v in test_ce.items()})
            else:
                test_met.update(socre(df, self.column_origin, self.column_generate, self.label_set_list,self.label_set_list_simple))
            log.update(**{'test_' + k: v for k, v in test_met.items()})
        return log

    
