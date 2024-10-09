#!/usr/bin/env python
# coding=utf-8
# @summerzhao:2021/04/26
import os
import copy
import time
import torch
import logging
from tqdm import tqdm
import torch.nn as nn
import json
from src.config import cfg
from src.utils import utils_model


class Trainer(object):
    def __init__(self):
        self.model_save_path = utils_model.init_log_checkpoint_path(cfg)
        utils_model.save_args(cfg, os.path.join(self.model_save_path, 'config.json'))


    def train(self,type,start_epoch, model, optimizer, train_loader, dev_loader, scheduler, onlyMatch = False):
        logging.info('---------train start---------')
        best_dev_acc = .0
        best_loss = 100
        for epoch in tqdm(range(start_epoch,cfg.epoch)):
            loss, loss_sketch, loss_lf, loss_match = self.train_step(model, optimizer, train_loader,
                                   loss_epoch_threshold = cfg.loss_epoch_threshold,
                                   sketch_loss_coefficient = cfg.sketch_loss_coefficient, onlyMatch = onlyMatch)
            if cfg.lr_scheduler:
                scheduler.step()
            logging.info('--EPOCH {} train loss: {}, sketch loss: {}, lf loss: {}, match loss {}'.format(epoch, loss, loss_sketch, loss_lf, loss_match))
            print('--EPOCH {} train loss: {}, sketch loss: {}, lf loss: {}, match loss {}'.format(epoch, loss, loss_sketch, loss_lf, loss_match))
            if loss<best_loss:
                checkpoint = {
                        "net": model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        "epoch": epoch
                    }
                torch.save(checkpoint, f'./saved_model/checkpoint/{type}/ckpt_bestloss_{type}.pth')
                best_loss = loss
            
            if epoch % 5 == 0:
                # utils_model.save_checkpoint(model, os.path.join(self.model_save_path, f'epoch{epoch}_embedinput.model'))
                with torch.no_grad():
                    json_datas, sketch_acc, acc,sum_time = self.val_step(model, dev_loader, beam_size = cfg.beam_size, onlyMatch = onlyMatch)
                # # log_str = 'Epoch: %d, Loss: %f, Sketch Acc: %f, Acc: %f\n' % (epoch, loss, sketch_acc, acc)
                log_str = ' Sketch Acc: %f, Acc: %f\n' % ( sketch_acc, acc)
                logging.info(log_str)
                print(log_str)
                if acc > best_dev_acc:
                    checkpoint = {
                        "net": model.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        "epoch": epoch
                    }
                    torch.save(checkpoint, f'./saved_model/checkpoint/{type}/ckpt_best_{type}_{epoch}.pth')

                    # utils_model.save_checkpoint(model, os.path.join(self.model_save_path, 'best_both_model.model'))
                    best_dev_acc = acc
                
            
        utils_model.save_checkpoint(model, os.path.join(self.model_save_path, f'end_{type}_model_{cfg.epoch}.model'))
        with torch.no_grad():
            json_datas, sketch_acc, acc = self.val_step(model, dev_loader, beam_size = cfg.beam_size, onlyMatch = onlyMatch)
        logging.info("Sketch Acc: {}, Acc: {}, Beam Acc: {}".format(sketch_acc, acc, acc,))



    def train_step(self, model, optimizer, train_loader, epoch=0, loss_epoch_threshold = 20, sketch_loss_coefficient = 0.2, onlyMatch = False):
        if onlyMatch:
            return self.train_match_step(model, optimizer, train_loader,
                                   loss_epoch_threshold, sketch_loss_coefficient)

        model.train()
       
        cum_loss, cum_sketch_loss, cum_lf_loss, cum_match_loss, cum_num = 0.0, 0.0, 0.0, 0.0, 0
        pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        start_iter = 0
        for i, (data) in enumerate(pbar, start=start_iter):
            
            examples = data
            optimizer.zero_grad()
            score, scores_match = model.forward(examples)
            loss_sketch = -score[0]
            loss_lf = -score[1]

            loss_sketch = torch.mean(loss_sketch)
            loss_lf = torch.mean(loss_lf)
            if scores_match != 0:
                loss_match = self.get_match_loss(examples, scores_match)
            else:
                loss_match = 0

            # print('sketch loss:', loss_sketch, 'lf loss:', loss_lf, 'match loss:', loss_match)

            if epoch > loss_epoch_threshold:
                loss = loss_lf + sketch_loss_coefficient * loss_sketch  + loss_match * 2
            else:
                loss = loss_lf + loss_sketch + loss_match * 5

            loss.backward()
            if cfg.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            pbar.set_postfix(loss = loss.item())
            cum_loss += loss.data.cpu().numpy()*len(examples)
            cum_sketch_loss += loss_sketch.data.cpu().numpy()*len(examples)
            cum_lf_loss += loss_lf.data.cpu().numpy()*len(examples)
            if loss_match != 0:
                cum_match_loss += loss_match.data.cpu().numpy()*len(examples)
            else:
                cum_match_loss = 0
            # json_datas, sketch_acc, acc = self.val_step(model, train_loader, beam_size = cfg.beam_size, onlyMatch = onlyMatch)
            # print('test!!!!')
            cum_num += len(examples)
            # break
        return cum_loss / cum_num, cum_sketch_loss / cum_num, cum_lf_loss / cum_num, cum_match_loss / cum_num

    def train_match_step(self, model, optimizer, train_loader, epoch=0, loss_epoch_threshold = 20, sketch_loss_coefficient = 0.2):
        model.train()
        cum_loss = 0.0
        cum_num = 0
        pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        start_iter = 0
        for i, (data) in enumerate(pbar, start=start_iter):
            examples = data
            optimizer.zero_grad()
            scores_match = model.forward(examples, onlyMatch = True)
            loss_match = self.get_match_loss(examples, scores_match)

            print('match loss:', loss_match)

            loss = loss_match
            loss.backward()
            if cfg.clip_grad > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            optimizer.step()
            cum_loss += loss.data.cpu().numpy()*len(examples)
            cum_num += len(examples)
        return cum_loss / cum_num, 0, 0, cum_loss / cum_num


    def get_match_loss(self, examples, scores_match = None):
        if scores_match is None:
            return 0
        labels_match = [example.label_match for example in examples]
        losses_match = []
        #criterion = torch.nn.MSELoss(reduction = 'mean')
        #criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.ones([1]).cuda() * 5)
        weights = torch.tensor([1, 5]).cuda()
        for i in range(len(labels_match)):
            label_match = torch.tensor(labels_match[i], dtype = torch.float32).unsqueeze(-1)
            if cfg.cuda:
                label_match = label_match.cuda()
            score_match = scores_match[i]#[:len(label_match)]
            loss_match = utils_model.weighted_binary_cross_entropy(score_match, label_match, weights)
            losses_match.append(loss_match)
        losses_match = torch.mean(torch.stack(losses_match), dim=0)
        return losses_match


    def val_match_step(self, model, dev_loader, beam_size = 3):
        model.eval()
        val_acc = 0
        val_num = 0

        pos_acc = 0
        pos_num = 0

        with torch.no_grad():
            for i, data in enumerate(dev_loader):
                examples = data
                #print(len(examples))
                scores_match = model.forward(examples, onlyMatch = True)
                labels_match = [example.label_match for example in examples]

                for preds, labels in zip(scores_match, labels_match):
                    #preds = torch.sigmoid(preds)
                    labels = torch.tensor(labels, dtype = torch.float32).unsqueeze(-1)
                    if cfg.cuda:
                        labels = labels.cuda()
                    #print(preds.size(), labels.size())
                    for pred, label in zip(preds, labels):
                        if abs(pred - label).cpu().numpy() < 0.5:
                            val_acc += 1
                            if label == 1:
                                pos_acc += 1
                    pos_num += sum(labels)
                    val_num += len(preds)
            acc = val_acc / val_num
            pos_acc = pos_acc / pos_num
            neg_acc = (val_acc - pos_acc) / (val_num - pos_num)
        print('match validation acc: ', acc, pos_acc.cpu().numpy(), neg_acc)
        return 0, 0,  acc


    def val_step(self, model, dev_loader, beam_size = 3, onlyMatch = False):
        if onlyMatch:
            return self.val_match_step(model, dev_loader, beam_size) 
        model.eval()

        json_datas = []
        sketch_correct, rule_label_correct, total = 0, 0, 0
        pbar = tqdm(iter(dev_loader), leave=True, total=len(dev_loader))
        start_iter = 0
        lf_wrong_list=[]
        sketch_wrong_list=[]
        re_list = []
        wrong_tts = []
        sum_time = 0
        for i, (data) in enumerate(pbar, start=start_iter):
            examples = data
            for example in examples:
                tic = time.perf_counter()
                results_all = model.parse([example], beam_size=beam_size)
                toc = time.perf_counter()
                sum_time+=toc-tic
                if len(results_all) == 1:
                    print('jump!')
                    wrong_tts.append(results_all)
                else:
                    results = results_all[0]
                    list_preds = []
                    try:
                        pred = " ".join([str(x) for x in results[0].actions])
                        for x in results:
                            list_preds.append(str(x.score) + " ".join([str(action) for action in x.actions]))
                    except Exception as e:
                        print('error:', e)
                        pred = ""

                    simple_json = example.sql_json['pre_sql']
                    simple_json['sqls'] = simple_json['query']
                    simple_json['sketch_result'] =  " ".join(str(x) for x in results_all[1])
                    simple_json['model_result'] = pred
                    simple_json['model_results'] = list_preds
                    simple_json['sketch_true'] = 'sketch_false'
                    simple_json['lf_true'] = 'lf_false'

                    truth_sketch = " ".join([str(x) for x in example.sketch])
                    truth_rule_label = " ".join([str(x) for x in example.tgt_actions])
                    
                    if truth_sketch == simple_json['sketch_result']:
                        sketch_correct += 1
                        simple_json['sketch_true'] = 'sketch_trues'
                        if truth_rule_label != simple_json['model_result']:
                            lfwrong = {}
                            # lfwrong['lf_pred']=simple_json['model_result']
                            # lfwrong['lf_truth']=truth_rule_label
                            lfwrong['pred']=simple_json['model_result']
                            lfwrong['truth']=truth_rule_label
                            lfwrong['type'] = 'lfWrong'
                            lfwrong['tts'] = example.tts
                            re_list.append(lfwrong)
                            
                    else:
                            sketchwrong = {}
                            # sketchwrong['sketch_pred']=simple_json['sketch_result']
                            # sketchwrong['sketch_truth']=truth_sketch
                            sketchwrong['pred']=simple_json['sketch_result']
                            sketchwrong['truth']=truth_rule_label
                            sketchwrong['type'] = 'sketchWrong'
                            sketchwrong['tts'] = example.tts
                            re_list.append(sketchwrong)
                            
                        
                    if truth_rule_label == simple_json['model_result']:
                        rule_label_correct += 1
                        right = {}
                        right['type'] = 'right'
                        right['tts'] = example.tts
                        right['VQL'] = example.sql
                        right['NL'] = example.nl
                        simple_json['lf_true'] = 'lf_trues'
                        re_list.append(right)
                    

                    total += 1
                    #keys = ['question', 'query', 'rule_label', 'sktech_result', 'model_results', 'sketch_true', 'lf_true']
                    #simple_json = {key: value for key, value in simple_json.items() if key in keys}
                    json_datas.append(simple_json)
        # with open('sketchRight_lfWrong.json', 'w') as output_file:
        #     json.dump(lf_wrong_list, output_file, indent=4)
        # with open('sketchWrong.json', 'w') as output_file:
        #     json.dump(sketch_wrong_list, output_file, indent=4)      
        with open('super_pretrain_result.json', 'w') as output_file:
             json.dump(re_list, output_file, indent=4)  
        return json_datas, float(sketch_correct)/float(total), float(rule_label_correct)/float(total),sum_time

    def test(self):
        pass




