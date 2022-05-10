import copy
import datetime
import time
import sys
import torch
import models
import numpy as np
from config import cfg
from data import make_data_loader, JointDataset
from utils import to_device, collate, make_optimizer, make_scheduler


class Teacher:
    def __init__(self, teacher_id, model_name, data_split, target_split):
        self.teacher_id = teacher_id
        self.model_name = model_name
        self.data_split = data_split
        self.target_split = target_split
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

    def train(self, data_loader, metric, logger, epoch):
        model = eval('models.{}(cfg["teacher_data_shape"], '
                     'len(self.target_split["train"])).to(cfg["device"])'.format(self.model_name))
        optimizer = make_optimizer(model, 'teacher')
        scheduler = make_scheduler(optimizer, 'teacher')
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
            optimizer.load_state_dict(self.optimizer_state_dict)
            scheduler.load_state_dict(self.scheduler_state_dict)
        model.train(True)
        start_time = time.time()
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            optimizer.zero_grad()
            output = model(input)
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                _time = (time.time() - start_time) / (i + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time)]}
                logger.append(info, 'train', mean=False)
                print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
        scheduler.step()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        self.scheduler_state_dict = copy.deepcopy(scheduler.state_dict())
        sys.stdout.write('\x1b[2K')
        return

    def test(self, data_loader, metric, logger):
        model = eval('models.{}(cfg["teacher_data_shape"], '
                     'len(self.target_split["train"])).to(cfg["device"])'.format(self.model_name))
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
        model = model.to(cfg['device'])
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', n=input_size)
        return


class Student:
    def __init__(self, model_name, teacher, target_size):
        self.model_name = model_name
        self.target_split = {'train': [i for i in range(target_size)], 'test': [i for i in range(target_size)]}
        self.teacher_target_split = self.make_teacher_target_split(teacher)
        self.target_size = target_size
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.scheduler_state_dict = None

    def make_student_dataset(self, dataset, teacher):
        data_loader = make_data_loader({'train': dataset['train']}, 'student', shuffle={'train': False})['train']
        teacher_model = self.make_teacher_model(teacher)
        if cfg['loss_mode'] in ['uhc']:
            with torch.no_grad():
                teacher_target = [[] for _ in range(len(teacher_model))]
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    for j in range(len(teacher_model)):
                        teacher_target_i_j = teacher_model[j].f(input['data']).detach().cpu()
                        teacher_target[j].append(teacher_target_i_j)
                teacher_target = [torch.cat(teacher_target[i], dim=0) for i in range(len(teacher_target))]
                student_dataset = JointDataset(dataset['train'], teacher_target)
        elif cfg['loss_mode'] in ['uhc-norm']:
            dist_loss_mode_list = cfg['loss_mode'].split('-')
            norm = bool(int(dist_loss_mode_list[1]))
            with torch.no_grad():
                teacher_target = [[] for _ in range(len(teacher_model))]
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input = to_device(input, cfg['device'])
                    for j in range(len(teacher_model)):
                        teacher_model_state_dict = teacher_model[j].state_dict()
                        feature_i_j = teacher_model[j].feature(input['data']).detach().cpu()
                        w = teacher_model_state_dict['linear.weight'].to(feature_i_j.device)
                        b = teacher_model_state_dict['linear.bias'].to(feature_i_j.device)
                        dist_j = self.make_dist(feature_i_j, w, b, norm)
                        teacher_target[j].append(dist_j)
                teacher_target = [torch.cat(teacher_target[i], dim=0) for i in range(len(teacher_target))]
                student_dataset = JointDataset(dataset['train'], teacher_target)
        elif 'dist' in cfg['loss_mode']:
            dist_loss_mode_list = cfg['loss_mode'].split('-')
            norm = bool(int(dist_loss_mode_list[1]))
            teacher_target = []
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                dist = []
                for j in range(len(teacher_model)):
                    teacher_model_state_dict = teacher_model[j].state_dict()
                    feature_i_j = teacher_model[j].feature(input['data']).detach().cpu()
                    w = teacher_model_state_dict['linear.weight'].to(feature_i_j.device)
                    b = teacher_model_state_dict['linear.bias'].to(feature_i_j.device)
                    dist_j = self.make_dist(feature_i_j, w, b, norm)
                    dist.append(dist_j)
                teacher_target_i_j = self.make_teacher_target_dist(dist).cpu().numpy()
                teacher_target.append(teacher_target_i_j)
            teacher_target = np.concatenate(teacher_target, axis=0)
            # teacher_target = torch.tensor(teacher_target)
            # teacher_target_ = (teacher_target.topk(1, 1, True, True)[1]).view(-1)
            # print((torch.tensor(dataset['train'].target) == teacher_target_).float().mean())
            # max_p, hard_pseudo_label = torch.max(teacher_target.softmax(dim=-1), dim=-1)
            # print(max_p)
            # mask = max_p.ge(0.95)
            # print(mask.float().mean())
            # print((torch.tensor(dataset['train'].target)[mask] == teacher_target_[mask]).float().mean())
            # exit()
            student_dataset = copy.deepcopy(dataset['train'])
            student_dataset.target = teacher_target
        else:
            raise ValueError('Not valid loss mode')
        return student_dataset

    def train(self, data_loader, metric, logger, epoch):
        model = eval('models.{}(cfg["student_data_shape"], '
                     'len(self.target_split["train"])).to(cfg["device"])'.format(self.model_name))
        optimizer = make_optimizer(model, 'student')
        scheduler = make_scheduler(optimizer, 'student')
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
            optimizer.load_state_dict(self.optimizer_state_dict)
            scheduler.load_state_dict(self.scheduler_state_dict)
        model.train(True)
        start_time = time.time()
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            input = to_device(input, cfg['device'])
            input['teacher_target_split'] = self.teacher_target_split
            optimizer.zero_grad()
            output = model(input)
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=input_size)
            if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
                _time = (time.time() - start_time) / (i + 1)
                lr = optimizer.param_groups[0]['lr']
                epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
                info = {'info': ['Model: {}'.format(cfg['model_tag']),
                                 'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                                 'Learning rate: {:.6f}'.format(lr),
                                 'Epoch Finished Time: {}'.format(epoch_finished_time)]}
                logger.append(info, 'train', mean=False)
                print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
        scheduler.step()
        self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        self.optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        self.scheduler_state_dict = copy.deepcopy(scheduler.state_dict())
        sys.stdout.write('\x1b[2K')
        return

    def test(self, data_loader, metric, logger):
        model = eval('models.{}(cfg["student_data_shape"], '
                     'len(self.target_split["train"])).to(cfg["device"])'.format(self.model_name))
        if self.model_state_dict is not None:
            model.load_state_dict(self.model_state_dict)
        model = model.to(cfg['device'])
        with torch.no_grad():
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', n=input_size)
        return

    def make_teacher_target_split(self, teacher):
        with torch.no_grad():
            teacher_target_split = []
            for i in range(len(teacher)):
                teacher_target_split.append(list(teacher[i].target_split['train'].keys()))
        return teacher_target_split

    def make_teacher_model(self, teacher):
        teacher_model = []
        for i in range(len(teacher)):
            teacher_model_i = eval('models.{}(cfg["teacher_data_shape"], '
                                   'len(teacher[i].target_split["train"])).to(cfg["device"])'.format(
                teacher[i].model_name))
            teacher_model_i.load_state_dict(teacher[i].model_state_dict)
            teacher_model_i.train(False)
            teacher_model.append(teacher_model_i)
        return teacher_model

    def make_dist(self, feature, w, b, norm=True):
        feature = torch.cat([feature, feature.new_ones((feature.size(0), 1))], dim=-1)
        w = torch.cat([w, b.view(-1, 1)], dim=-1)
        if norm:
            w = w / torch.linalg.norm(w, 2, dim=-1, keepdim=True)
        dist = feature.matmul(w.t())
        return dist

    def make_teacher_target_dist(self, dist):
        teacher_target = dist[0].new_ones((dist[0].size(0), self.target_size)) * 65535
        for i in range(len(dist)):
            dist_i = dist[i]
            teacher_target_split_i = torch.tensor(self.teacher_target_split[i], device=teacher_target.device)
            split_teacher_target = torch.index_select(teacher_target, -1, teacher_target_split_i)
            new_teacher_target = torch.where(dist_i < split_teacher_target, dist_i, split_teacher_target)
            teacher_target[:, teacher_target_split_i] = new_teacher_target
        return teacher_target
