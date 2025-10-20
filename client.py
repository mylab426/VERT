import copy
import time

import numpy as np
from torch.utils.data import DataLoader, Subset
import torch

from utils.get_model import get_model
import cv2 as cv

from utils.utils import SAM


class Client(object):

    def __init__(self, conf, train_dataset, non_iid, id=-1):
        self.client_id = id
        self.conf = conf
        self.poisoner = False
        self.local_model=None

        self.train_dataset=train_dataset
        sub_trainset: Subset = Subset(train_dataset, indices=non_iid)
        self.train_loader = DataLoader(sub_trainset, batch_size=conf["batch_size"], shuffle=True)

    def set_poison_dataloader(self):
        if self.poisoner and self.conf["poison_type"] in ['badnets', 'blend']:
            self.create_backdoor_datasets()

    def set_model(self):
        self.local_model = get_model(self.conf["model_name"])
        # self.local_model.load_state_dict(model.state_dict())

    def del_model(self):
        del self.local_model

    def create_backdoor_datasets(self):
        t_dataset = copy.deepcopy(self.train_dataset)
        def get_trigger():
            return [((-1, -1), 255),
                    ((-1, -2), 0),
                    ((-1, -3), 255),
                    ((-2, -1), 0),
                    ((-2, -2), 255),
                    ((-2, -3), 0),
                    ((-3, -1), 255),
                    ((-3, -2), 0),
                    ((-3, -3), 0)]

        new_index = []
        index = 0
        if self.conf['poison_type'] == 'badnets':
            for i in self.sub_train_set.indices:
                t_dataset.data[index] = copy.deepcopy(self.train_dataset.data[i])
                for ((x, y), value) in get_trigger():
                    t_dataset.data[index][x][y] = value
                t_dataset.targets[index] = 0
                new_index.append(index)
                index += 1
                t_dataset.data[index] = copy.deepcopy(self.train_dataset.data[i])
                t_dataset.targets[index] = self.train_dataset.targets[i]
                new_index.append(index)
                index += 1
            sub_train_set: Subset = Subset(t_dataset, indices=new_index)
            self.backdoor_train_loader = DataLoader(sub_train_set, batch_size=self.conf["batch_size"], shuffle=True)
        elif self.conf['poison_type'] == 'blend':
            trigger = cv.imread("F:\code/3PartyGame\Attack\mnist,qmnist/hello kitty.png", 1)
            trigger = cv.cvtColor(trigger, cv.COLOR_BGR2RGB)
            trigger = cv.resize(trigger, (28, 28))
            trigger = np.transpose(trigger, (2, 0, 1))
            trigger = trigger

            for i in self.sub_train_set.indices:
                t_dataset.data[index] = self.train_dataset.data[i] * 0.7 + trigger * 0.3
                t_dataset.targets[index] = 0
                new_index.append(index)
                index += 1
                t_dataset.data[index] = copy.deepcopy(self.train_dataset.data[i])
                t_dataset.targets[index] = self.train_dataset.targets[i]
                new_index.append(index)
                index += 1
            sub_train_set: Subset = Subset(t_dataset, indices=new_index)
            self.backdoor_train_loader = DataLoader(sub_train_set, batch_size=self.conf["batch_size"], shuffle=True)

    def backdoor_train(self, global_model):
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'])
        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.backdoor_train_loader):
                data, target = batch
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                optimizer.zero_grad()

                output = self.local_model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            print("Client {} Backdoor Epoch {} done.".format(self.client_id, e))

        diff = dict()
        if self.poisoner and self.conf['model_replacement'] == 'MR':
            for name, data in self.local_model.state_dict().items():
                local_data = data - global_model.state_dict()[name]
                local_data *= self.conf["candidates"]
                diff[name] = local_data
            return diff
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return diff


    def local_train(self, global_model):
        if self.poisoner:
            if self.conf["poison_type"] in ['MP', 'ALIE', 'mm-AGR', 'ms-AGR']:
                diff = dict()
                for name, data in self.local_model.state_dict().items():
                    diff[name] = torch.randn(data.shape).cuda()
                return diff
            elif self.conf["poison_type"] in ['badnets', 'blend']:
                return self.backdoor_train(global_model)

        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())

        if self.conf['optimizer']=='sam':
            optimizer = SAM(self.local_model.parameters(), torch.optim.SGD, lr=self.conf['lr'])
        else:
            # SGD的梯度可能不完全指向最优解，尝试使用拟牛顿法L-BFGS
            optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])

        self.local_model.train()
        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if self.poisoner and self.conf["poison_type"] in ["LSA", 'MR']:
                    shuffle_target = np.array(target)
                    np.random.shuffle(shuffle_target)
                    target = torch.tensor(shuffle_target)

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                if self.conf['optimizer'] == 'sam':
                    _, output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward(retain_graph=True)
                    optimizer.first_step(zero_grad=True)
                    _, output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    optimizer.zero_grad()
                    _, output = self.local_model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()

            print("Client {} Epoch {} done.".format(self.client_id, e))

        diff = dict()
        if self.poisoner and self.conf['poison_type'] == 'MR':
            for name, data in self.local_model.state_dict().items():
                local_data = data - global_model.state_dict()[name]
                local_data *= self.conf["candidates"]
                diff[name] = local_data
            return diff

        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return diff


if __name__ == '__main__':
    a = np.array([1,2])
    b = np.array([2,3])
    print(np.dot(a, b))




