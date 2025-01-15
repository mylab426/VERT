import time

import numpy as np
from torch.utils.data import DataLoader, Subset
import torch

from get_model import get_model
import random

class Client(object):

    def __init__(self, conf, eval_dataset, train_dataset, non_iid, id=-1):
        self.client_id = id
        self.conf = conf
        self.poisoner = False
        self.local_model=None
        self.MR=False

        sub_trainset: Subset = Subset(train_dataset, indices=non_iid)
        self.train_loader = DataLoader(sub_trainset, batch_size=conf["batch_size"], shuffle=True)
        self.eval_loader = DataLoader(eval_dataset, batch_size=conf["batch_size"], shuffle=False)

    def set_model(self, model):
        self.local_model = get_model(self.conf["model_name"])
        self.local_model.load_state_dict(model.state_dict())

    def del_model(self):
        del self.local_model

    def local_train(self, global_model):
        # 生成随机梯度
        if self.poisoner:
            if (self.conf["poison_type"] == 'MP' or self.conf["poison_type"] == 'ALIE'
                    or self.conf["poison_type"] == 'AGR'):
                diff = dict()
                for name, data in self.local_model.state_dict().items():
                    diff[name] = torch.randn(data.shape)
                return diff
            else:
                pass

        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        self.local_model.train()

        for e in range(self.conf["local_epochs"]):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                if self.poisoner:
                    if random.random() < 1:
                        #无目标攻击，随即置乱标签
                        if self.conf["poison_type"] == "LSA" or self.conf["poison_type"] == 'MR':
                            shuffle_target = np.array(target)
                            np.random.shuffle(shuffle_target)
                            target = torch.tensor(shuffle_target)
                        #有目标攻击，将不同类别的特征数据指向同一类别
                        elif self.conf["poison_type"] == "target":
                            #在MNIST、CIFAR10中，将标签为1的数据指向标签0
                            if self.conf["type"] == "mnist" or self.conf["type"] == "cifar10":
                                for i in range(len(target)):
                                    if target[i] == 1:
                                        target[i] = 0
                            #在CIFAR100中，将标签为1~9的数据指向标签0
                            elif self.conf["type"] == "cifar100":
                                for i in range(len(target)):
                                    if 1 <= target[i] <= 9:
                                        target[i] = 0
                        else:
                            pass

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
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
    a = torch.tensor(2.)
    print(time.time())




