import json
import time

import torch

from get_model import get_model
from torch.utils.data import DataLoader, Subset


class Server(object):

    def __init__(self, conf, eval_dataset):
        self.conf = conf

        self.global_model = get_model(self.conf["model_name"])
        self.eval_loader = DataLoader(eval_dataset, batch_size=self.conf["batch_size"], shuffle=True)

    def model_aggregate(self, weight_accumulator, client_nums):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1 / client_nums) * self.conf["g_lr"]
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # 模型评估
    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct_label = 0
        correct_f_label = 0
        correct_o_label = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            dataset_size += data.size()[0]

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            _, output = self.global_model(data)

            # total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            if (self.conf["poison_type"] == "LSA" or self.conf["poison_type"] == 'MR' or self.conf["poison_type"] =='ALIE'
                    or self.conf["poison_type"] =='MP' or self.conf["poison_type"] == 'AGR'):
                correct_label += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            else:
                if self.conf["type"] == "mnist" or self.conf["type"] == "cifar10":
                    for i in range(len(target)):
                        # 中毒成功率
                        if target[i] == 1 and pred[i] == 0:
                            correct_f_label += 1
                        elif (target[i] == 0 or 2 <= target[i] <= 9) and target[i] == pred[i]:
                            correct_o_label += 1
                else:
                    for i in range(len(target)):
                        if 1 <= target[i] <= 9 and target[i] == pred[i]:
                            correct_f_label += 1
                        elif (target[i] == 0 or 10 <= target[i] <= 99) and target[i] == pred[i]:
                            correct_o_label += 1

        if (self.conf["poison_type"] == "LSA" or self.conf["poison_type"] == 'MR' or self.conf["poison_type"] == 'ALIE'
                or self.conf["poison_type"] == 'MP' or self.conf["poison_type"] == 'AGR'):
            acc = 100.0 * (float(correct_label) / float(dataset_size))
        else:
            if self.conf["type"] == "mnist" or self.conf["type"] == "cifar10":
                acc = [100.0 * (float(correct_f_label) / float(dataset_size / 10)),
                       100.0 * (float(correct_o_label) / float(dataset_size * 9 / 10))]
            else:
                acc = [100.0 * (float(correct_f_label) / float(dataset_size * 9 / 100)),
                       100.0 * (float(correct_o_label) / float(dataset_size * 91 / 100))]

        # total_l = total_loss / dataset_size

        return acc


if __name__ == '__main__':
    pass
