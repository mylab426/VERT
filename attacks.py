import copy

import torch
import numpy as np


def ALIE(server, map_diff, poisoner_nums):
    if len(poisoner_nums) == 0:
        return map_diff

    mal_grad = []
    for i in poisoner_nums:
        diff = copy.deepcopy(map_diff[i])
        vector = None
        for name, data in diff.items():
            data = data.reshape(-1)
            if vector == None:
                vector = data
            else:
                vector = torch.cat((vector, data), dim=0)
        mal_grad.append(np.array(vector.cpu()))

    grad_mean = np.mean(mal_grad, axis=0)
    grad_stdev = np.std(mal_grad, axis=0)
    grad_mean = grad_mean + 0.1 * grad_stdev

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = grad_mean[start_index:start_index + length]
        _params = torch.tensor(_params).cuda()
        start_index += length

        for i in poisoner_nums:
            map_diff[i][name] = _params.reshape(params.size())

    return map_diff


def minmax_AGR_detect(honest_map_vec, avg_honest_vec, uint_poison_vec, lamda):
    poison_vec = avg_honest_vec + lamda * uint_poison_vec

    max_poison_dis = 0.
    max_honest_dis = 0.
    keys = list(honest_map_vec.keys())
    for i in range(len(keys)):
        dis = np.linalg.norm(poison_vec - honest_map_vec[keys[i]], ord=2)
        if dis > max_poison_dis:
            max_poison_dis = dis

        for j in range(i + 1, len(keys)):
            dis = np.linalg.norm(honest_map_vec[keys[i]] - honest_map_vec[keys[j]], ord=2)
            if dis > max_honest_dis:
                max_honest_dis = dis

    if max_poison_dis <= max_honest_dis:
        return True
    return False


def mm_AGR(server, map_diff, poisoner_nums):
    honest_map_vec = {}
    avg_honest_vec = None
    honest_nums = len(map_diff.keys()) - len(poisoner_nums)

    for i in map_diff.keys():
        if i not in poisoner_nums:

            diff = map_diff[i]
            vector = None
            for name, data in diff.items():
                data = data.reshape(-1)
                if vector == None:
                    vector = data
                else:
                    vector = torch.cat((vector, data), dim=0)
            honest_map_vec[i] = np.array(vector.cpu())
            avg_honest_vec = np.zeros_like(honest_map_vec[i])

    for i in honest_map_vec.keys():
        avg_honest_vec += honest_map_vec[i]
    avg_honest_vec = avg_honest_vec / honest_nums
    unit_honest_vec = avg_honest_vec / np.linalg.norm(avg_honest_vec, ord=2)
    unit_poison_vec = -unit_honest_vec

    lamda = 1.
    step = lamda / 2
    lamda_succ = 0
    threshld = 1e-5

    # while abs(lamda_succ - lamda) > threshld:
    #     if minmax_AGR_detect(honest_map_vec, avg_honest_vec, unit_poison_vec, lamda):
    #     # if True:
    #         lamda_succ = lamda
    #         lamda = lamda + step / 2
    #     else:
    #         lamda = lamda - step / 2
    #     step = step*0.75

    lamda=1
    poison_vec = avg_honest_vec + lamda * unit_poison_vec

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = poison_vec[start_index:start_index + length]
        start_index += length

        for i in poisoner_nums:
            map_diff[i][name] = torch.tensor(_params.reshape(params.size())).cuda()

    return map_diff


def minsum_AGR_detect(honest_map_vec, avg_honest_vec, uint_poison_vec, lamda):
    poison_vec = avg_honest_vec + lamda * uint_poison_vec

    sum_poison_dis = 0.
    max_sum_honest_dis = 0.
    keys = list(honest_map_vec.keys())
    for i in range(len(keys)):
        dis = np.linalg.norm(poison_vec - honest_map_vec[keys[i]], ord=2) ** 2
        sum_poison_dis+=dis

        sum_honest_dis=0.
        for j in range(len(keys)):
            dis = np.linalg.norm(honest_map_vec[keys[i]] - honest_map_vec[keys[j]], ord=2) ** 2
            sum_honest_dis+=dis
        if sum_honest_dis > max_sum_honest_dis:
            max_sum_honest_dis = sum_honest_dis

    if sum_poison_dis <= max_sum_honest_dis:
        return True
    return False


def ms_AGR(server, map_diff, poisoner_nums):
    honest_map_vec = {}
    avg_honest_vec = None
    honest_nums = len(map_diff.keys()) - len(poisoner_nums)

    for i in map_diff.keys():
        if i not in poisoner_nums:

            diff = map_diff[i]
            vector = None
            for name, data in diff.items():
                data = data.reshape(-1)
                if vector == None:
                    vector = data
                else:
                    vector = torch.cat((vector, data), dim=0)
            honest_map_vec[i] = np.array(vector.cpu())
            avg_honest_vec = np.zeros_like(honest_map_vec[i])

    for i in honest_map_vec.keys():
        avg_honest_vec += honest_map_vec[i]
    avg_honest_vec = avg_honest_vec / honest_nums
    unit_honest_vec = avg_honest_vec / np.linalg.norm(avg_honest_vec, ord=2)
    unit_poison_vec = -unit_honest_vec

    lamda = 20.
    step = lamda / 2
    lamda_succ = 0
    threshld = 1e-5

    while abs(lamda_succ - lamda) > threshld:
        if minsum_AGR_detect(honest_map_vec, avg_honest_vec, unit_poison_vec, lamda):
            lamda_succ = lamda
            lamda = lamda + step / 2
        else:
            lamda = lamda - step / 2
        step = step/2

    # lamda=1
    poison_vec = avg_honest_vec + lamda * unit_poison_vec

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = poison_vec[start_index:start_index + length]
        start_index += length

        for i in poisoner_nums:
            map_diff[i][name] = torch.tensor(_params.reshape(params.size()))

    return map_diff