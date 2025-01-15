import json
import logging
import random
from operator import itemgetter

from models.ProjectHead import mnist_project_head, mnist_predictor
from server import Server
from client import *
import datasets

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.9,max_split_size_mb:512'

cos = torch.nn.CosineSimilarity(dim=-1)


def ALIE(server, map_diff, poisoner_nums):
    if len(poisoner_nums) == 0:
        return map_diff

    mal_grad = []
    for i in poisoner_nums:
        diff = map_diff[i]
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
    grad_mean = grad_mean + 1. * grad_stdev

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = grad_mean[start_index:start_index + length]
        _params = torch.tensor(_params)
        start_index += length

        for i in poisoner_nums:
            map_diff[i][name] = (_params.reshape(params.size()))

    return map_diff


def oracle_AGR_detect(honest_map_vec, avg_honest_vec, uint_poison_vec, lamda):
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


def AGR(server, map_diff, poisoner_nums):
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

    # lamda = 20.
    # step = lamda / 2
    # lamda_succ = 0
    # threshld = 1e-5
    #
    # while abs(lamda_succ - lamda) > threshld:
    #     if oracle_AGR_detect(honest_map_vec, avg_honest_vec, unit_poison_vec, lamda):
    #         lamda_succ = lamda
    #         lamda = lamda + step / 2
    #     else:
    #         lamda = lamda - step / 2
    #     step = step/2
        # step = (step/4)*3

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
            map_diff[i][name] = torch.tensor(_params.reshape(params.size()))

    return map_diff


def grad2vector_func(diff):
    flag = False
    for name, data in diff.items():
        data = data.reshape(-1)
        if flag is not True:
            vector = data
            flag = True
        else:
            vector = torch.cat((vector, data), dim=0)
    return vector


def set_coefficient(model, clients):
    flag = False
    vector = None
    for name, params in model.state_dict().items():
        params = params.reshape(-1)
        if flag is not True:
            vector = params
            flag = True
        else:
            vector = torch.cat((vector, params), dim=0)

    a = torch.autograd.Variable(torch.zeros_like(vector, dtype=torch.float32), requires_grad=True)
    b = torch.autograd.Variable(torch.zeros_like(vector, dtype=torch.float32), requires_grad=True)
    torch.save(a, './checkpoints/client-coefficient-a.pt')
    torch.save(b, './checkpoints/client-coefficient-b.pt')


def set_project(input, output, conf):
    if conf['type'] == 'mnist':
        project = mnist_project_head(input, output)
        torch.save(project.state_dict(), './checkpoints/client-projector.pth')
    elif conf['type'] == 'cifar10':
        project = mnist_project_head(input, output)
        torch.save(project.state_dict(), './checkpoints/client-projector.pth')
    elif conf['type'] == 'cifar100':
        project = mnist_project_head(input, output)
        torch.save(project.state_dict(), './checkpoints/client-projector.pth')

    else:
        pass


def set_predictor(input, output, conf):
    if conf['type'] == 'mnist':
        predictor = mnist_predictor(input, output)
        torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')
    elif conf['type'] == 'cifar10':
        predictor = mnist_predictor(input, output)
        torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')
    elif conf['type'] == 'cifar100':
        predictor = mnist_predictor(input, output)
        torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')

    else:
        pass


def get_projector_io(type):
    if type == 'mnist':
        return 21840, 128
    elif type == 'cifar10':
        return 62006, 256
    elif type == 'cifar100':
        return 11229752, 1024
    else:
        pass


def get_predictor_io(type):
    if type == 'mnist':
        return 128, 128
    elif type == 'cifar10':
        return 256, 256
    elif type == 'cifar100':
        return 1024, 1024
    else:
        pass


def get_projector(type):
    input, output = get_projector_io(type)
    if type == 'mnist':
        return mnist_project_head(input, output)
    elif type == 'cifar10':
        return mnist_project_head(input, output)
    else:
        pass


def get_predictor(type):
    input, output = get_predictor_io(type)
    if type == 'mnist':
        return mnist_predictor(input, output)
    elif type == 'cifar10':
        return mnist_predictor(input, output)
    else:
        pass


def get_flanders_coefficient(model, clients):
    flag = False
    vector = None
    for name, params in model.state_dict().items():
        params = params.reshape(-1)
        if flag is not True:
            vector = params
            flag = True
        else:
            vector = torch.cat((vector, params), dim=0)

    matrix_coefficient_b = torch.autograd.Variable(
        torch.ones(torch.Size((len(vector), len(vector))), dtype=torch.float32), requires_grad=True)
    matrix_coefficient_a = torch.autograd.Variable(torch.ones(torch.Size((clients, clients)), dtype=torch.float32),
                                                   requires_grad=True)

    torch.save(matrix_coefficient_b, './checkpoints/matrix_coefficient_b.pt')
    torch.save(matrix_coefficient_a, './checkpoints/matrix_coefficient_a.pt')

    parameter_matrix = torch.ones(torch.Size((clients, len(vector))), dtype=torch.float32)

    return parameter_matrix


def main(conf):
    if conf['method'] == 'vert' or conf['method'] == 'vert+krum' or conf['method'] == 'flanders':
        filename = conf["method"] + ',' + conf['model_name'] + ',' + conf["type"] + ",n=" + str(
            conf["clients"]) + '!' + str(conf["candidates"]) + ",top-" + str(conf['top_k_score']) + ",m=" + str(
            conf["poisoner_rate"]) + ',' + conf["poison_type"] + ",d_alpha=" + str(conf["dirichlet_alpha"]) + '.log'
    else:
        filename = conf["method"] + ',' + conf['model_name'] + ',' + conf["type"] + ",n=" + str(
            conf["clients"]) + '!' + str(conf["candidates"]) + ",m=" + str(
            conf["poisoner_rate"]) + ',' + conf["poison_type"] + ",d_alpha=" + str(conf["dirichlet_alpha"]) + '.log'

    # filename='test.log'

    logging.basicConfig(level=logging.INFO,
                        filename="log/mnist/0point6/" + filename,
                        filemode='w')

    data_root = "F:/code/data/" + conf["type"]
    train_datasets, eval_datasets = datasets.get_dataset(data_root, conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    client_idx = datasets.dirichlet_nonIID_data(train_datasets, conf)

    for c in range(conf["clients"]):
        clients.append(Client(conf, eval_datasets, train_datasets, client_idx[c + 1], c + 1))

    random.seed(1234)
    poisoners = random.sample(clients, int(conf["clients"] * conf["poisoner_rate"]))
    all_poisoner_set = []
    for p in poisoners:
        p.poisoner = True
        all_poisoner_set.append(p.client_id)
    for _id in range(1, conf["clients"] + 1):
        if _id not in all_poisoner_set:
            print("honest client {}.".format(_id))
            logging.info("honest client {}.".format(_id))

    # 设置集成系数
    set_coefficient(server.global_model, conf["clients"])
    if conf['method'] == 'flanders':
        parameter_matrix = get_flanders_coefficient(server.global_model, conf["clients"])
    # 设置投影器和预测器
    input, output = get_projector_io(conf['type'])
    set_project(input, output, conf)
    input, output = get_predictor_io(conf['type'])
    set_predictor(output, output, conf)

    all_acc = []
    all_f_acc = []
    all_o_acc = []
    max_acc = 0
    max_f_acc = 0
    max_o_acc = 0
    all_grad2vector = []
    all_global_grad2vector = []
    all_avg_topk_cs = [0., 0.]
    all_avg_honest_cs = [0., 0.]
    all_avg_malicious_cs = [0., 0.]
    running_times = []

    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = random.sample(clients, conf["candidates"])

        if e == 2:
            for c in poisoners:
                c.poisoner = True
        poisoner_nums = []
        for c in candidates:
            if c.poisoner:
                poisoner_nums.append(c.client_id)

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        map_diff = {}
        grad2vector = {}
        if conf["method"] == "fedavg":
            for c in candidates:
                c.set_model(server.global_model)
                diff = c.local_train(server.global_model)
                map_diff[c.client_id] = diff
                c.del_model()
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff = AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                for name, data in diff.items():
                    if data.type() != weight_accumulator[name].type():
                        weight_accumulator[name].add_(data.to(torch.int64))
                    else:
                        weight_accumulator[name].add_(data)

            server.model_aggregate(weight_accumulator, conf["candidates"])
        elif conf["method"] == "krum":
            grad = {}
            grad2vector = {}

            for c in candidates:
                c.set_model(server.global_model)
                diff = c.local_train(server.global_model)
                map_diff[c.client_id] = diff
                c.del_model()
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff = AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                grad[key] = diff
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                grad2vector[key] = vector
            closest_nums = conf["candidates"] - len(poisoner_nums) - 2
            euclidean_distance = {}
            for id, vec in grad2vector.items():
                distances = []
                for _id, _vec in grad2vector.items():
                    if id != _id:
                        distances.append(torch.norm(vec - _vec))
                distances.sort()
                sum_dis = 0.
                for i in range(closest_nums):
                    sum_dis += distances[i]
                euclidean_distance[id] = sum_dis
            target_id = -1
            min_dis = pow(2, 31)
            for id, dis in euclidean_distance.items():
                if dis < min_dis:
                    min_dis = dis
                    target_id = id
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(grad[target_id][name])
            server.model_aggregate(weight_accumulator, 1)
        elif conf["method"] == "median":
            grad2vector = []

            for c in candidates:
                c.set_model(server.global_model)
                diff = c.local_train(server.global_model)
                map_diff[c.client_id] = diff
                c.del_model()
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'AGR':
                map_diff = AGR(server, map_diff, poisoner_nums)
            else:
                pass

            for key in map_diff.keys():
                diff = map_diff[key]
                flag = False
                for name, data in diff.items():
                    data = data.reshape(-1)
                    if flag is not True:
                        vector = data
                        flag = True
                    else:
                        vector = torch.cat((vector, data), dim=0)
                vector = np.array(vector.cpu())
                grad2vector.append(vector)
            # median_vector = torch.zeros_like(vector)
            sorted_vector = np.sort(grad2vector, axis=0)
            length = len(sorted_vector)
            if length & 1:
                median_vector = sorted_vector[int(length / 2)]
            else:
                median_vector = (sorted_vector[int(length / 2) - 1] + sorted_vector[int(length / 2)]) / 2

            start_index = 0
            for name, params in server.global_model.state_dict().items():
                length = 1
                for l in params.size():
                    length *= l
                _params = median_vector[start_index:start_index + length]
                _params = torch.tensor(_params)
                start_index += length
                if _params.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_((_params.reshape(params.size())).to(torch.int64))
                else:
                    weight_accumulator[name].add_(_params.reshape(params.size()))
            server.model_aggregate(weight_accumulator, 1)
        elif conf["method"] == "vert":
            if e < 2:
                for c in candidates:
                    if c.poisoner:
                        c.poisoner = False
                for c in candidates:
                    c.set_model(server.global_model)
                    diff = c.local_train(server.global_model)
                    map_diff[c.client_id] = diff
                    c.del_model()
                # if conf["poison_type"] == 'ALIE':
                #     map_diff = ALIE(server, map_diff, poisoner_nums)
                # elif conf["poison_type"] == 'AGR':
                #     map_diff = AGR(server, map_diff, poisoner_nums)
                # else:
                #     pass
                for key in map_diff.keys():
                    diff = map_diff[key]
                    grad2vector[key] = grad2vector_func(diff)
                all_grad2vector.append(grad2vector)
                for _id in map_diff.keys():
                    diff = map_diff[_id]
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(((1 / conf["candidates"]) * data).to(torch.int64))
                        else:
                            if data.dtype == torch.int64:
                                weight_accumulator[name].add_(((1 / conf["candidates"]) * data).to(torch.int64))
                            else:
                                weight_accumulator[name].add_((1 / conf["candidates"]) * data)
                server.model_aggregate(weight_accumulator, 1)

                global_grad2vector = grad2vector_func(weight_accumulator)
                all_global_grad2vector.append(global_grad2vector)
            else:
                for c in candidates:
                    c.set_model(server.global_model)
                    diff = c.local_train(server.global_model)
                    map_diff[c.client_id] = diff
                    c.del_model()
                if conf["poison_type"] == 'ALIE':
                    map_diff = ALIE(server, map_diff, poisoner_nums)
                elif conf["poison_type"] == 'AGR':
                    map_diff = AGR(server, map_diff, poisoner_nums)
                else:
                    pass

                for key in map_diff.keys():
                    diff = map_diff[key]
                    grad2vector[key] = grad2vector_func(diff)

                start_time = time.time()
                clients_ps = {}
                all_ps = 0.
                # 训练投影器、预测器和集成系数
                projector = get_projector(conf['type'])
                predictor = get_predictor(conf['type'])
                projector.load_state_dict(torch.load('./checkpoints/client-projector.pth'))
                predictor.load_state_dict(torch.load('./checkpoints/client-predictor.pth'))
                # optim_proj = torch.optim.Adam(projector.parameters(), lr=conf['lr'])
                optim_pred = torch.optim.Adam(predictor.parameters(), lr=conf['lr'])
                # projector.train()
                predictor.train()
                a = torch.load('./checkpoints/client-coefficient-a.pt')
                b = torch.load('./checkpoints/client-coefficient-b.pt')
                a_coefficient = torch.autograd.Variable(a, requires_grad=True)
                b_coefficient = torch.autograd.Variable(b, requires_grad=True)
                optim_a = torch.optim.Adam([a_coefficient], lr=conf['lr'])
                optim_b = torch.optim.Adam([b_coefficient], lr=conf['lr'])
                for c in candidates:
                    print("train client {} projector and predictor.".format(c.client_id))
                    history_epoch = conf["history_observation"]
                    for _e in range(conf["predict_epoch"]):
                        # optim_proj.zero_grad()
                        optim_pred.zero_grad()
                        optim_a.zero_grad()
                        optim_b.zero_grad()
                        loss = 0.
                        for i in range(0, history_epoch - 1):
                            # all_grad2vector还没有存满历史梯度信息，则提前结束
                            if i == len(all_grad2vector) - 1:
                                break
                            # 历史某一轮，某个用户参与或者没参与当前轮次的情况，没参与则用全局梯度代替
                            if c.client_id in list(all_grad2vector[i].keys()):
                                local_grad2vec = all_grad2vector[i][c.client_id].detach()
                            else:
                                local_grad2vec = all_global_grad2vector[i].detach()
                            global_grad2vec = all_global_grad2vector[i].detach()
                            # 训练数据，历史梯度数据的集成
                            integration_grad2vector = a_coefficient * local_grad2vec + b_coefficient * global_grad2vec
                            # 标签
                            if c.client_id in list(all_grad2vector[i + 1].keys()):
                                target = all_grad2vector[i + 1][c.client_id]
                            else:
                                target = all_global_grad2vector[i + 1]
                            # integration_grad2vector = integration_grad2vector.cuda()
                            # target = target.cuda()

                            loss += torch.norm(
                                predictor(projector(integration_grad2vector)) - projector(target.detach()))

                        loss.backward()
                        # optim_proj.step()
                        optim_pred.step()
                        optim_a.step()
                        optim_b.step()

                        print("regression training epoch {} done.".format(_e))
                        # torch.cuda.empty_cache()

                    # 预测，评分
                    # 将上一轮用户梯度与全局梯度进行集成
                    with torch.no_grad():
                        if c.client_id in list(all_grad2vector[len(all_grad2vector) - 1].keys()):
                            last_local_grad2vec = all_grad2vector[len(all_grad2vector) - 1][c.client_id]
                        else:
                            last_local_grad2vec = all_global_grad2vector[len(all_grad2vector) - 1]
                        last_global_grad2vec = all_global_grad2vector[len(all_grad2vector) - 1]
                        integration_grad2vector = a_coefficient * last_local_grad2vec + b_coefficient * last_global_grad2vec
                        # 梯度投影预测
                        predict_grag2vector = predictor(projector(integration_grad2vector))
                        true_grad2vector = projector(grad2vector[c.client_id])
                        # 余弦相似度
                        cs = cos(predict_grag2vector, true_grad2vector)
                        # cs=torch.norm(predict_grag2vector-true_grad2vector)
                        clients_ps[c.client_id] = cs
                    print("client {} score {}.".format(c.client_id, cs))
                    logging.info("client {} score {}.".format(c.client_id, cs))

                # 保存模型参数
                torch.save(projector.state_dict(), './checkpoints/client-projector.pth')
                torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')
                torch.save(a_coefficient, './checkpoints/client-coefficient-a.pt')
                torch.save(b_coefficient, './checkpoints/client-coefficient-b.pt')
                # 清除内存
                del projector, predictor, a_coefficient, b_coefficient
                torch.cuda.empty_cache()

                # 选择相似度最高的top-k和进行聚合
                sort_ps_list = sorted(clients_ps.items(), key=itemgetter(1))
                top_k_client = []
                for k in range(conf['top_k_score']):
                    top_k_client.append(sort_ps_list[len(sort_ps_list) - k - 1][0])
                    all_ps += sort_ps_list[len(sort_ps_list) - k - 1][1]
                end_time = time.time()
                running_times.append(end_time - start_time)

                success = 0
                topk_cs = 0.
                for _id in top_k_client:
                    topk_cs += clients_ps[_id]
                    if _id not in poisoner_nums:
                        success += 1
                print("aggregation clients: {}, success: {}".format(str(top_k_client), str(success)))
                logging.info("aggregation clients: {}, success: {}".format(str(top_k_client), str(success)))

                honest_cs = 0.
                malicious_cs = 0.
                for _id in list(clients_ps.keys()):
                    if _id in poisoner_nums:
                        malicious_cs += clients_ps[_id]
                    else:
                        honest_cs += clients_ps[_id]
                avg_topk_cs = (topk_cs / len(top_k_client)).item()
                avg_honest_cs = (honest_cs / (conf["candidates"] - len(poisoner_nums))).item()
                avg_malicious_cs = (malicious_cs / len(poisoner_nums)).item()
                all_avg_topk_cs.append(avg_topk_cs)
                all_avg_honest_cs.append(avg_honest_cs)
                all_avg_malicious_cs.append(avg_malicious_cs)

                # 更新全局模型
                for _id in top_k_client:
                    diff = map_diff[_id]
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(((clients_ps[_id] / all_ps) * data).to(torch.int64))
                        else:
                            if data.dtype == torch.int64:
                                weight_accumulator[name].add_(((clients_ps[_id] / all_ps) * data).to(torch.int64))
                            else:
                                weight_accumulator[name].add_((clients_ps[_id] / all_ps) * data)
                server.model_aggregate(weight_accumulator, 1)

                # 更新历史观测梯度数据
                if len(all_grad2vector) >= conf['history_observation']:
                    all_grad2vector.pop(0)
                    all_global_grad2vector.pop(0)
                global_grad2vector = grad2vector_func(weight_accumulator)
                all_global_grad2vector.append(global_grad2vector)
                # 利用全局梯度替换本轮恶意用户梯度
                for _id in list(grad2vector.keys()):
                    if _id not in top_k_client:
                        grad2vector[_id] = global_grad2vector
                all_grad2vector.append(grad2vector)
        elif conf["method"] == "vert+krum":
            if e < 2:
                for c in candidates:
                    if c.poisoner:
                        c.poisoner = False
                for c in candidates:
                    c.set_model(server.global_model)
                    diff = c.local_train(server.global_model)
                    map_diff[c.client_id] = diff
                    c.del_model()
                for key in map_diff.keys():
                    diff = map_diff[key]
                    grad2vector[key] = grad2vector_func(diff)
                all_grad2vector.append(grad2vector)
                for _id in map_diff.keys():
                    diff = map_diff[_id]
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(((1 / conf["candidates"]) * data).to(torch.int64))
                        else:
                            if data.dtype == torch.int64:
                                weight_accumulator[name].add_(((1 / conf["candidates"]) * data).to(torch.int64))
                            else:
                                weight_accumulator[name].add_((1 / conf["candidates"]) * data)
                server.model_aggregate(weight_accumulator, 1)

                global_grad2vector = grad2vector_func(weight_accumulator)
                all_global_grad2vector.append(global_grad2vector)
            else:
                for c in candidates:
                    c.set_model(server.global_model)
                    diff = c.local_train(server.global_model)
                    map_diff[c.client_id] = diff
                    c.del_model()
                if conf["poison_type"] == 'ALIE':
                    map_diff = ALIE(server, map_diff, poisoner_nums)
                elif conf["poison_type"] == 'AGR':
                    map_diff = AGR(server, map_diff, poisoner_nums)
                else:
                    pass

                for key in map_diff.keys():
                    diff = map_diff[key]
                    grad2vector[key] = grad2vector_func(diff)

                start_time = time.time()
                clients_ps = {}
                all_ps = 0.
                # 训练投影器、预测器和集成系数
                projector = get_projector(conf['type'])
                predictor = get_predictor(conf['type'])
                projector.load_state_dict(torch.load('./checkpoints/client-projector.pth'))
                predictor.load_state_dict(torch.load('./checkpoints/client-predictor.pth'))
                # optim_proj = torch.optim.Adam(projector.parameters(), lr=conf['lr'])
                optim_pred = torch.optim.Adam(predictor.parameters(), lr=conf['lr'])
                # projector.train()
                predictor.train()
                a = torch.load('./checkpoints/client-coefficient-a.pt')
                b = torch.load('./checkpoints/client-coefficient-b.pt')
                a_coefficient = torch.autograd.Variable(a, requires_grad=True)
                b_coefficient = torch.autograd.Variable(b, requires_grad=True)
                optim_a = torch.optim.Adam([a_coefficient], lr=conf['lr'])
                optim_b = torch.optim.Adam([b_coefficient], lr=conf['lr'])
                for c in candidates:
                    print("train client {} projector and predictor.".format(c.client_id))
                    history_epoch = conf["history_observation"]
                    for _e in range(conf["predict_epoch"]):
                        # optim_proj.zero_grad()
                        optim_pred.zero_grad()
                        optim_a.zero_grad()
                        optim_b.zero_grad()
                        loss = 0.
                        for i in range(0, history_epoch - 1):
                            # all_grad2vector还没有存满历史梯度信息，则提前结束
                            if i == len(all_grad2vector) - 1:
                                break
                            # 历史某一轮，某个用户参与或者没参与当前轮次的情况，没参与则用全局梯度代替
                            if c.client_id in list(all_grad2vector[i].keys()):
                                local_grad2vec = all_grad2vector[i][c.client_id].detach()
                            else:
                                local_grad2vec = all_global_grad2vector[i].detach()
                            global_grad2vec = all_global_grad2vector[i].detach()
                            # 训练数据，历史梯度数据的集成
                            integration_grad2vector = a_coefficient * local_grad2vec + b_coefficient * global_grad2vec
                            # 标签
                            if c.client_id in list(all_grad2vector[i + 1].keys()):
                                target = all_grad2vector[i + 1][c.client_id]
                            else:
                                target = all_global_grad2vector[i + 1]
                            loss += torch.norm(
                                predictor(projector(integration_grad2vector)) - projector(target.detach()))

                        loss.backward()
                        # optim_proj.step()
                        optim_pred.step()
                        optim_a.step()
                        optim_b.step()

                        print("regression training epoch {} done.".format(_e))

                    # 预测，评分
                    # 将上一轮用户梯度与全局梯度进行集成
                    with torch.no_grad():
                        if c.client_id in list(all_grad2vector[len(all_grad2vector) - 1].keys()):
                            last_local_grad2vec = all_grad2vector[len(all_grad2vector) - 1][c.client_id]
                        else:
                            last_local_grad2vec = all_global_grad2vector[len(all_grad2vector) - 1]
                        last_global_grad2vec = all_global_grad2vector[len(all_grad2vector) - 1]
                        integration_grad2vector = a_coefficient * last_local_grad2vec + b_coefficient * last_global_grad2vec
                        # 梯度投影预测
                        predict_grag2vector = predictor(projector(integration_grad2vector))
                        true_grad2vector = projector(grad2vector[c.client_id])
                        # 余弦相似度
                        cs = cos(predict_grag2vector, true_grad2vector)
                        # cs=torch.norm(predict_grag2vector-true_grad2vector)
                        clients_ps[c.client_id] = cs
                    print("client {} score {}.".format(c.client_id, cs))
                    logging.info("client {} score {}.".format(c.client_id, cs))

                # 保存模型参数
                torch.save(projector.state_dict(), './checkpoints/client-projector.pth')
                torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')
                torch.save(a_coefficient, './checkpoints/client-coefficient-a.pt')
                torch.save(b_coefficient, './checkpoints/client-coefficient-b.pt')
                # 清除内存
                del projector, predictor, a_coefficient, b_coefficient
                torch.cuda.empty_cache()

                # 选择相似度最高的top-k和进行聚合
                sort_ps_list = sorted(clients_ps.items(), key=itemgetter(1))
                top_k_client = []
                for k in range(conf['top_k_score']):
                    top_k_client.append(sort_ps_list[len(sort_ps_list) - k - 1][0])
                    all_ps += sort_ps_list[len(sort_ps_list) - k - 1][1]
                end_time = time.time()
                running_times.append(end_time - start_time)

                success = 0
                topk_cs = 0.
                for _id in top_k_client:
                    topk_cs += clients_ps[_id]
                    if _id not in poisoner_nums:
                        success += 1
                print("aggregation clients: {}, success: {}".format(str(top_k_client), str(success)))
                logging.info("aggregation clients: {}, success: {}".format(str(top_k_client), str(success)))

                honest_cs = 0.
                malicious_cs = 0.
                for _id in list(clients_ps.keys()):
                    if _id in poisoner_nums:
                        malicious_cs += clients_ps[_id]
                    else:
                        honest_cs += clients_ps[_id]
                avg_topk_cs = (topk_cs / len(top_k_client)).item()
                avg_honest_cs = (honest_cs / (conf["candidates"] - len(poisoner_nums))).item()
                avg_malicious_cs = (malicious_cs / len(poisoner_nums)).item()
                all_avg_topk_cs.append(avg_topk_cs)
                all_avg_honest_cs.append(avg_honest_cs)
                all_avg_malicious_cs.append(avg_malicious_cs)

                # 更新全局模型
                closest_nums = len(top_k_client) - 1
                euclidean_distance = {}
                for id in top_k_client:
                    vec = grad2vector[id]
                    distances = []
                    for _id in top_k_client:
                        _vec = grad2vector[_id]
                        if id != _id:
                            distances.append(torch.norm(vec - _vec))
                    distances.sort()
                    sum_dis = 0.
                    for i in range(closest_nums):
                        sum_dis += distances[i]
                    euclidean_distance[id] = sum_dis
                target_id = -1
                min_dis = pow(2, 31)
                for id, dis in euclidean_distance.items():
                    if dis < min_dis:
                        min_dis = dis
                        target_id = id
                for name, params in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(map_diff[target_id][name])

                server.model_aggregate(weight_accumulator, 1)

                # 更新历史观测梯度数据
                if len(all_grad2vector) >= conf['history_observation']:
                    all_grad2vector.pop(0)
                    all_global_grad2vector.pop(0)
                global_grad2vector = grad2vector_func(weight_accumulator)
                all_global_grad2vector.append(global_grad2vector)
                # 利用全局梯度替换本轮恶意用户梯度
                for _id in list(grad2vector.keys()):
                    if _id not in top_k_client:
                        grad2vector[_id] = global_grad2vector
                all_grad2vector.append(grad2vector)
        elif conf['method'] == 'flanders':
            if e < 2:
                for c in candidates:
                    c.set_model(server.global_model)
                    diff = c.local_train(server.global_model)
                    map_diff[c.client_id] = diff
                    c.del_model()
                if conf["poison_type"] == 'ALIE':
                    map_diff = ALIE(server, map_diff, poisoner_nums)
                elif conf["poison_type"] == 'AGR':
                    map_diff = AGR(server, map_diff, poisoner_nums)
                else:
                    pass
                for key in map_diff.keys():
                    diff = map_diff[key]
                    parameter_matrix[key - 1] = grad2vector_func(diff)

                for key in map_diff.keys():
                    diff = map_diff[key]
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(data.to(torch.int64))
                        else:
                            weight_accumulator[name].add_(data)
                server.model_aggregate(weight_accumulator, conf["candidates"])

                global_grad2vector = grad2vector_func(weight_accumulator)
                all_global_grad2vector.append(global_grad2vector)
                for i in range(conf['clients']):
                    if i + 1 not in list(map_diff.keys()):
                        parameter_matrix[i] = global_grad2vector
                all_grad2vector.append(parameter_matrix)
            else:
                for c in candidates:
                    c.set_model(server.global_model)
                    diff = c.local_train(server.global_model)
                    map_diff[c.client_id] = diff
                    c.del_model()
                if conf["poison_type"] == 'ALIE':
                    map_diff = ALIE(server, map_diff, poisoner_nums)
                elif conf["poison_type"] == 'AGR':
                    map_diff = AGR(server, map_diff, poisoner_nums)
                else:
                    pass
                for key in map_diff.keys():
                    diff = map_diff[key]
                    parameter_matrix[key - 1] = grad2vector_func(diff)

                # 训练
                start_time = time.time()
                a = torch.load('./checkpoints/matrix_coefficient_a.pt')
                b = torch.load('./checkpoints/matrix_coefficient_b.pt')
                matrix_coefficient_a = torch.autograd.Variable(a, requires_grad=True)
                matrix_coefficient_b = torch.autograd.Variable(b, requires_grad=True)
                optim_a = torch.optim.Adam([matrix_coefficient_a], lr=conf['lr'])
                optim_b = torch.optim.Adam([matrix_coefficient_b], lr=conf['lr'])
                history_epoch = conf["history_observation"]
                for _e in range(conf["predict_epoch"]):
                    optim_a.zero_grad()
                    optim_b.zero_grad()
                    loss = 0.
                    for i in range(0, history_epoch - 1):
                        if i == len(all_grad2vector) - 1:
                            break
                        input = torch.mm(torch.mm(matrix_coefficient_a, all_grad2vector[i].detach()),
                                         matrix_coefficient_b)
                        target = all_grad2vector[i + 1]
                        loss += torch.norm(input - target.detach())

                    loss.backward()
                    optim_a.step()
                    optim_b.step()
                    print("regression training epoch {} done.".format(_e))

                # 预测
                last_local_grad2vec = all_grad2vector[len(all_grad2vector) - 1]
                predict = torch.mm(torch.mm(matrix_coefficient_a, last_local_grad2vec), matrix_coefficient_b)
                target = parameter_matrix
                clients_cs = {}
                all_cs = 0.
                for i in range(conf['clients']):
                    if i + 1 in list(map_diff.keys()):
                        cs = cos(predict[i], target[i])
                        clients_cs[i + 1] = cs
                        print("client {} score {}.".format(i + 1, cs))
                        logging.info("client {} score {}.".format(i + 1, cs))

                torch.save(matrix_coefficient_a, './checkpoints/matrix_coefficient_a.pt')
                torch.save(matrix_coefficient_b, './checkpoints/matrix_coefficient_b.pt')
                del matrix_coefficient_a, matrix_coefficient_b
                torch.cuda.empty_cache()

                sort_cs_list = sorted(clients_cs.items(), key=itemgetter(1))
                top_k_client = []
                for k in range(conf['top_k_score']):
                    top_k_client.append(sort_cs_list[len(sort_cs_list) - k - 1][0])
                    all_cs += sort_cs_list[len(sort_cs_list) - k - 1][1]
                end_time = time.time()
                running_times.append(end_time - start_time)

                success = 0
                for _id in top_k_client:
                    if _id not in poisoner_nums:
                        success += 1
                print("aggregation clients: {}, success: {}".format(str(top_k_client), str(success)))
                logging.info("aggregation clients: {}, success: {}".format(str(top_k_client), str(success)))

                # 更新全局模型
                for _id in top_k_client:
                    diff = map_diff[_id]
                    for name, data in diff.items():
                        if data.type() != weight_accumulator[name].type():
                            weight_accumulator[name].add_(((clients_cs[_id] / all_cs) * data).to(torch.int64))
                        else:
                            if data.dtype == torch.int64:
                                weight_accumulator[name].add_(((clients_cs[_id] / all_cs) * data).to(torch.int64))
                            else:
                                weight_accumulator[name].add_((clients_cs[_id] / all_cs) * data)
                server.model_aggregate(weight_accumulator, 1)

                # 更新历史观测梯度数据
                if len(all_grad2vector) >= conf['history_observation']:
                    all_grad2vector.pop(0)
                    all_global_grad2vector.pop(0)
                global_grad2vector = grad2vector_func(weight_accumulator)
                all_global_grad2vector.append(global_grad2vector)
                # 利用全局梯度替换本轮恶意用户梯度
                for _id in range(conf['clients']):
                    if _id + 1 not in top_k_client:
                        parameter_matrix[_id] = global_grad2vector
                all_grad2vector.append(parameter_matrix)

        else:
            print("method is unexisting!")

        acc = server.model_eval()

        if (conf["poison_type"] == "LSA" or conf["poison_type"] == 'MR' or conf["poison_type"] ==
                'ALIE' or conf["poison_type"] == 'MP' or conf["poison_type"] == 'AGR'):
            if acc > max_acc:
                max_acc = acc
            all_acc.append(acc)
            print("Global Epoch %d, acc: %f\n" % (e, acc))
            logging.info("Global Epoch %d, acc: %f\n" % (e, acc))
        else:
            all_f_acc.append(acc[0])
            if acc[0] > max_f_acc:
                max_f_acc = acc[0]
            all_o_acc.append(acc[1])
            if acc[1] > max_o_acc:
                max_o_acc = acc[1]
            print("Global Epoch %d, f_acc: %f, o_acc: %f\n" % (e, acc[0], acc[1]))
            logging.info("Global Epoch %d, f_acc: %f, o_acc: %f\n" % (e, acc[0], acc[1]))

    if (conf["poison_type"] == "LSA" or conf["poison_type"] == 'MR' or conf["poison_type"] ==
            'ALIE' or conf["poison_type"] == 'MP' or conf["poison_type"] == 'AGR'):
        print(all_acc)
        logging.info(str(all_acc))
        print(max_acc)
        logging.info('max:{}'.format(max_acc))
        print(np.mean(np.array(all_acc)))
        logging.info('Average:{}'.format(np.mean(np.array(all_acc))))
        print('all average top k cos: {}'.format(all_avg_topk_cs))
        logging.info('all average top k cos: {}'.format(all_avg_topk_cs))
        print('all average honest cos: {}'.format(all_avg_honest_cs))
        logging.info('all average honest cos: {}'.format(all_avg_honest_cs))
        print('all average malicious cos: {}'.format(all_avg_malicious_cs))
        logging.info('all average malicious cos: {}'.format(all_avg_malicious_cs))
        print('all running times: {}'.format(str(running_times)))
        logging.info('all running times: {}'.format(str(running_times)))
    else:
        print(all_f_acc)
        print(all_o_acc)
        logging.info(str(all_f_acc))
        logging.info(str(all_o_acc))
        print(max_f_acc)
        logging.info(str(max_f_acc))
        print(max_o_acc)
        logging.info(str(max_o_acc))


if __name__ == '__main__':
    with open("conf.json", 'r') as f:
        conf = json.load(f)

    main(conf)
