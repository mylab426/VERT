import copy
import logging
import time
from operator import itemgetter
import hdbscan
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from utils.utils import grad2vector_func, get_projector, get_predictor, shortest_dis

cos = torch.nn.CosineSimilarity(dim=-1)


def fedavg(map_diff, weight_accumulator):
    for _id in map_diff.keys():
        diff = map_diff[_id]
        for name, data in diff.items():
            if data.type() != weight_accumulator[name].type():
                weight_accumulator[name].add_(((1 / len(map_diff)) * data).to(torch.int64))
            else:
                if data.dtype == torch.int64:
                    weight_accumulator[name].add_(((1 / len(map_diff)) * data).to(torch.int64))
                else:
                    weight_accumulator[name].add_((1 / len(map_diff)) * data)



def krum(map_diff, weight_accumulator, conf, server):
    grad = {}
    grad2vector = {}
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
    closest_nums = conf["candidates"] - 2
    euclidean_distance = {}
    for id, vec in grad2vector.items():
        distances = []
        for _id, _vec in grad2vector.items():
            if id != _id:
                distances.append(np.linalg.norm(vec.cpu().numpy() - _vec.cpu().numpy()))
        # distances.sort()
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




def median(map_diff, weight_accumulator, server):
    grad2vector = []
    # start_time=time.time()
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
    # end_time = time.time()
    for name, params in server.global_model.state_dict().items():
        length = 1
        for l in params.size():
            length *= l
        _params = median_vector[start_index:start_index + length]
        _params = torch.tensor(torch.tensor(_params).cuda())
        start_index += length
        if _params.type() != weight_accumulator[name].type():
            weight_accumulator[name].add_((_params.reshape(params.size())).to(torch.int64))
        else:
            weight_accumulator[name].add_(_params.reshape(params.size()))


def vert_fedavg(e, candidates, map_diff, weight_accumulator, poisoner_nums, conf, all_grad2vector,
                all_global_grad2vector, all_avg_topk_cs, all_avg_honest_cs, all_avg_malicious_cs):
    grad2vector = {}
    if e < 2:
        for key in map_diff.keys():
            diff = map_diff[key]
            grad2vector[key] = grad2vector_func(diff).cuda()
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

        global_grad2vector = grad2vector_func(weight_accumulator)
        all_global_grad2vector.append(global_grad2vector.cuda())
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)
            all_global_grad2vector.pop(0)
    else:
        for key in map_diff.keys():
            diff = map_diff[key]
            grad2vector[key] = grad2vector_func(diff)

        start_time = time.time()
        clients_ps = {}
        # 训练投影器、预测器和集成系数
        projector = get_projector(conf['type'])
        predictor = get_predictor(conf['type'])
        # projector.load_state_dict(torch.load('./checkpoints/client-projector.pth'))
        # predictor.load_state_dict(torch.load('./checkpoints/client-predictor.pth'))
        # optim_proj = torch.optim.Adam(projector.parameters(), lr=conf['lr'])
        optim_pred = torch.optim.Adam(predictor.parameters(), lr=conf['lr'])
        # projector.train()
        predictor.train()
        a = torch.load('./checkpoints/client-coefficient-a.pt')
        b = torch.load('./checkpoints/client-coefficient-b.pt')
        a_coefficient = torch.autograd.Variable(a, requires_grad=True).cuda()
        b_coefficient = torch.autograd.Variable(b, requires_grad=True).cuda()
        optim_a = torch.optim.Adam([a_coefficient], lr=conf['lr'])
        optim_b = torch.optim.Adam([b_coefficient], lr=conf['lr'])
        for c in candidates:
            # predictor = get_predictor(conf['type'])
            # optim_pred = torch.optim.Adam(predictor.parameters(), lr=conf['lr'])
            # predictor.train()
            # a = torch.load('./checkpoints/client-coefficient-a.pt')
            # b = torch.load('./checkpoints/client-coefficient-b.pt')
            # a_coefficient = torch.autograd.Variable(a, requires_grad=True).cuda()
            # b_coefficient = torch.autograd.Variable(b, requires_grad=True).cuda()
            # optim_a = torch.optim.Adam([a_coefficient], lr=conf['lr'])
            # optim_b = torch.optim.Adam([b_coefficient], lr=conf['lr'])

            print("train client {} predictor.".format(c.client_id))
            history_epoch = conf["history_observation"]
            # if (e-5)%4 == 0:
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
                    loss += torch.norm(predictor(projector(integration_grad2vector)) - projector(target.detach()))

                loss.backward()
                # optim_proj.step()
                optim_pred.step()
                optim_a.step()
                optim_b.step()
                # print("regression training epoch {} done, loss{}".format(_e, loss))

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
                clients_ps[c.client_id] = cs.cpu().item()

        # 保存模型参数
        # torch.save(projector.state_dict(), './checkpoints/client-projector.pth')
        # torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')
        # torch.save(a_coefficient, './checkpoints/client-coefficient-a.pt')
        # torch.save(b_coefficient, './checkpoints/client-coefficient-b.pt')
        # 清除内存
        del predictor, a_coefficient, b_coefficient

        sort_ps_list = sorted(clients_ps.items(), key=itemgetter(1))
        for i in range(len(sort_ps_list)):
            print("client {} score {}.".format(sort_ps_list[i][0], sort_ps_list[i][1]))
            logging.info("client {} score {}.".format(sort_ps_list[i][0], sort_ps_list[i][1]))

        # 选择相似度最高的top-k和进行聚合
        top_k_client = []
        for k in range(conf['top_k_score']):
            top_k_client.append(sort_ps_list[-k-1][0])

        # 利用K-means cluster
        # sort_client_dis_list = []
        # top_k_client=[]
        # for d in sort_ps_list:
        #     sort_client_dis_list.append(d[1])
        #
        # print(sort_ps_list)
        # _sort_client_dis_list = np.column_stack([sort_client_dis_list, np.zeros_like(sort_client_dis_list)])
        # kmeans = KMeans(n_clusters=2, random_state=42)
        # labels = kmeans.fit_predict(_sort_client_dis_list)
        # target_label=labels[-1]
        # print(labels, target_label)
        # for i in range(len(labels)):
        #     if labels[i]==target_label:
        #         top_k_client.append(sort_ps_list[i][0])

        success = 0
        topk_cs = 0.
        for _id in top_k_client:
            topk_cs += clients_ps[_id]
            if _id not in poisoner_nums:
                success += 1
        print("aggregation clients: {}, all {}, success: {}".format(top_k_client, len(top_k_client), success))
        logging.info("aggregation clients: {}, all {}, success: {}".format(top_k_client, len(top_k_client), success))

        honest_cs = 0.
        malicious_cs = 0.
        for _id in list(clients_ps.keys()):
            if _id in poisoner_nums:
                malicious_cs += clients_ps[_id]
            else:
                honest_cs += clients_ps[_id]
        avg_topk_cs = (topk_cs / len(top_k_client))
        avg_honest_cs = (honest_cs / (conf["candidates"] - len(poisoner_nums)))
        avg_malicious_cs = (malicious_cs / len(poisoner_nums))
        all_avg_topk_cs.append(avg_topk_cs)
        all_avg_honest_cs.append(avg_honest_cs)
        all_avg_malicious_cs.append(avg_malicious_cs)
        print('avg topk score: {}'.format(avg_topk_cs))
        logging.info('avg topk score: {}'.format(avg_topk_cs))
        print('avg honest score: {}'.format(avg_honest_cs))
        logging.info('avg honest score: {}'.format(avg_honest_cs))
        print('avg malicious score: {}'.format(avg_malicious_cs))
        logging.info('avg malicious score: {}'.format(avg_malicious_cs))

        # 更新全局模型
        for _id in top_k_client:
            diff = map_diff[_id]
            for name, data in diff.items():
                if data.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                else:
                    if data.dtype == torch.int64:
                        weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                    else:
                        weight_accumulator[name].add_((1 / len(top_k_client)) * data)

        # 更新历史观测梯度数据
        global_grad2vector = grad2vector_func(weight_accumulator)
        all_global_grad2vector.append(global_grad2vector.cuda())
        # 利用全局梯度替换本轮恶意用户梯度
        for _id in list(grad2vector.keys()):
            if _id not in top_k_client:
                grad2vector[_id] = global_grad2vector.cuda()
        all_grad2vector.append(grad2vector)
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)
            all_global_grad2vector.pop(0)


def vert_krum(e, candidates, map_diff, weight_accumulator, poisoner_nums, conf,
                all_grad2vector, all_global_grad2vector, all_avg_topk_cs, all_avg_honest_cs,
                all_avg_malicious_cs):
    grad2vector = {}
    if e < 5:
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
        global_grad2vector = grad2vector_func(weight_accumulator)
        all_global_grad2vector.append(global_grad2vector)
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)
            all_global_grad2vector.pop(0)
    else:
        for key in map_diff.keys():
            diff = map_diff[key]
            grad2vector[key] = grad2vector_func(diff)

        start_time = time.time()
        clients_ps = {}
        all_ps = 0.
        # 训练投影器、预测器和集成系数
        projector = get_projector(conf['type'])
        predictor = get_predictor(conf['type'])
        # projector.load_state_dict(torch.load('./checkpoints/client-projector.pth'))
        # predictor.load_state_dict(torch.load('./checkpoints/client-predictor.pth'))
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

                # print("regression training epoch {} done.".format(_e))

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
        # torch.save(projector.state_dict(), './checkpoints/client-projector.pth')
        # torch.save(predictor.state_dict(), './checkpoints/client-predictor.pth')
        # torch.save(a_coefficient, './checkpoints/client-coefficient-a.pt')
        # torch.save(b_coefficient, './checkpoints/client-coefficient-b.pt')
        # 清除内存
        del projector, predictor, a_coefficient, b_coefficient
        # torch.cuda.empty_cache()

        # 选择相似度最高的top-k和进行聚合
        sort_ps_list = sorted(clients_ps.items(), key=itemgetter(1))
        top_k_client = []
        for k in range(conf['top_k_score']):
            top_k_client.append(sort_ps_list[len(sort_ps_list) - k - 1][0])
            all_ps += sort_ps_list[len(sort_ps_list) - k - 1][1]
        # end_time = time.time()
        # running_times.append(end_time - start_time)

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
        print('avg topk score: {}'.format(avg_topk_cs))
        logging.info('avg topk score: {}'.format(avg_topk_cs))
        print('avg honest score: {}'.format(avg_honest_cs))
        logging.info('avg honest score: {}'.format(avg_honest_cs))
        print('avg malicious score: {}'.format(avg_malicious_cs))
        logging.info('avg malicious score: {}'.format(avg_malicious_cs))

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

        for name, data in map_diff[target_id].items():
            weight_accumulator[name].add_(map_diff[target_id][name])

        # 更新历史观测梯度数据
        global_grad2vector = grad2vector_func(weight_accumulator)
        all_global_grad2vector.append(global_grad2vector)
        # 利用全局梯度替换本轮恶意用户梯度
        for _id in list(grad2vector.keys()):
            if _id not in top_k_client:
                grad2vector[_id] = global_grad2vector
        all_grad2vector.append(grad2vector)
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)
            all_global_grad2vector.pop(0)


def MUDHoG(e, map_diff, weight_accumulator, conf, poisoner_nums, all_grad2vector):
    grad2vector = {}
    if e < 5:
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
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)
    else:
        user_gradients={}
        for grad2vector in all_grad2vector:
            for _id in list(grad2vector.keys()):
                if _id in list(user_gradients.keys()):
                    user_gradients[_id].append(grad2vector[_id].cpu().numpy())
                else:
                    user_gradients[_id]=[grad2vector[_id].cpu().numpy()]
        s_hog = {}
        median_hog=[]
        for _id in list(user_gradients.keys()):
            s_hog[_id] = torch.mean(torch.tensor(user_gradients[_id]), dim=0)
            median_hog.append(np.array(s_hog[_id]))

        sorted_median_hog = np.sort(median_hog, axis=0)
        length = len(sorted_median_hog)
        if length & 1:
            median_hog = sorted_median_hog[int(length / 2)]
        else:
            median_hog = (sorted_median_hog[int(length / 2) - 1] + sorted_median_hog[int(length / 2)]) / 2

        target_ids=[]
        for _id in list(s_hog.keys()):
            if cos(torch.tensor(median_hog), s_hog[_id])>0 and _id in map_diff.keys():
                target_ids.append(_id)

        success = 0
        for _id in target_ids:
            if _id not in poisoner_nums:
                success += 1
        print("aggregation clients: {}, success rate: {}".format(target_ids, success/len(target_ids)))
        logging.info("aggregation clients: {}, success: {}".format(target_ids, success/len(target_ids)))

        for _id in target_ids:
            diff = map_diff[_id]
            for name, data in diff.items():
                if data.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_(((1 / len(target_ids)) * data).to(torch.int64))
                else:
                    if data.dtype == torch.int64:
                        weight_accumulator[name].add_(((1 / len(target_ids)) * data).to(torch.int64))
                    else:
                        weight_accumulator[name].add_((1 / len(target_ids)) * data)

        for key in map_diff.keys():
            diff = map_diff[key]
            grad2vector[key] = grad2vector_func(diff)
        all_grad2vector.append(grad2vector)
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)


def RFLPA(e, map_diff, weight_accumulator, conf, server, global_grad):
    if e == 0:
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
    else:
        grad2vector = {}
        for key in map_diff.keys():
            diff = map_diff[key]
            grad2vector[key] = grad2vector_func(diff)

        for _id in grad2vector.keys():
            grad2vector[_id]=(torch.norm(global_grad[0])/torch.norm(grad2vector[_id]))*grad2vector[_id]
            _diff=dict()
            start_index = 0
            for name, params in server.global_model.state_dict().items():
                length = 1
                for l in params.size():
                    length *= l
                _params = grad2vector[_id][start_index:start_index + length]
                _diff[name]=_params.reshape(params.shape)
                start_index += length
            map_diff[_id]=_diff

        mu = {}
        all_mu = 0.
        for _id in grad2vector.keys():
            mu[_id] = max(0, cos(global_grad[0], grad2vector[_id]))
            all_mu += mu[_id]

        for _id in map_diff.keys():
            diff = map_diff[_id]
            for name, data in diff.items():
                if data.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_(((mu[_id] / (all_mu+1e-7)) * data).to(torch.int64))
                else:
                    if data.dtype == torch.int64:
                        weight_accumulator[name].add_(((mu[_id] / (all_mu+1e-7)) * data).to(torch.int64))
                    else:
                        weight_accumulator[name].add_((mu[_id] / (all_mu+1e-7)) * data)

    global_grad[0] = grad2vector_func(weight_accumulator)


def flanders(e, candidates, map_diff, weight_accumulator, poisoner_nums, conf, all_grad2vector):
    parameter_matrix = torch.ones(torch.Size((conf['clients'], 87267)), dtype=torch.float32).cuda()
    if e < 5:
        for c in candidates:
            if c.poisoner:
                c.poisoner = False
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

        global_grad2vector = grad2vector_func(weight_accumulator)
        for i in range(conf['clients']):
            if i + 1 not in list(map_diff.keys()):
                parameter_matrix[i] = global_grad2vector
        all_grad2vector.append(parameter_matrix)
    else:
        for key in map_diff.keys():
            diff = map_diff[key]
            parameter_matrix[key - 1] = grad2vector_func(diff)
        # 训练
        start_time = time.time()
        a = torch.load('./checkpoints/matrix_coefficient_a.pt')
        b = torch.load('./checkpoints/matrix_coefficient_b.pt')
        matrix_coefficient_a = torch.autograd.Variable(a, requires_grad=True)
        matrix_coefficient_b = torch.autograd.Variable(b, requires_grad=True)
        optim_a = torch.optim.SGD([matrix_coefficient_a], lr=conf['lr'])
        optim_b = torch.optim.SGD([matrix_coefficient_b], lr=conf['lr'])
        history_epoch = conf["history_observation"]
        for _e in range(conf["predict_epoch"]):
            optim_a.zero_grad()
            optim_b.zero_grad()
            loss = 0.
            for i in range(0, history_epoch - 1):
                if i == len(all_grad2vector) - 1:
                    break
                predict = torch.mm(torch.mm(matrix_coefficient_a, all_grad2vector[i].detach()), matrix_coefficient_b)
                target = all_grad2vector[i + 1]
                loss += torch.norm(predict - target.detach()) ** 2

            loss.backward()
            optim_a.step()
            optim_b.step()
            print("regression training epoch {} done.".format(_e))

        # 预测
        last_local_grad2vec = all_grad2vector[-1]
        predict = torch.mm(torch.mm(matrix_coefficient_a, last_local_grad2vec), matrix_coefficient_b)
        target = parameter_matrix
        clients_cs = {}
        for i in range(conf['clients']):
            if i + 1 in list(map_diff.keys()):
                cs = torch.norm(predict[i]-target[i]) ** 2
                clients_cs[i + 1] = cs
                print("client {} score {}.".format(i + 1, cs))
                logging.info("client {} score {}.".format(i + 1, cs))

        # torch.save(matrix_coefficient_a, './checkpoints/matrix_coefficient_a.pt')
        # torch.save(matrix_coefficient_b, './checkpoints/matrix_coefficient_b.pt')
        del matrix_coefficient_a, matrix_coefficient_b

        sort_cs_list = sorted(clients_cs.items(), key=itemgetter(1))
        top_k_client = []
        for k in range(conf['top_k_score']):
            top_k_client.append(sort_cs_list[k][0])

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
                    weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                else:
                    if data.dtype == torch.int64:
                        weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                    else:
                        weight_accumulator[name].add_((1 / len(top_k_client)) * data)

        # 更新历史观测梯度数据
        global_grad2vector = grad2vector_func(weight_accumulator)
        # 利用全局梯度替换本轮恶意用户梯度
        for _id in range(conf['clients']):
            if _id + 1 not in top_k_client:
                parameter_matrix[_id] = global_grad2vector
        all_grad2vector.append(parameter_matrix)
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)


def flame(map_diff, weight_accumulator, server, epsilon=3000, delta=0.01):
    gradients=[]
    for key in map_diff.keys():
        diff = map_diff[key]
        gradients.append(grad2vector_func(diff))

    n = len(gradients)
    # compute pairwise cosine distances
    cos_dist = torch.zeros((n, n), dtype=torch.double).cuda()
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - F.cosine_similarity(gradients[i], gradients[j], dim=0, eps=1e-9)
            cos_dist[i, j], cos_dist[j, i] = d, d

    # clustering of gradients
    np_cos_dist = cos_dist.cpu().numpy()
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=1, min_cluster_size=(n // 2) + 1,
                                cluster_selection_epsilon=0.0, allow_single_cluster=True).fit(np_cos_dist)

    # compute clipping bound
    euclid_dist = []
    for grad in gradients:
        euclid_dist.append(torch.norm(grad, p=2))

    clipping_bound, _ = torch.median(torch.stack(euclid_dist).reshape((-1, 1)), dim=0)

    # gradient clipping
    clipped_gradients = []
    for i in range(n):
        if clusterer.labels_[i] == 0:
            gamma = clipping_bound / euclid_dist[i]
            clipped_gradients.append(gradients[i] * torch.min(torch.ones((1,)).cuda(), gamma))

    # aggregation
    # global_update = torch.mean(torch.cat(clipped_gradients, dim=0), dim=-1)
    global_update = torch.mean(torch.stack(clipped_gradients), dim=0)
    # adaptive noise
    std = (clipping_bound * np.sqrt(2 * np.log(1.25 / delta)) / epsilon) ** 2
    global_update += torch.normal(mean=0, std=std.item(), size=tuple(global_update.size())).cuda()

    start_index = 0
    for name, params in server.global_model.state_dict().items():
        length = len(params.view(-1))
        _params = global_update[start_index:start_index + length]
        start_index += length
        if _params.type() != weight_accumulator[name].type():
            weight_accumulator[name].add_((_params.reshape(params.size())).to(torch.int64))
        else:
            weight_accumulator[name].add_(_params.reshape(params.size()))


def FLDetector(e, map_diff, weight_accumulator, last_grad2vector, all_global_grad2vector, all_global_weight, conf, poisoner_nums,  all_success_rates):

    def lbfgs(weights, grads, v):
        S_k_list=[]
        Y_k_list=[]
        for i in range(len(weights)):
            if i==0:
                continue
            S_k_list.append(weights[i].reshape(-1, 1)-weights[i-1].reshape(-1, 1)) # (n,1)
            Y_k_list.append(grads[i].reshape(-1, 1)-grads[i-1].reshape(-1, 1))

        curr_S_k = torch.cat(S_k_list, dim=1) # (n,m) 模型参数量，用户量
        curr_Y_k = torch.cat(Y_k_list, dim=1) # (n,m)
        S_k_time_Y_k = torch.mm(curr_S_k.T, curr_Y_k) # (m,m)
        S_k_time_S_k = torch.mm(curr_S_k.T, curr_S_k) # (m,m)
        R_k = torch.triu(S_k_time_Y_k) # 上三角 (m,m)
        L_k = S_k_time_Y_k - R_k # 下三角 (m,m)
        sigma_k = torch.mm(Y_k_list[-1].T, S_k_list[-1]) / (torch.mm(S_k_list[-1].T, S_k_list[-1])) # 常数
        D_k_diag = torch.diag(S_k_time_Y_k) # 对角线元素 (1,m)
        upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1) # (m,2m)
        lower_mat = torch.cat([L_k.T, -torch.diag(D_k_diag)], dim=1) # (m,2m)
        mat = torch.cat([upper_mat, lower_mat], dim=0) # (2m,2m)
        mat_inv = torch.inverse(mat) # (2m,2m)

        v=v.reshape(-1,1) # (n,1)
        approx_prod = sigma_k * v # (n,1)
        p_mat = torch.cat([torch.mm(curr_S_k.T, sigma_k * v), torch.mm(curr_Y_k.T, v)], dim=0) # (2m,1)
        approx_prod -= torch.mm(torch.mm(torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat) # (n,1)
        return approx_prod.reshape(1, -1) # (1,n)

    def predict(cur_gradients, old_gradients, hvp):
        distance = {}
        for _id in list(old_gradients.keys()):
            if _id in list(cur_gradients.keys()):
                pred_grad=old_gradients[_id] + hvp
                distance[_id]=torch.norm(pred_grad-cur_gradients[_id])
        return distance

    cur_grad2vector = {}
    for key in map_diff.keys():
        diff = map_diff[key]
        cur_grad2vector[key] = grad2vector_func(diff)

    if e < 5:
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
        global_grad2vector = grad2vector_func(weight_accumulator)
        all_global_grad2vector.append(global_grad2vector)
    else:
        start_time=time.time()
        hvp=lbfgs(all_global_weight[:len(all_global_grad2vector)], all_global_grad2vector, all_global_weight[-1]-all_global_weight[-2])
        # 如果上一轮用户没有参与怎么办？ 作fldetector对比试验时，可以单独设置用户参与度为100%
        distance=predict(cur_grad2vector, last_grad2vector, hvp)
        sort_ps_list = sorted(distance.items(), key=itemgetter(1))
        top_k_client = []
        for k in range(conf['top_k_score']):
            top_k_client.append(sort_ps_list[k][0])
        print('aggregation client: {}'.format(top_k_client))
        logging.info('aggregation client: {}'.format(top_k_client))
        errors = 0
        for _id in poisoner_nums:
            if _id in top_k_client:
                errors += 1
        all_success_rates.append(1 - errors / len(top_k_client))
        print('success rate:{}'.format(all_success_rates[-1]))
        logging.info('success rate:{}'.format(all_success_rates[-1]))

        for _id in top_k_client:
            diff = map_diff[_id]
            for name, data in diff.items():
                if data.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_(((1/conf['top_k_score']) * data).to(torch.int64))
                else:
                    if data.dtype == torch.int64:
                        weight_accumulator[name].add_(((1/conf['top_k_score']) * data).to(torch.int64))
                    else:
                        weight_accumulator[name].add_((1/conf['top_k_score']) * data)

        # 更新历史观测梯度数据
        global_grad2vector = grad2vector_func(weight_accumulator)
        all_global_grad2vector.append(global_grad2vector)
        if len(all_global_grad2vector) > conf['history_observation']:
            all_global_grad2vector.pop(0)
            all_global_weight.pop(0)

        end_time = time.time()
        print(end_time - start_time)

    for id in list(cur_grad2vector.keys()):
        last_grad2vector[id]=cur_grad2vector[id]


def FedREDefense(e, map_diff, weight_accumulator, conf, poisoner_nums, local_knowledge, server):
    def kd_loss(output, y):
        soft_label = F.softmax(y, dim=1)
        # soft_label = y
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(- soft_label * logsoftmax(output))

    ipc=1
    for _id in map_diff.keys():
        if _id not in local_knowledge.keys():
            hard_label = [np.ones(ipc, dtype=np.int64) * i for i in range(conf['classes'])]
            label_syn = torch.nn.functional.one_hot(torch.tensor(hard_label).reshape(-1), num_classes=conf['classes']).float()
            label_syn = label_syn * 0.
            label_syn = label_syn.detach().cuda().requires_grad_(True)
            image_syn = torch.randn(size=(conf['classes'] * ipc, conf['channels'], conf['size'], conf['size']), dtype=torch.float)
            syn_lr = torch.tensor(0.1).cuda()
            image_syn = image_syn.detach().cuda().requires_grad_(True)
            syn_lr = syn_lr.detach().cuda().requires_grad_(True)
            local_knowledge[_id]=[image_syn, label_syn, syn_lr]

    all_re_loss={}
    for _id in map_diff.keys():
        syn_images = local_knowledge[_id][0].requires_grad_(True)
        y_hat = local_knowledge[_id][1].requires_grad_(True)
        syn_lr = local_knowledge[_id][2].requires_grad_(True)
        optimizer_img = torch.optim.SGD([syn_images], lr=0.1, momentum=0.5)
        optimizer_label = torch.optim.SGD([y_hat], lr=0.1, momentum=0.5)
        optimizer_lr = torch.optim.SGD([syn_lr], lr=5e-5, momentum=0.5)
        for i in range(100):
            virtual_model=copy.deepcopy(server.global_model).cuda()
            vir_optimizer = torch.optim.SGD(virtual_model.parameters(), lr=syn_lr)
            for step in range(5):
                indices = torch.randperm(len(syn_images))
                x = syn_images[indices]
                this_y = y_hat[indices]
                _, x = virtual_model(x)
                ce_loss = kd_loss(x, this_y)
                vir_optimizer.zero_grad()
                ce_loss.backward()
                vir_optimizer.step()

            virtual_gradient=dict()
            for name, data in virtual_model.state_dict().items():
                if data.dtype == torch.int64:
                    virtual_gradient[name] = torch.tensor(data - server.global_model.state_dict()[name], dtype=torch.float32).requires_grad_(True)
                else:
                    virtual_gradient[name] = (data - server.global_model.state_dict()[name]).requires_grad_(True)

            re_loss = torch.tensor(0.0).cuda()
            target_grad = map_diff[_id]
            for name, data in virtual_model.state_dict().items():
                re_loss+=torch.norm(virtual_gradient[name]-target_grad[name]) ** 2

            target_norm=torch.norm(grad2vector_func(target_grad)) ** 2
            re_loss/=target_norm

            # if re_loss.detach().cpu() < 0.6:
            #     break

            optimizer_img.zero_grad()
            optimizer_label.zero_grad()
            optimizer_lr.zero_grad()
            re_loss.backward()
            optimizer_img.step()
            optimizer_lr.step()
            optimizer_label.step()

        local_knowledge[_id][0]=syn_images
        local_knowledge[_id][1]=y_hat
        local_knowledge[_id][2]=syn_lr

        all_re_loss[_id]=re_loss.item()

        print('client {} reconstruct done.'.format(_id))

    sort_ps_list = sorted(all_re_loss.items(), key=itemgetter(1))
    for i in range(len(sort_ps_list)):
        print("client {} re loss {}.".format(sort_ps_list[i][0], sort_ps_list[i][1]))
        logging.info("client {} re loss {}.".format(sort_ps_list[i][0], sort_ps_list[i][1]))
    top_k_client = []
    for k in range(conf['top_k_score']):
        top_k_client.append(sort_ps_list[k][0])

    success = 0
    for _id in top_k_client:
        if _id not in poisoner_nums:
            success += 1

    print("aggregation clients: {}, success: {}".format(top_k_client, success))
    logging.info("aggregation clients: {}, success: {}".format(top_k_client, success))

    # 更新全局模型
    for _id in top_k_client:
        diff = map_diff[_id]
        for name, data in diff.items():
            if data.type() != weight_accumulator[name].type():
                weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
            else:
                if data.dtype == torch.int64:
                    weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                else:
                    weight_accumulator[name].add_((1 / len(top_k_client)) * data)


def FLBeeline(e, candidates, map_diff, weight_accumulator, all_grad2vector, all_global_weight, poisoner_nums, conf,
              all_success_rates, all_avg_honest_cs, all_avg_malicious_cs):
    grad2vector = {}
    if e < conf["history_observation"]:
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
    else:
        for key in map_diff.keys():
            diff = map_diff[key]
            grad2vector[key] = grad2vector_func(diff)
        all_grad2vector.append(grad2vector)
        if len(all_grad2vector) > conf['history_observation']:
            all_grad2vector.pop(0)
            all_global_weight.pop(0)

        # 计算用户历史梯度之间的最短距离
        client_dis={}
        client_point={}
        for c in candidates:
            history_epoch = conf["history_observation"]
            all_dis=[]
            all_points=[]
            for h in range(history_epoch):
                if c.client_id in all_grad2vector[h].keys():
                    h_client_grad = all_grad2vector[h][c.client_id]
                else:
                    continue
                h_client_weight=all_global_weight[h]
                for _h in range(h+1, history_epoch):
                    if c.client_id in all_grad2vector[_h].keys():
                        _h_client_grad=all_grad2vector[_h][c.client_id]
                    else:
                        continue
                    _h_client_weight = all_global_weight[_h]
                    _p1, _p2, dis = shortest_dis(h_client_weight, h_client_grad, _h_client_weight, _h_client_grad)
                    # all_points.append(_p1)
                    # all_points.append(_p2)
                    all_dis.append(dis)

            client_dis[c.client_id]=np.mean(all_dis)
            # client_point[c.client_id]=np.mean(all_points, axis=0)

        sort_client_dis = sorted(client_dis.items(), key=itemgetter(1))
        honest_dis=[]
        poison_dis=[]
        for d in sort_client_dis:
            if d[0] in poisoner_nums:
                print('poison {}, {}.'.format(d[0], d[1]))
                logging.info('poison {}, {}.'.format(d[0], d[1]))
                poison_dis.append(d[1])
            else:
                print('honest {}, {}.'.format(d[0], d[1]))
                logging.info('honest {}, {}.'.format(d[0], d[1]))
                honest_dis.append(d[1])
        all_avg_honest_cs.append(np.mean(honest_dis))
        all_avg_malicious_cs.append(np.mean(poison_dis))
        print('honest average dis:{}'.format(all_avg_honest_cs[-1]))
        logging.info('honest average dis:{}'.format(all_avg_honest_cs[-1]))
        print('poison average dis:{}'.format(all_avg_malicious_cs[-1]))
        logging.info('poison average dis:{}'.format(all_avg_malicious_cs[-1]))

        top_k_client=[]
        if conf['method'] == 'FLBeeline+topk':
            for i in range(conf['top_k_score']):
                top_k_client.append(sort_client_dis[i][0])
        elif conf['method'] == 'FLBeeline+hdbscan':
            sort_client_dis_list=[]
            for d in sort_client_dis:
                sort_client_dis_list.append(d[1])
            _sort_client_dis_list = np.column_stack([sort_client_dis_list, np.zeros_like(sort_client_dis_list)])
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5,
                                        cluster_selection_epsilon=5,
                                        min_samples=1,
                                        metric='euclidean')
            clusterer.fit(_sort_client_dis_list)
            labels = clusterer.labels_  # 每个点的聚类标签(-1表示噪声点)
            for i in range(len(labels)):
                if labels[i]==0:
                    top_k_client.append(sort_client_dis[i][0])
        elif conf['method'] == 'FLBeeline+kmeans':
            sort_client_dis_list = []
            for d in sort_client_dis:
                sort_client_dis_list.append(d[1])
            _sort_client_dis_list = np.column_stack([sort_client_dis_list, np.zeros_like(sort_client_dis_list)])
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(_sort_client_dis_list)
            zero_nums, one_nums = 0, 0
            for l in labels:
                if l == 0:
                    zero_nums += 1
                else:
                    one_nums += 1
            if zero_nums < one_nums:
                t = 0
            else:
                t = 1
            for i in range(len(labels)):
                if labels[i] == t:
                    top_k_client.append(sort_client_dis[i][0])

        print('aggregation client: {}'.format(top_k_client))
        errors=0
        for _id in poisoner_nums:
            if _id in top_k_client:
                print('poison client:{}'.format(_id))
                errors += 1
        all_success_rates.append(1 - errors/len(top_k_client))
        print('success rate:{}'.format(all_success_rates[-1]))
        logging.info('success rate:{}'.format(all_success_rates[-1]))

        # end_time = time.time()

        for _id in top_k_client:
            diff = map_diff[_id]
            for name, data in diff.items():
                if data.type() != weight_accumulator[name].type():
                    weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                else:
                    if data.dtype == torch.int64:
                        weight_accumulator[name].add_(((1 / len(top_k_client)) * data).to(torch.int64))
                    else:
                        weight_accumulator[name].add_((1 / len(top_k_client)) * data)



