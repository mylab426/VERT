import json
import logging
import random

from defenses import fedavg, krum, median, vert_fedavg, vert_krum, flanders, flame, FLDetector, FLBeeline, FedREDefense, \
    MUDHoG, RFLPA
from server import Server
from client import *
from utils import datasets
from attacks import ALIE, mm_AGR, ms_AGR

import os

from utils.utils import get_flanders_coefficient, set_coefficient, get_projector_io, set_project, \
    get_predictor_io, set_predictor, model2vector_func

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.9,max_split_size_mb:512'


def main(conf):
    if conf['method'] in ['vert+krum','vert+fedavg','fldetector','flanders','FLBeeline+topk','fedredefense']:
        filename = conf["method"] + ',' + conf['model_name'] + ',' + conf["type"] + ",n=" + str(
            conf["clients"]) + '!' + str(conf["candidates"]) + ",top-" + str(conf['top_k_score']) + ",m=" + str(
            conf["poisoner_rate"]) + ',' + conf["poison_type"] + ",d_alpha=" + str(conf["dirichlet_alpha"]) + '.log'
    else:
        filename = conf["method"] + ',' + conf['model_name'] + ',' + conf["type"] + ",n=" + str(
            conf["clients"]) + '!' + str(conf["candidates"]) + ",m=" + str(
            conf["poisoner_rate"]) + ',' + conf["poison_type"] + ",d_alpha=" + str(conf["dirichlet_alpha"]) + '.log'
    filename = 'test.log'
    logging.basicConfig(level=logging.INFO,
                        filename="/home/ln/JinBo/code/Against-Large-Scale-Poison/log/" + filename,
                        filemode='w')

    data_root = "/home/ln/JinBo/code/data/" + conf["type"]
    train_datasets, eval_datasets = datasets.get_dataset(data_root, conf["type"])

    server = Server(conf, eval_datasets)

    if conf["poison_type"] in ['badnets', 'blend']:
        server.set_backdoor_eval_dataloader(conf["poison_type"])
    clients = []
    client_idx = datasets.dirichlet_nonIID_data(train_datasets, conf)
    for c in range(conf["clients"]):
        clients.append(Client(conf, train_datasets, client_idx[c + 1], c + 1))

    random.seed(1234)
    poisoners = random.sample(clients, int(conf["clients"] * conf["poisoner_rate"]))
    # poisoners = [clients[0]]
    all_poisoner_set = []
    for p in poisoners:
        p.poisoner = True
        p.set_poison_dataloader()
        all_poisoner_set.append(p.client_id)
    for _id in range(1, conf["clients"] + 1):
        if _id not in all_poisoner_set:
            print("honest client {}.".format(_id))
            logging.info("honest client {}.".format(_id))
        else:
            print("malicious client {}.".format(_id))
            logging.info("malicious client {}.".format(_id))

    if conf['method'] == 'flanders':
        # parameter_matrix = get_flanders_coefficient(server.global_model, conf["clients"])
        get_flanders_coefficient(server.global_model, conf["clients"])
        # pass
    if 'vert' in conf['method']:
        set_coefficient(server.global_model, conf["clients"])
        input, output = get_projector_io(conf['type'])
        set_project(input, output, conf)
        input, output = get_predictor_io(conf['type'])
        set_predictor(output, output, conf)

    all_acc = []
    all_asr = []
    all_grad2vector = []
    all_global_grad2vector = []
    all_avg_topk_cs = [0., 0.]
    all_avg_honest_cs = [0., 0.]
    all_avg_malicious_cs = [0., 0.]
    running_times = []
    all_global_weight = []
    all_success_rates=[]
    last_grad2vector= {}
    local_knowledge= {}
    global_grad=[0]
    for e in range(conf["global_epochs"]):
        print("Global Epoch %d" % e)
        candidates = random.sample(clients, conf["candidates"])
        # 存储全局模型参数
        all_global_weight.append(model2vector_func(server.global_model))
        if e < 2:
            for c in poisoners:
                c.poisoner = False
        elif e == 2:
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
        for c in candidates:
            c.set_model()
            diff = c.local_train(server.global_model)
            map_diff[c.client_id] = diff
            c.del_model()
        # attack
        if len(poisoner_nums)>0:
            if conf["poison_type"] == 'ALIE':
                map_diff = ALIE(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'mm-AGR':
                map_diff = mm_AGR(server, map_diff, poisoner_nums)
            elif conf["poison_type"] == 'ms-AGR':
                map_diff = ms_AGR(server, map_diff, poisoner_nums)

        # defense
        if conf["method"] == "fedavg":
            fedavg(map_diff, weight_accumulator)
        elif conf["method"] == "krum":
            krum(map_diff, weight_accumulator, conf, server)
        elif conf["method"] == "median":
            median(map_diff, weight_accumulator, server)
        elif conf["method"] == "vert+fedavg":
            vert_fedavg(e, candidates, map_diff, weight_accumulator, poisoner_nums, conf,
                        all_grad2vector, all_global_grad2vector, all_avg_topk_cs, all_avg_honest_cs,
                        all_avg_malicious_cs)
        elif conf["method"] == "vert+krum":
            vert_krum(e, candidates, map_diff, weight_accumulator, poisoner_nums, conf,
                      all_grad2vector, all_global_grad2vector, all_avg_topk_cs, all_avg_honest_cs,
                      all_avg_malicious_cs)
        elif conf['method'] == 'flanders':
            flanders(e, candidates, map_diff, weight_accumulator, poisoner_nums, conf, all_grad2vector)
        elif conf['method'] == 'flame':
            flame(map_diff, weight_accumulator, server, epsilon=3000, delta=0.01)
        elif conf['method'] == 'fldetector':
            FLDetector(e, map_diff, weight_accumulator, last_grad2vector, all_global_grad2vector, all_global_weight,
                       conf, poisoner_nums, all_success_rates)
        elif conf['method'] == 'fedredefense':
            FedREDefense(e, map_diff, weight_accumulator, conf, poisoner_nums, local_knowledge, server)
        elif conf['method'] == 'mudhog':
            MUDHoG(e, map_diff, weight_accumulator, conf, poisoner_nums, all_grad2vector)
        elif conf['method'] == 'rflpa':
            RFLPA(e, map_diff, weight_accumulator, conf, server, global_grad)
        elif 'FLBeeline' in conf['method']:
            FLBeeline(e, candidates, map_diff, weight_accumulator, all_grad2vector, all_global_weight, poisoner_nums,
                      conf, all_success_rates, all_avg_honest_cs, all_avg_malicious_cs)

        else:
            print("method is unexisting!")

        server.model_aggregate(weight_accumulator)

        if conf["poison_type"] in ["LSA", 'ALIE', 'MP', 'mm-AGR', 'ms-AGR']:
            acc = server.eval()
            all_acc.append(acc)
            print("Global Epoch %d, acc: %f\n" % (e, acc))
            logging.info("Global Epoch %d, acc: %f\n" % (e, acc))
        else:
            asr = server.backdoor_eval(target_label=1)
            all_asr.append(asr)
            print("Global Epoch %d, asr: %f\n" % (e, asr))
            logging.info("Global Epoch %d, asr: %f\n" % (e, asr))

    if conf["poison_type"] in ["LSA", 'ALIE', 'MP', 'mm-AGR', 'ms-AGR']:
        print(all_acc)
        logging.info(str(all_acc))
    else:
        print(all_asr)
        logging.info(str(all_asr))
    print('all average top k cos: {}'.format(all_avg_topk_cs))
    logging.info('all average top k cos: {}'.format(all_avg_topk_cs))
    print('all average honest cos: {}'.format(all_avg_honest_cs))
    logging.info('all average honest cos: {}'.format(all_avg_honest_cs))
    print('all average malicious cos: {}'.format(all_avg_malicious_cs))
    logging.info('all average malicious cos: {}'.format(all_avg_malicious_cs))
    print('all running times: {}'.format(str(running_times)))
    logging.info('all running times: {}'.format(str(running_times)))


if __name__ == '__main__':
    with open("conf.json", 'r') as f:
        conf = json.load(f)

    main(conf)
