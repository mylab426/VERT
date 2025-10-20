from operator import itemgetter

import numpy as np
import torch

from models.ProjectHead import mnist_predictor, mnist_project_head


def grad2vector_func(diff):
    flag = False
    vector = None
    for name, data in diff.items():
        data = data.reshape(-1)
        if flag is not True:
            vector = data
            flag = True
        else:
            vector = torch.cat((vector, data), dim=0)
    return vector


def model2vector_func(model):
    flag = False
    vector = None
    for name, data in model.state_dict().items():
        data = data.reshape(-1)
        if flag is not True:
            vector = data
            flag = True
        else:
            vector = torch.cat((vector, data), dim=0)
    return vector


def shortest_dis(p1, v1, p2, v2):
    p1 = p1.cpu().numpy()
    v1 = v1.cpu().numpy()
    p2 = p2.cpu().numpy()
    v2 = v2.cpu().numpy()
    D = np.dot((p1 - p2), v1)
    E = np.dot((p1 - p2), v2)
    A = np.dot(v1, v1)
    C = np.dot(v1, v2)
    B = np.dot(v2, v2)

    s = (A * E - D * C) / (A * B - C * C)
    t = (C * E - B * D) / (A * B - C * C)

    if s < 0:
        s = 1
    if t < 0:
        t = 1

    # print(s, t)

    _p1 = p1 + t * v1
    _p2 = p2 + s * v2
    dis = np.linalg.norm(_p1 - _p2)
    return _p1, _p2, dis


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
    a = torch.autograd.Variable(torch.ones_like(vector, dtype=torch.float32), requires_grad=True)
    b = torch.autograd.Variable(torch.ones_like(vector, dtype=torch.float32), requires_grad=True)
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
        # return 87267, 128
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
        return mnist_project_head(input, output).cuda()
    elif type == 'cifar10':
        return mnist_project_head(input, output).cuda()
    else:
        pass


def get_predictor(type):
    input, output = get_predictor_io(type)
    if type == 'mnist':
        return mnist_predictor(input, output).cuda()
    elif type == 'cifar10':
        return mnist_predictor(input, output).cuda()
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

    # 右乘矩阵太大了不能运行怎么办，可以用某一层的参数进行替换
    matrix_coefficient_b = torch.autograd.Variable(
        torch.ones(torch.Size((len(vector), len(vector))), dtype=torch.float32), requires_grad=True)
    matrix_coefficient_a = torch.autograd.Variable(torch.ones(torch.Size((clients, clients)), dtype=torch.float32),
                                                   requires_grad=True)
    torch.save(matrix_coefficient_b, './checkpoints/matrix_coefficient_b.pt')
    torch.save(matrix_coefficient_a, './checkpoints/matrix_coefficient_a.pt')
    # parameter_matrix = torch.ones(torch.Size((clients, len(vector))), dtype=torch.float32)

    # return parameter_matrix


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                # 梯度上升，最大化损失
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # 梯度下降，最小化损失
        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


def K_means():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs

    X=[0.011, 0.028, 0.030, 0.033, 0.037, 0.038, 0.04, 0.041, 0.042]
    _X=np.column_stack([X, np.zeros_like(X)])
    n_clusters = 2
    # 创建KMeans实例
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # 拟合模型并预测聚类标签
    predicted_labels = kmeans.fit_predict(_X)
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    print(predicted_labels)
    print(centers)






if __name__ == '__main__':
    print(max(0,1))



