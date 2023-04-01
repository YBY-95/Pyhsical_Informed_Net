import torch
import math
import numpy as np
import matplotlib.pyplot as plt

class PhiLayer_1D(torch.nn.Module):
    """
    一维信号，物理信息接入的神经网络层
    """
    def __init__(self, sim_feature, num_class, in_features=4096, out_features=4096, bias=True):
        super(PhiLayer_1D, self).__init__()
        self.num_class = num_class
        self.in_features = in_features
        self.out_features = out_features
        self.sim_feature_index, self.sim_feature_amp, self.sim_feature_num = feature_detec(sim_feature, 1)
        self.shape_parameter = torch.nn.Parameter(torch.zeros([int(num_class), np.sum(self.sim_feature_num)*2]))
        torch.nn.init.constant(self.shape_parameter, 0.5)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros([self.num_class, int(np.sum(self.sim_feature_num)*2)]))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs, n_harmonic=1):
        weights = torch.zeros([self.num_class, self.in_features]).cuda()
        out = torch.zeros(inputs.shape).cuda()
        x = inputs
        for i in range(self.num_class):
            for index, j in enumerate(self.sim_feature_index[i].T):
                weights[i, :] += RBF_comp(self.in_features, j, n_harmonic,
                                          self.shape_parameter[i, index*2],
                                          self.shape_parameter[i, index*2+1])
            weights[i, :] += weights[i, :] + 1
        weights = torch.tensor(weights).cuda()
        for k in range(self.num_class):
            weight_class = weights[0].repeat(2, 1)
            weight_class = weight_class.unsqueeze(0).unsqueeze(0)
            out += weight_class * x

        return out

    # def reset_parameters(self):


    def output_num(self):
        return self.out_features


def RBF_comp(data_len, f_loc, n_harmonic, shape_p1, shape_p2):
    """
    基于径向基高斯衰减函数的仿真信号权重构建
    :param data_len: 数据长度
    :param f_loc: 特征频率位置
    :param n_harmonic: 谐波数量
    :param shape_p1: 波形参数1
    :param shape_p2: 波形参数2
    :return:weight_seq: 权重序列
    """
    RBF_function = lambda l, l1, a, b, n: a*math.exp(-(l-n*l1)**2/(2*b**2))

    map_list = list(np.array([np.arange(0, data_len, 1),
                              f_loc[1] * np.ones((data_len)),
                              float(shape_p1) * np.ones((data_len)),
                              float(shape_p2) * np.ones((data_len)),
                              n_harmonic * np.ones((data_len))]))
    weight_seq = list(map(RBF_function, map_list[0], map_list[1], map_list[2], map_list[3], map_list[4]))

    # plt.plot(np.array(weight_seq))
    # plt.show()

    return torch.Tensor(weight_seq).cuda()


def feature_detec(feature, threshold):
    """
    筛选信号中大于阈值的频率成分
    :param feature: 信号  size: channel*data_len
    :param threshold: 阈值
    :return: index, amp  索引、幅值
    """
    amp = []
    index_ = []
    num = []
    data = np.array(feature)
    index = np.where(data > threshold)
    for i in range(feature.shape[0]):
        index_i = np.where(index[0] == i)
        index_.append(np.vstack((index[0][index_i], index[1][index_i])))
        amp.append(feature[index_[0][0], index_[0][1]])
        num.append(len(feature[index_[0][0], index_[0][1]]))

    return index_, amp, num