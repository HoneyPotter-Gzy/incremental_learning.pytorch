import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

from inclearn.lib import utils


def closest_to_mean(features, nb_examplars):
    features = features / (np.linalg.norm(features, axis=0) + 1e-8)
    class_mean = np.mean(features, axis=0)

    return _l2_distance(features, class_mean).argsort()[:nb_examplars]


def icarl_selection(features, nb_examplars):  # 传入特征，要选择的exemplars个数
    D = features.T  # features是【样本个数*256维】，D：256维*样本个数，features是每行一个样本，D是每列一个样本
    D = D / (np.linalg.norm(D, axis=0) + 1e-8)  # 沿0轴归一化，
    mu = np.mean(D, axis=1)  # 沿1轴求均值，求到的是所有样本的样本均值
    herding_matrix = np.zeros((features.shape[0],))  # 初始化一个和features大小一样的一维0向量

    w_t = mu  # 样本均值，就是μ
    iter_herding, iter_herding_eff = 0, 0

    while not (  # 开始筛选
        np.sum(herding_matrix != 0) == min(nb_examplars, features.shape[0])
    ) and iter_herding_eff < 1000:
        tmp_t = np.dot(w_t, D)
        ind_max = np.argmax(tmp_t)
        iter_herding_eff += 1
        if herding_matrix[ind_max] == 0:
            herding_matrix[ind_max] = 1 + iter_herding
            iter_herding += 1

        w_t = w_t + mu - D[:, ind_max]

    herding_matrix[np.where(herding_matrix == 0)[0]] = 10000

    return herding_matrix.argsort()[:nb_examplars]


def random(features, nb_examplars):
    return np.random.permutation(len(features))[:nb_examplars]


def kmeans(features, nb_examplars, k=5):
    """Samples examplars for memory according to KMeans.

    :param features: The image features of a single class.
    :param nb_examplars: Number of images to keep.
    :param k: Number of clusters for KMeans algo, defaults to 5
    :return: A numpy array of indexes.
    """
    model = KMeans(n_clusters=k)
    cluster_assignements = model.fit_predict(features)

    nb_per_clusters = nb_examplars // k
    indexes = []
    for c in range(k):
        c_indexes = np.random.choice(np.where(cluster_assignements == c)[0], size=nb_per_clusters)
        indexes.append(c_indexes)

    return np.concatenate(indexes)


def confusion(ypreds, ytrue, nb_examplars, class_id=None, minimize_confusion=True):
    """Samples examplars for memory according to the predictions.

    :param ypreds: All the predictions (shape [b, c]).
    :param ytrue: The true label.
    :param nb_examplars: Number of images to keep.
    :param minimize_confusion: Samples easiest examples or hardest.
    """
    indexes = np.where(ytrue == class_id)[0]
    ypreds, ytrue = ypreds[indexes], ytrue[indexes]

    ranks = ypreds.argsort(axis=1)[:, ::-1][np.arange(len(ypreds)), ytrue]

    indexes = ranks.argsort()
    if minimize_confusion:
        return indexes[:nb_examplars]
    return indexes[-nb_examplars:]


def minimize_confusion(inc_dataset, network, memory, class_index, nb_examplars):
    _, new_loader = inc_dataset.get_custom_loader(class_index, mode="test")
    new_features, _ = utils.extract_features(network, new_loader)
    new_mean = np.mean(new_features, axis=0)

    from sklearn.cluster import KMeans

    n_clusters = 4
    model = KMeans(n_clusters=n_clusters)
    model.fit(new_features)

    indexes = []
    for i in range(n_clusters):
        cluster = model.cluster_centers_[i]
        distances = _l2_distance(cluster, new_features)

        indexes.append(distances.argsort()[:nb_examplars // n_clusters])

    return np.concatenate(indexes)

    if memory is None:
        # First task
        #return icarl_selection(new_features, nb_examplars)
        return np.random.permutation(new_features.shape[0])[:nb_examplars]

    distances = _l2_distance(new_mean, new_features)

    data_memory, targets_memory = memory
    for indexes in _split_memory_per_class(targets_memory):
        _, old_loader = inc_dataset.get_custom_loader(
            [], memory=(data_memory[indexes], targets_memory[indexes]), mode="test"
        )

        old_features, _ = utils.extract_features(network, old_loader)
        old_mean = np.mean(old_features, axis=0)

        # The larger the distance to old mean
        distances -= _l2_distance(old_mean, new_features)

    return distances.argsort()[:int(nb_examplars)]


def var_ratio(memory_per_class, network, loader, select="max", type=None):
    var_ratios = []
    for input_dict in loader:
        inputs = input_dict["inputs"].to(network.device)
        with torch.no_grad():
            outputs = network(inputs)
        var_ratios.append(outputs["var_ratio"])
    var_ratios = np.concatenate(var_ratios)

    indexes = var_ratios.argsort()
    if select == "max":
        return indexes[-memory_per_class:]
    elif select == "min":
        return indexes[:memory_per_class]
    raise ValueError("Only possible value for <select> are [max, min], not {}.".format(select))


def mcbn(memory_per_class, network, loader, select="max", nb_samples=100, type=None):
    if not hasattr(network.convnet, "sampling_mode"):
        raise ValueError("Network must be MCBN-compatible.")
    network.convnet.sampling_mode()

    all_probs = []
    for input_dict in loader:
        inputs = input_dict["inputs"].to(network.device)

        probs = []
        for _ in range(nb_samples):
            with torch.no_grad():
                outputs = network(inputs)
                logits = outputs["logits"]
                probs.append(F.softmax(logits, dim=-1).cpu().numpy())

        probs = np.stack(probs)
        all_probs.append(probs)
    network.convnet.normal_mode()

    all_probs = np.concatenate(all_probs, axis=1)
    var_ratios = _var_ratio(all_probs.transpose(1, 0, 2))

    indexes = var_ratios.argsort()
    assert len(indexes) == all_probs.shape[1]
    if select == "max":
        return indexes[-memory_per_class:]
    elif select == "min":
        return indexes[:memory_per_class]
    raise ValueError("Only possible value for <select> are [max, min], not {}.".format(select))

def triple():
    pass



# def grad_based_selection_origin(features, grad, nb_exemplars):
#     # 基于梯度的样本选择
#     # 基于梯度的样本选择，每个minibatch选择一次
#     # 考虑两个部分：新知识的代表性和旧知识的损害
#     # 新知识的代表性：可以有四种实现：样本原始特征均值、产生的梯度更新均值、输出的高维特征空间均值、另接一个特征映射器产生的均值
#     # 旧知识的损害：对于旧模型的梯度更新损害
#     # 先实现第一种
#     # params: samples: minibatch的所有样本
#     #         grad: 模型此次更新的梯度
#     #         nb_exemplars: 要选择的样本个数
#     D = features.T  # 转置
#     D = D / (np.linalg.norm(D, axis = 0) + 1e-8)  # 归一化，压缩行，对各列求均值，返回1*n，前面已经转置了
#     mu = np.mean(D, axis = 1)  # 计算每一行的均值，这里返回一个实数（就是正文中的μ
#
#     herding_matrix = np.zeros((features.shape[0],))  # 初始化一个和features行数一样的n*1向量
#
#     w_t = mu  # 实数，
#     iter_herding, iter_herding_eff = 0, 0  # eff？是什么
#
#     while not (  # 开始筛选
#             # herding_matrix中非0的元素个数==应当挑选的样本数，意思就是还没挑完
#             np.sum(herding_matrix != 0) == min(nb_exemplars, samples.shape[0])
#     ) and iter_herding_eff < 1000:
#         tmp_t = np.dot(w_t, D)  # w_t和D相乘
#         ind_max = np.argmax(tmp_t)  # 找到tmp_t中最大元素对应的下标，会对高维数组降为至1维，返回压缩后的下标
#         iter_herding_eff += 1  # eff+=1
#         if herding_matrix[ind_max] == 0:  # 如果ind_max是0
#             herding_matrix[ind_max] = 1 + iter_herding  # 在第iter轮添加的
#             iter_herding += 1
#
#         w_t = w_t + mu - D[:, ind_max]
#
#     herding_matrix[np.where(herding_matrix == 0)[0]] = 10000
#
#     return herding_matrix.argsort()[:nb_exemplars]
#
#
# def gradient_based(exam_grad, exemplars_grad, nb_exemplars):
#     """
#     基于梯度更新的样本选择方法。
#     @param exam_grad:  样本梯度
#     @param exemplars_grad: 历史样本梯度
#     @param nb_exemplars:  需要选择的样本个数
#     """
#     # herding_index = np.zeros((exam_grad.shape[0],))
#
#     # 计算样本梯度均值
#     exam_grad_T = exam_grad.T
#     exam_grad_T = exam_grad_T / (np.linalg.norm(exam_grad_T, axis = 0) + 1e-8)
#     exam_mean_grad = np.mean(exam_grad_T, axis = 1)
#
#     # 计算历史样本梯度均值
#     exemplars_grad_T = exemplars_grad.T
#     exemplars_grad_T = exemplars_grad_T / (np.linalg.norm(exemplars_grad_T,
#                                                           axis = 0) + 1e-8)
#     exemplars_mean_grad = np.mean(exemplars_grad_T, axis = 1)
#
#     score = cosine_similarity(exam_grad, exam_mean_grad) + \
#             cosine_similarity(exam_grad, exemplars_mean_grad)
#
#     # 选出top K个样本
#     index_max = np.argpartition(score,
#                                 nb_exemplars,
#                                 axis= 0)[-nb_exemplars, len(score)]
#     # herding_index[index_max] = 1
#     # 对应的样本
#     # herding_exemplars = [index_max]
#     # return herding_exemplars
#     return index_max


# ---------
# Utilities
# ---------


def _var_ratio(sampled_probs):
    predicted_class = sampled_probs.max(axis=2)

    hist = np.array(
        [
            np.histogram(predicted_class[i, :], range=(0, 10))[0]
            for i in range(predicted_class.shape[0])
        ]
    )

    return 1. - hist.max(axis=1) / sampled_probs.shape[1]


def _l2_distance(x, y):
    return np.power(x - y, 2).sum(-1)


def _split_memory_per_class(targets):
    max_class = max(targets)

    for class_index in range(max_class):
        yield np.where(targets == class_index)[0]