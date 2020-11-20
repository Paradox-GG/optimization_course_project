# Optimization course project
# With GPL licence
# Xiangyu Gao

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from openpyxl import load_workbook

from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn import metrics


def normlization(x):
    # 数据归一化处理
    m = np.mean(x, 0, keepdims=True)
    sigma = np.std(x, 0, keepdims=True)
    return (x - m) / sigma, m, sigma


def eval_the_binmodel(pred, gt, pos=1):
    # 用于二分类分类器指标评价 (used for binary classification)
    tp, fp, fn, tn = 0, 0, 0, 0
    for p, g in zip(pred, gt):
        if p == g:
            if g == pos:
                tp += 1
            else:
                tn += 1
        else:
            if g == pos:
                fn += 1
            else:
                fp += 1
    se = tp / (tp+fn)                         # 敏感性
    sp = tn / (tn+fp)                         # 特异性
    acc = (tp+tn)/(tp+fp+tn+fn)               # 准确率

    # area under curve
    fpr, tpr, thresholds = metrics.roc_curve(gt, pred, pos_label=pos)
    auc = metrics.auc(fpr, tpr)
    print('when positive label is {}, SE: {:.2f}%, SP: {:.2f}%, ACC: {:.2f}%, AUC: {:.2f}.'.format(pos, 100*se, 100*sp, 100*acc, auc))
    return se, sp, acc, auc


def split_by_label(X, y):
    d = dict()
    for label in set(y):
        d[label] = []
    for k, label in enumerate(y):
        d[label].append(X[k, :])
    for k in d:
        d[k] = np.array(d[k])
    return d


def cal_inclass_diff(X, X_m):
    X_mc = np.mean(X, axis=0, keepdims=True)
    Sw = (X - X_mc).T @ (X - X_mc) / X.shape[0]
    Sb = (X_mc - X_m).T @ (X_mc - X_m)
    return Sw, Sb


def cal_fitness(X, y):
    samples_dict = split_by_label(X, y)
    X_m = np.mean(X, axis=0, keepdims=True)
    P_priors = [samples_dict[one_class].shape[0] / X.shape[0] for one_class in samples_dict]
    Sws_and_Sbs = [cal_inclass_diff(samples_dict[one_class], X_m) for one_class in samples_dict]

    Sw = np.zeros_like(Sws_and_Sbs[0][0])
    Sb = np.zeros_like(Sws_and_Sbs[0][1])
    for p, swb in zip(P_priors, Sws_and_Sbs):
        Sw += p * swb[0]
        Sb += p * swb[1]

    return np.trace(Sb) / np.trace(Sw)


def select(community, X, y):

    fitness_in_a_generation = list()
    for gene in community:
        fitness_in_a_generation.append(cal_fitness(X[:, np.where(gene)[0]], y))

    fitness_in_a_generation = np.array(fitness_in_a_generation)

    # @@ selection enhancement @@
    enhance = 7
    fitness_in_a_generation_ = np.exp(enhance * fitness_in_a_generation)
    # fitness_in_a_generation_ = fitness_in_a_generation

    select_possibility_in_a_generation = fitness_in_a_generation_ / np.sum(fitness_in_a_generation_)
    belongs_to = np.cumsum(select_possibility_in_a_generation)

    new_community = list()
    for k in range(len(community)):
        who = np.random.rand()
        new_community.append(community[np.sum(belongs_to < who)].copy())   # 不使用copy()方法就会影响后续操作！！！

    max_adapation_in_this_generation = np.max(fitness_in_a_generation)
    return new_community, max_adapation_in_this_generation


def cross(community, p_mutate, K):
    community = community[:]
    random.shuffle(community)
    for k in range(0, len(community), 2):
        p = np.random.rand()
        if p < p_mutate:
            gene1 = community[k].copy()                               # 不要和原列表搅在一起， 下面要进行交叉
            gene2 = community[k+1].copy()

            start_point = np.random.randint(len(gene1))

            new_gene1 = np.hstack((gene1[:start_point].copy(), gene2[start_point:].copy()))      # 交叉项不要和上面的关联
            new_gene2 = np.hstack((gene2[:start_point].copy(), gene1[start_point:].copy()))

            if sum(new_gene1) == K and sum(new_gene2) == K:        # 为了保证选出来的特征一定有K个， 交叉结果必须有两个选中， 否则不交换
                community[k] = new_gene1                           # 下一次会有新的， 不用担心关联
                community[k+1] = new_gene2
    return community


def mutate(community, p_mutate):
    for k in range(len(community)):
        p = np.random.rand()
        if p < p_mutate:
            remaining = np.where(community[k])[0]
            mutate_point = remaining[0]
            while mutate_point in remaining:                         # 保证码重约束的同时， 实现最小码距变化（最小变动码距：2）
                mutate_point = np.random.randint(len(community[k]))
            community[k][mutate_point] = 1
            community[k][np.random.choice(remaining)] = 0


def initialize_the_community(N, K, group_num_limit=True):
    '''
    if the number of schemes is close toto the number of the genes,
    traverse is better.
    '''
    period = N // K
    limit_num = 16

    left_num = N - (K - 1) * (N // K)

    community = list()
    one_poses = list()
    flag = False
    init_one_pos = np.array(list(range(0, N, period))[:K])
    for bb in range(left_num):
        for b in range(N // K):
            new_one_pos = np.zeros((K), dtype=np.int)
            new_one_pos[:-1] = init_one_pos[:-1].copy() + b
            new_one_pos[-1] = init_one_pos[-1].copy() + bb
            one_poses.append(new_one_pos)

            if len(one_poses) == limit_num and group_num_limit:
                flag = True
            if flag:
                break
        if flag:
            break

    if len(one_poses) % 2 != 0:
        one_poses = one_poses[:-1]

    for one_pos in one_poses:
        gene = np.zeros((N))
        gene[one_pos] = 1
        community.append(gene)
    print('the number of genes: ', len(community))
    return community


def random_initialize_community(N, K):
    group_num = 16
    community = list()
    for k in range(group_num):
        gene = np.zeros((N))
        gene[np.random.choice(np.arange(N), size=K, replace=False)] = 1
        community.append(gene)
    return community


def GA_select_features(X, y, iter_times, K):
    '''our initialization method'''

    assert K < X.shape[1], 'the number of selected features should be less than the number of original features!'

    community = initialize_the_community(X.shape[1], K)
    p_cross = 0.1
    p_mutate = 0.05

    max_fitness = list()
    for iter_time in range(iter_times):
        community_selected, max_fitness_in_this_iter = select(community, X, y)
        max_fitness.append(max_fitness_in_this_iter)
        if iter_time < iter_times - 1:
            community = cross(community_selected, p_cross, K)
            mutate(community, p_mutate)
    return community, max_fitness


def GA_select_features_(X, y, iter_times, K):
    '''the random initializatio method.'''

    community = random_initialize_community(X.shape[1], K)
    p_cross = 0.1
    p_mutate = 0.05

    max_fitness = list()
    for iter_time in range(iter_times):
        community_selected, max_fitness_in_this_iter = select(community, X, y)
        max_fitness.append(max_fitness_in_this_iter)
        if iter_time < iter_times - 1:
            community = cross(community_selected, p_cross, K)
            mutate(community, p_mutate)
    return community, max_fitness


def svm_classification(X, y, kno):
    kernels = ["linear", "poly", "rbf", "sigmoid"]   # 本次作业中选择的核函数
    kernel = kernels[kno]
    print()
    print('choose the kernel: {}.'.format(kernel))
    index = list(range(len(y)))
    kf = KFold(n_splits=5)
    m_rs = list()
    for k, (train_index, val_index) in enumerate(kf.split(index[:], index[:])):
        print('for fold {}:'.format(k + 1))

        X_val, y_val, X_train, y_train = X[val_index, :], y[val_index], X[train_index, :], y[train_index]
        # print(X_val.shape, X_train.shape)
        support_vector_machine = svm.SVC(kernel=kernel)
        support_vector_machine.fit(X_train, y_train)

        pred = support_vector_machine.predict(X_val)

        se, sp, acc, auc = eval_the_binmodel(pred, y_val, 1)
        m_rs.append([k + 1, se, sp, acc, auc])
    return m_rs


def select_the_features(community, X, y):
    max_fitness = 0
    best_scheme = community[0].copy()
    for gene in community:
        fitness = cal_fitness(X[:, np.where(gene)[0]], y)
        if max_fitness < fitness:
            best_scheme = np.where(gene)[0]
            max_fitness = fitness
    return best_scheme


def create_the_sheet(filepath, sheet_name, content, columns, rows):
    writer = pd.ExcelWriter(filepath, engine='openpyxl')
    book = load_workbook(filepath)
    writer.book = book
    df = pd.DataFrame(content, columns=columns, index=rows)
    df.to_excel(writer, sheet_name=sheet_name)
    writer.save()


def get_avg_acc(rf):
    avg_acc = 0
    for r in rf:
        avg_acc += r[-2]
    print('avg acc: ', avg_acc / len(rf))


def main():
    exp_datasets = load_breast_cancer()

    X = exp_datasets.data    # 纵轴是样本数， 横轴是特征数
    y = exp_datasets.target  # 标签

    if type(X) != np.ndarray:
        X = np.array(X)
    if type(y) != np.ndarray:
        y = np.array(y)

    # 归一化， 对任务有影响
    X, _, _ = normlization(X)            # 样本数x特征数

    # 选K个特征
    K = 5
    iter_times = 1000

    # different initialization methods
    community, max_adap_var = GA_select_features(X, y, iter_times, K)
    community_, max_adap_var_ = GA_select_features_(X, y, iter_times, K)

    # for gene in community:
    #     print(gene)

    plt.figure(1)
    plt.plot(max_adap_var, color='b', linestyle='-')
    plt.plot(max_adap_var_, color='r', linestyle='--')
    plt.legend(['our initialization method', 'random initialization method'])
    plt.title('GA')
    plt.ylabel('max fitness in each generation')
    plt.xlabel('iter times')
    plt.show()

    selected_features_no = select_the_features(community, X, y)
    print('selected features: ', selected_features_no)
    print('max fitness after iterations: {}, {}.'.format(max_adap_var[-1], max_adap_var_[-1]))

    print(X.shape)
    res_full_features = svm_classification(X, y, 0)

    print(X[:, selected_features_no].shape)
    res_selected_features = svm_classification(X[:, selected_features_no], y, 0)

    # random_selected_features_no = np.random.choice(np.arange(X.shape[1]), size=K, replace=False)
    # random_res = svm_classification(X[:, random_selected_features_no], y, 0)
    # print(random_selected_features_no, cal_fitness(X[:, random_selected_features_no], y))
    # create_the_sheet(filepath='random_res.xlsx', sheet_name='sheet1', content=random_res,
    #                                    columns=['fold', 'se', 'sp', 'acc', 'auc'], rows=list(range(1, 6)))

    get_avg_acc(res_full_features)
    get_avg_acc(res_selected_features)
    # get_avg_acc(random_res)

    # create_the_sheet(filepath='enhance_10.xlsx',  sheet_name='sheet1', content=res_full_features,
    #                  columns=['fold', 'se', 'sp', 'acc', 'auc'], rows=list(range(1, 6)))
    # create_the_sheet(filepath='enhance_10.xlsx', sheet_name='sheet2', content=res_selected_features,
    #                  columns=['fold', 'se', 'sp', 'acc', 'auc'], rows=list(range(1, 6)))


if __name__ == '__main__':
    main()
