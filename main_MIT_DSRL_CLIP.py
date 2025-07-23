# import wandb
import torch
import torch.optim as optim
import numpy as np
from model_DSRL_CLIP import DSRL
from dataset_CLIP import DataLoader
from classifier_DSRL_CLIP import eval_zsl
import pickle
import yaml
import scipy.io as scio

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.linalg import solve_sylvester

st_scaler = StandardScaler()
# rs_scaler = RobustScaler()
# mm_scaler = MinMaxScaler(feature_range=(0, 1))
# ma_scaler = MaxAbsScaler()
pca = PCA(n_components=0.99)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Config:
    def __init__(self, data):
        self._data = data
        for key in data:
            self.__dict__[key] = data[key]['value']

    def __str__(self):
        res = '{'
        for key in self._data:
            res += '\''
            res += key
            res += '\': '
            if type(data[key]['value']) == str:
                res += '\''
                res += data[key]['value']
                res += '\''
            else:
                res += str(data[key]['value'])
            res += ', '
        res = res[:-2] + '}'
        return res


def norm_cols(mat_in):
    l2norms = np.sqrt(np.sum(mat_in * mat_in, axis=1)).reshape((mat_in.shape[0], 1))
    mat_out = mat_in / np.tile(l2norms, (1, mat_in.shape[1]))
    return mat_out


def generate_x_p_s(train_num, dataloader_data, random=False, device='cpu'):
    feature = dataloader_data['feature']

    # data = scio.loadmat('places365_resnet152_67.mat')
    # feature = data['feature']
    # feature_st = pca.fit_transform(feature)    # dimension reduction

    # feature_st = norm_cols(feature)
    feature_st = st_scaler.fit_transform(feature)
    # feature_st = feature

    label_list = np.array(dataloader_data['label_list'])
    if random:
        train_label_index = np.random.choice(len(label_list), train_num, replace=False)
    else:
        train_label_index = range(0, train_num)

    seenclasses = label_list[train_label_index]
    unseenclasses = np.setdiff1d(label_list, seenclasses)
    train_p = np.array([])
    s = []
    train_index_list = np.array([]).astype(int)
    test_index_list = np.array([]).astype(int)
    train_label_list = np.array([])
    test_label_list = np.array([])
    current_index = 0
    i = 0
    for key, value in dataloader_data['label_num'].items():
        if np.isin(key, seenclasses):
            train_index_list = np.append(train_index_list, range(current_index, current_index + value))
            train_label_list = np.append(train_label_list, [key] * value)
            s_temp = [0] * train_num
            s_temp[i] = 1
            i += 1
            s.append(s_temp)
            # KMeans train_P
            K = 20   # 10
            cluster_i = KMeans(n_clusters=K, random_state=0).fit(feature_st[current_index: current_index + value])
            train_p_i = cluster_i.cluster_centers_
            train_p_i = np.tile(train_p_i, (value, 1))
            train_p_i = train_p_i.reshape(value, K, train_p_i.shape[-1])
            if train_p.shape[0] == 0:
                train_p = train_p_i
            else:
                train_p = np.vstack((train_p, train_p_i))
        else:
            test_index_list = np.append(test_index_list, range(current_index, current_index + value))
            test_label_list = np.append(test_label_list, [key] * value)
        current_index += value

    s_array = np.array(s)
    # feature_st = st_scaler.fit_transform(feature)
    train_feature = feature_st[train_index_list, :]
    test_feature = feature_st[test_index_list, :]

    train_p = st_scaler.fit_transform(train_p.reshape(train_p.shape[0]*train_p.shape[1], train_p.shape[2]))
    train_p = train_p.reshape(train_feature.shape[0], K, train_p_i.shape[-1])

    return torch.from_numpy(train_feature).float().to(device), torch.from_numpy(test_feature).float().to(device), torch.from_numpy(train_p).float().to(device),\
        torch.from_numpy(s_array).float().to(device), train_label_list, test_label_list, seenclasses, unseenclasses


def gen_b(text_features, label_num, train_label_list, device='cpu'):
    # data = scio.loadmat('attribute300.mat')
    # feature = data['attribute300']
    bert_fea = st_scaler.fit_transform(text_features)
    # bert_fea = text_features
    train_b = np.array([])
    test_b = np.array([])
    # with open(fea_dir, 'rb') as f:
    #     bert_fea = pickle.load(f)
    i = 0
    for (key, value) in label_num.items():
        if key in train_label_list:
            if train_b.shape[0] == 0:
                train_b = np.tile(bert_fea[i, :], (value, 1))
            else:
                train_b = np.vstack((train_b, np.tile(bert_fea[i, :], (value, 1))))
        else:
            if test_b.shape[0] == 0:
                test_b = bert_fea[i, :]
            else:
                test_b = np.vstack((test_b, bert_fea[i, :]))
        i += 1

    return torch.from_numpy(train_b).float().to(device), torch.from_numpy(test_b).float().to(device)


def get_idx_classes(train_num, seenclasses, train_label):
    idxs_list = []
    for i in range(train_num):
        # idx_c = torch.nonzero(train_label == seenclasses[i].cpu()).cpu().numpy()
        # idx_c = np.squeeze(idx_c)

        idxs_list.append(np.where(train_label == seenclasses[i])[0])
    return idxs_list


def next_batch(train_num, seenclasses, batch_size, train_x, train_label, train_b, train_p, is_balance=True, device='cpu'):
    if is_balance:
        idx = []
        n_samples_class = max(batch_size // train_num, 1)
        # sampled_idx_c = np.random.choice(np.arange(train_num), min(train_num, batch_size), replace=False).tolist()
        sampled_idx_c = list(range(train_num))
        idxs_list = get_idx_classes(train_num, seenclasses, train_label)
        for i_c in sampled_idx_c:
            idxs = idxs_list[i_c]
            idx.append(np.random.choice(idxs, n_samples_class, replace=False))
        idx = np.concatenate(idx)
        idx = torch.from_numpy(idx)
    else:
        idx = torch.randperm(train_num)[0:batch_size]

    batch_feature = train_x[idx].to(device)
    batch_label = train_label[idx]
    batch_att_b = train_b[idx].to(device)
    batch_att_p = train_p[idx].to(device)
    return batch_label, batch_feature, batch_att_b, batch_att_p


if __name__ == '__main__':
    with open('wandb_config/MIT_CLIP_czsl.yaml', 'r') as file:
        # 将yaml文件内容转换为字典
        data = yaml.safe_load(file)
        config = Config(data)
    print('Config file from wandb:', config)
    train_num = 57
    batch_size = 57  # 57

    # visual features
    dataloader = DataLoader(fea_dir='./data/MIT/MIT_CLIP_L_67.pkl', img_dir='../image/MIT_67', device=config.device)
    train_x, test_x, train_p, s, train_label, test_label, seenclasses, unseenclasses = generate_x_p_s(
        train_num=train_num, dataloader_data=dataloader.data, random=False, device=config.device)

    # Semantic features
    train_b, test_b = gen_b(dataloader.data['text_features'], dataloader.data['label_num'], seenclasses, device=config.device)
    del dataloader

    # set random seed
    seed = config.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # DSRL model
    model = DSRL(config).to(config.device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.1, momentum=0.9)    # 0.00005>=0.0001
    # optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    # main loop
    # niters = train_x.size(0) * config.epochs // batch_size
    niters = 6000   #
    report_interval = 30  # niters // config.epochs
    best_performance_zsl = 0
    for i in range(0, niters):
        model.train()
        optimizer.zero_grad()

        batch_label, batch_feature, batch_att_b, batch_att_p = next_batch(train_num, seenclasses, batch_size, train_x, train_label, train_b, train_p, device=config.device)
        loss = model(batch_label, batch_feature, batch_att_b, batch_att_p)

        loss.backward()
        optimizer.step()
        # scheduler.step()

        # report test result
        if i % report_interval == 0:
            model.eval()
            with torch.no_grad():
                print('-' * 30)
                # get test_p
                sim = cosine_similarity(test_b.detach().cpu().numpy(), train_b.detach().cpu().numpy())
                sij = torch.FloatTensor(np.max(sim, axis=1)).to(config.device)
                test_p = torch.FloatTensor(test_b.size(0), train_p.size(1), train_p.size(2)).fill_(0).to(config.device)
                for ii in range(len(test_b)):
                    test_p[ii, :, :] = test_b[ii, :].repeat(train_p.size(1), 1) - sij[ii] * (train_b[np.argsort(sim, axis=1)[ii, -1], :].repeat(train_p.size(1), 1) - train_p[np.argsort(sim, axis=1)[ii, -1], :, :].squeeze())

                # test_b test_p --> test_a
                f = torch.cat((test_b.reshape(test_b.shape[0], 1, test_b.shape[1]), test_p), 1)
                f = model.layer_norm1(f)
                a, _ = model.mhatt(f, f, f)
                a = model.adaptive_pool(a)
                test_a = a.squeeze()
                # test_a = test_a + test_b
                # test_a = a.reshape(a.size(0), a.size(1) * a.size(2))
                # test_a = a.mean(dim=1).squeeze()

            acc_zsl = eval_zsl(test_a, test_x, test_label, unseenclasses, model, config.device)

            if acc_zsl > best_performance_zsl:
                best_performance_zsl = acc_zsl

            print('iter/epoch=%d/%d | loss=%.3f | acc_zsl=%.3f | best_performance_zsl=%.3f ' % (i, int(i // report_interval),
                      loss.item(), acc_zsl, best_performance_zsl))




