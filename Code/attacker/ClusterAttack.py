import math
import random
import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import grad_to_vector
from dataset import TrainDataset


class ClusterAttack():
    def __init__(self, attacker_user_data, args, device):
        self.attacker_user_data = attacker_user_data
        self.attacker_id_list = list(self.attacker_user_data.keys())
        self.args = args
        self.device = device
        self.seed_rng = np.random.default_rng(self.args.SEED)
        self.scale_gen = torch.Generator(device=self.device)
        self.num_cluster = args.NUM_CLUSTER
        self.decay_round = args.DECAY_ROUND
        self.inc_cnt, self.dec_cnt = 0, 0
        self.exp_avg, self.last_debiased_exp_avg = 0, None
        self.t = 0

    def prepare(self, server_model, step, *args):
        total_item_grad = []

        sample_attacker = random.sample(self.attacker_id_list,
                                        k=min(self.args.ATTACKER_SAMPLE_NUM,
                                              len(self.attacker_id_list)))
        for uid in sample_attacker:
            optimizer = optim.SGD(server_model.parameters(), lr=self.args.LR)

            client_data = self.attacker_user_data[uid]
            client_dataset = TrainDataset(uid, client_data)
            client_dataloader = DataLoader(client_dataset,
                                           shuffle=True,
                                           batch_size=self.args.BATCH_SIZE)
            client_sample_num = len(client_dataset)

            client_gradient_vec = 0

            for uid, iid in client_dataloader:
                uid = uid.to(self.device, non_blocking=True)
                iid = iid.to(self.device, non_blocking=True)

                _, bz_loss = server_model(uid, iid)
                optimizer.zero_grad()
                bz_loss.backward()

                batch_sample_num = len(uid)
                client_gradient_vec += grad_to_vector(server_model) * (batch_sample_num /
                                                                       client_sample_num)

            user_grad_param = self.args.NUM_USERS * self.args.EMBDIM
            item_grad_param = self.args.NUM_ITEMS * self.args.EMBDIM
            client_item_grad = client_gradient_vec[user_grad_param:user_grad_param +
                                                   item_grad_param].reshape(
                                                       self.args.NUM_ITEMS, self.args.EMBDIM)
            total_item_grad.append(client_item_grad)

        with torch.no_grad():
            total_item_grad = torch.cat(total_item_grad, dim=0)
            item_grad_norm = (total_item_grad**2).sum(dim=-1).sqrt()
            self.item_grad_norm_mean = item_grad_norm.mean()
            self.item_grad_norm_std = item_grad_norm.std()

        item_embedding = server_model.item_model.item_embedding.weight.detach()
        self.cluster_centroids, self.cluster_labels = kmeans(X=item_embedding,
                                                             num_clusters=self.num_cluster,
                                                             seed=self.seed_rng.integers(2**63),
                                                             init='random',
                                                             verbose=False,
                                                             initial_state=None)

        optimizer = optim.SGD(server_model.parameters(), lr=self.args.LR)

        item_embedding = server_model.item_model.item_embedding.weight

        total_distance_loss = ((item_embedding - self.cluster_centroids[self.cluster_labels]) **
                               2).sum() / self.num_cluster
        optimizer.zero_grad()
        total_distance_loss.backward()

        self.total_item_grad = server_model.item_model.item_embedding.weight.grad.clone().detach()
        self.item_grad_norm = (self.total_item_grad**2).sum(dim=-1, keepdim=True).sqrt()

        total_distance_loss = total_distance_loss.detach()
        new_exp_avg = self.args.DECAY_RATE * self.exp_avg + (
            1 - self.args.DECAY_RATE) * total_distance_loss
        self.t += 1
        new_debiased_exp_avg = new_exp_avg / (1 - self.args.DECAY_RATE**self.t)
        if self.last_debiased_exp_avg is not None:
            if new_debiased_exp_avg > self.last_debiased_exp_avg:
                self.inc_cnt += 1
            else:
                self.dec_cnt += 1
        self.exp_avg = new_exp_avg
        self.last_debiased_exp_avg = new_debiased_exp_avg
        if (self.inc_cnt - self.dec_cnt) >= self.decay_round:
            self.num_cluster = min(
                round(self.num_cluster + math.sqrt(self.args.MAX_CLUSTER - self.num_cluster)),
                self.args.MAX_CLUSTER)
            self.inc_cnt, self.dec_cnt = 0, 0
            self.exp_avg, self.last_debiased_exp_avg = 0, None
            self.t = 0
            self.decay_round = self.decay_round * (1 + self.args.CLUSTER_RATE)
        if (self.dec_cnt - self.inc_cnt) >= self.decay_round:
            self.num_cluster = max(
                round(self.num_cluster - math.sqrt(self.num_cluster - self.args.MIN_CLUSTER)),
                self.args.MIN_CLUSTER)
            self.inc_cnt, self.dec_cnt = 0, 0
            self.exp_avg, self.last_debiased_exp_avg = 0, None
            self.t = 0
            self.decay_round = self.decay_round * (1 + self.args.CLUSTER_RATE)
        return new_debiased_exp_avg, None, self.num_cluster

    def update(self, server_model, client_id):
        optimizer = optim.SGD(server_model.parameters(), lr=self.args.LR)

        client_data = self.attacker_user_data[client_id]
        client_dataset = TrainDataset(client_id, client_data)
        client_dataloader = DataLoader(client_dataset,
                                       shuffle=True,
                                       batch_size=self.args.BATCH_SIZE)
        client_sample_num = len(client_dataset)

        client_gradient_vec = 0

        for uid, iid in client_dataloader:
            uid = uid.to(self.device, non_blocking=True)
            iid = iid.to(self.device, non_blocking=True)

            _, bz_loss = server_model(uid, iid)
            optimizer.zero_grad()
            bz_loss.backward()

            batch_sample_num = len(uid)
            client_gradient_vec += grad_to_vector(server_model) * (batch_sample_num /
                                                                   client_sample_num)

        user_grad_param = self.args.NUM_USERS * self.args.EMBDIM
        item_grad_param = self.args.NUM_ITEMS * self.args.EMBDIM
        client_user_grad = client_gradient_vec[:user_grad_param].reshape(
            self.args.NUM_USERS, self.args.EMBDIM)
        user_grad_mask = torch.zeros(self.args.NUM_USERS, dtype=torch.float32).to(self.device)
        user_grad_mask[client_id] = 1.0
        client_user_grad = (client_user_grad * user_grad_mask.reshape(-1, 1)).reshape(-1)
        client_other_grad = client_gradient_vec[user_grad_param + item_grad_param:]

        with torch.no_grad():
            item_grad_norm_bound = self.item_grad_norm_mean + torch.rand(
                size=(self.args.NUM_ITEMS, 1), device=self.device,
                generator=self.scale_gen) * self.item_grad_norm_std * self.args.SCALE
            clip_coef = item_grad_norm_bound / (self.item_grad_norm + 1e-8)
            attacker_item_grad = (self.total_item_grad * clip_coef).reshape(-1)

        return client_user_grad, attacker_item_grad, client_other_grad, client_sample_num


@torch.no_grad()
def initialize(X, num_clusters, init, seed, initial_state=None):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param init: (str) how to initialize
    :param seed: (int) seed for kmeans
    :param initial_state: (torch.tensor) specified initial centroids
    :return: (torch.tensor) initial state
    """
    num_samples = len(X)
    rng = np.random.default_rng(seed)  # local rng
    if init == 'random':
        if initial_state is None:
            indices = rng.choice(range(num_samples), size=num_clusters, replace=False)
            initial_state = X[indices]
        else:
            num_centroids = len(initial_state)
            if num_centroids > num_clusters:
                indices = rng.choice(range(len(initial_state)), size=num_clusters, replace=False)
                initial_state = initial_state[indices]
            elif num_centroids < num_clusters:
                indices = rng.choice(range(num_samples),
                                     size=num_clusters - num_centroids,
                                     replace=False)
                initial_state = torch.cat([initial_state, X[indices]], dim=0)
        return initial_state
    elif init == 'kmeans++':
        first_center_idx = rng.choice(range(num_samples), size=1, replace=False)
        current_center = X[first_center_idx]
        while current_center.shape[0] < num_clusters:
            dist = pairwise_distance(X, current_center)
            closest_dist = dist.min(dim=-1).values
            select_prob = closest_dist / closest_dist.sum()
            next_center_idx = rng.choice(range(num_samples),
                                         size=1,
                                         replace=False,
                                         p=select_prob.cpu().numpy())
            current_center = torch.cat([current_center, X[next_center_idx]], dim=0)
        return current_center
    else:
        raise NotImplementedError


@torch.no_grad()
def kmeans(X,
           num_clusters,
           distance='euclidean',
           init='random',
           tol=1e-4,
           verbose=True,
           iter_limit=0,
           seed=None,
           initial_state=None):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean'] [default: 'euclidean']
    :param init: (str) how to initialize kmeans [options: 'random', 'kmeans++'] [default: 'random']
    :param tol: (float) threshold [default: 0.0001]
    :param tqdm_flag: (bool) turn logs on or off
    :param iter_limit: (int) hard limit for max number of iterations
    :param seed: (int) random seed
    :return: (cluster_centroids, labels)
    """
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    else:
        raise NotImplementedError

    # initialize
    if verbose:
        print('Initializing...')

    iteration = 0
    initial_state = initialize(X, num_clusters, init, seed, initial_state)

    if verbose:
        tqdm_meter = tqdm(desc='[running kmeans]')

    while True:
        dis = pairwise_distance_function(X, initial_state)
        choice_cluster = torch.argmin(dis, dim=-1, keepdim=True)
        initial_state_pre = initial_state.clone()

        for index in range(num_clusters):
            mask = (choice_cluster == index)
            num_cluster_samples = mask.sum()
            if num_cluster_samples == 0:
                initial_state[index] = initial_state_pre[index]
            else:
                initial_state[index] = (X * mask).sum(dim=0) / num_cluster_samples

        center_shift = ((initial_state - initial_state_pre)**2).sum().sqrt()

        # increment iteration
        iteration += 1

        # update tqdm meter
        if verbose:
            tqdm_meter.set_postfix(iteration=f'{iteration}',
                                   center_shift=f'{center_shift:0.8f}',
                                   tol=f'{tol:0.8f}')
            tqdm_meter.update()
        if center_shift < tol:
            break
        if iter_limit != 0 and iteration >= iter_limit:
            break

    return initial_state, choice_cluster.reshape(-1)


@torch.no_grad()
def pairwise_distance(data, cluster_center):
    '''
    data: N*M
    cluster_center: C*M
    return: N*C
    '''
    A = data.unsqueeze(dim=1)
    B = cluster_center.unsqueeze(dim=0)
    dis = ((A - B)**2).sum(dim=-1)
    return dis
