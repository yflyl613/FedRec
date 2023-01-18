import math
import os
import torch
import pickle
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.cluster import KMeans
from utils import vector_to_grad
from attacker import kmeans


class FedAdam():
    def __init__(self, server_model, args, device):
        self.server_model = server_model
        self.server_optimizer = optim.Adam(self.server_model.parameters(), lr=args.LR)
        self.args = args
        self.device = device
        self._reinit()

    def _reinit(self):
        self.client_user_grad = 0
        self.client_item_grad = []
        self.client_other_grad = []
        self.client_sample_num = []
        self.attacker_list = []
        self.server_optimizer.zero_grad()

    @torch.no_grad()
    def collect_client_update(self, client_user_grad, client_item_grad, client_other_grad,
                              client_sample_num, is_attacker):
        # direct add for user embedding
        self.client_user_grad += client_user_grad
        self.client_item_grad.append(client_item_grad)
        self.client_other_grad.append(client_other_grad)
        self.client_sample_num.append(client_sample_num)
        self.attacker_list.append(is_attacker)

    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_weight = client_sample_num.float() / client_sample_num.sum()

        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        agg_client_item_grad = torch.matmul(client_weight, client_item_grad)

        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        agg_client_other_grad = torch.matmul(client_weight, client_other_grad)

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()


class UNION(FedAdam):
    def __init__(self, server_model, args, device):
        super().__init__(server_model, args, device)
        self.rng = np.random.default_rng(133)
        if not os.path.exists(args.GAP_CACHE):
            print('Preparing cache data for Gap Statistics')
            rng = np.random.default_rng()
            cache_data = {k: [] for k in range(2)}
            for k in range(1, 3):
                tmp = []
                for _ in tqdm(range(10000)):
                    random_samples = rng.uniform(low=0, high=1, size=(args.USER_SAMPLE_NUM, 1))
                    random_kmeans = KMeans(n_clusters=k, tol=1e-7).fit(random_samples)
                    tmp.append(random_kmeans.inertia_)
                tmp = np.log(np.array(tmp))
                cache_data[k] = tmp
            self.cache_data = cache_data
            with open(args.GAP_CACHE, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Dumping cache data to', args.GAP_CACHE)
        else:
            with open(args.GAP_CACHE, 'rb') as f:
                self.cache_data = pickle.load(f)
            print('Loading cache data from', args.GAP_CACHE)

    @torch.no_grad()
    def cal_uniformity(self, emb):
        '''
            emb: num_client, num_items, emb_dim
        '''
        num_client, num_items, _ = emb.shape
        cnt = num_items * (num_items - 1)
        pdist = torch.norm(emb.unsqueeze(dim=2) - emb.unsqueeze(dim=1), dim=-1, p=2)
        return pdist.reshape(num_client, -1).sum(dim=-1) / cnt

    @torch.no_grad()
    def agg(self):
        num_clients = len(self.attacker_list)
        sample_iid = self.rng.choice(self.args.NUM_ITEMS, self.args.AGG_SAMPLE_NUM)
        eps = 1e-8
        beta1, beta2 = (0.9, 0.999)

        current_item_emb = \
            self.server_model.item_model.item_embedding.weight.clone().detach()[sample_iid]
        total_item_emb = current_item_emb.unsqueeze(dim=0).expand(num_clients, -1, -1)
        total_item_grad = torch.stack(self.client_item_grad,
                                      dim=0).reshape(num_clients, self.args.NUM_ITEMS,
                                                     self.args.EMBDIM)[:, sample_iid]

        item_emb_param = self.server_model.item_model.item_embedding.weight
        item_emb_state = self.server_optimizer.state.get(item_emb_param, {})
        if len(item_emb_state) == 0:
            step = 0
            exp_avg = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
            exp_avg_sq = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
        else:
            step = item_emb_state['step']
            exp_avg = item_emb_state['exp_avg'][sample_iid]
            exp_avg_sq = item_emb_state['exp_avg_sq'][sample_iid]
        exp_avg = exp_avg.unsqueeze(dim=0).expand_as(total_item_grad)
        exp_avg_sq = exp_avg_sq.unsqueeze(dim=0).expand_as(total_item_grad)
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        exp_avg = exp_avg.mul(beta1).add(total_item_grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(total_item_grad,
                                                   total_item_grad,
                                                   value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        step_size = self.args.LR / bias_correction1
        total_item_emb = total_item_emb.addcdiv(exp_avg, denom, value=-step_size)

        total_uniformity = self.cal_uniformity(total_item_emb).reshape(-1, 1)
        more_than_one, two_cluster_labels = GapStatistics(total_uniformity, self.args, self.cache_data)

        cluster_idx = {
            0: torch.where(two_cluster_labels == 0)[0],
            1: torch.where(two_cluster_labels == 1)[0]
        }
        abnormal_idx = 0 if len(cluster_idx[0]) < len(cluster_idx[1]) else 1
        selected_idx = 1 - abnormal_idx

        if more_than_one or len(cluster_idx[abnormal_idx]) <= self.args.ATTACKER_PER_ROUND:
            filtered_clients = cluster_idx[abnormal_idx]
            selected_clients = cluster_idx[selected_idx]
        else:
            selected_clients = list(range(num_clients))
            filtered_clients = []

        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)

        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        selected_grad = client_grad[selected_clients]
        selected_client_sample_num = client_sample_num[selected_clients]
        selected_client_weight = \
            selected_client_sample_num.float() / selected_client_sample_num.sum()

        agg_client_grad = torch.matmul(selected_client_weight, selected_grad)
        agg_client_item_grad = agg_client_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_client_grad[client_item_grad.shape[1]:]

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()

        return filter_precision, filter_recall, len(filtered_clients)


class MultiKrumUNION(UNION):
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        num_clients = len(self.attacker_list)
        sample_iid = self.rng.choice(self.args.NUM_ITEMS, self.args.AGG_SAMPLE_NUM)
        eps = 1e-8
        beta1, beta2 = (0.9, 0.999)

        # UNION
        current_item_emb = \
            self.server_model.item_model.item_embedding.weight.clone().detach()[sample_iid]
        total_item_emb = current_item_emb.unsqueeze(dim=0).expand(num_clients, -1, -1)
        total_item_grad = torch.stack(self.client_item_grad,
                                      dim=0).reshape(num_clients, self.args.NUM_ITEMS,
                                                     self.args.EMBDIM)[:, sample_iid]

        item_emb_param = self.server_model.item_model.item_embedding.weight
        item_emb_state = self.server_optimizer.state.get(item_emb_param, {})
        if len(item_emb_state) == 0:
            step = 0
            exp_avg = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
            exp_avg_sq = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
        else:
            step = item_emb_state['step']
            exp_avg = item_emb_state['exp_avg'][sample_iid]
            exp_avg_sq = item_emb_state['exp_avg_sq'][sample_iid]
        exp_avg = exp_avg.unsqueeze(dim=0).expand_as(total_item_grad)
        exp_avg_sq = exp_avg_sq.unsqueeze(dim=0).expand_as(total_item_grad)
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        exp_avg = exp_avg.mul(beta1).add(total_item_grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(total_item_grad,
                                                   total_item_grad,
                                                   value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        step_size = self.args.LR / bias_correction1
        total_item_emb = total_item_emb.addcdiv(exp_avg, denom, value=-step_size)

        total_uniformity = self.cal_uniformity(total_item_emb).reshape(-1, 1)
        more_than_one, two_cluster_labels = GapStatistics(total_uniformity, self.args, self.cache_data)

        cluster_idx = {
            0: torch.where(two_cluster_labels == 0)[0],
            1: torch.where(two_cluster_labels == 1)[0]
        }
        abnormal_idx = 0 if len(cluster_idx[0]) < len(cluster_idx[1]) else 1

        if more_than_one or len(cluster_idx[abnormal_idx]) <= self.args.ATTACKER_PER_ROUND:
            ours_filtered_clients = cluster_idx[abnormal_idx].cpu().tolist()
        else:
            ours_filtered_clients = []

        # MultiKrum
        client_scores = torch.zeros((num_clients, num_clients), device=self.device)
        for i in range(num_clients):
            client_scores[i] = ((client_grad - client_grad[i])**2).sum(dim=-1)
            client_scores[i, i] = client_scores[i].max() + 1

        topk_client_scores = torch.topk(client_scores,
                                        k=num_clients - self.args.ATTACKER_PER_ROUND - 2,
                                        dim=-1,
                                        largest=False).values
        sum_client_scores = torch.sum(topk_client_scores, dim=-1)
        mk_filtered_clients = torch.topk(sum_client_scores,
                                         k=self.args.ATTACKER_PER_ROUND,
                                         largest=True).indices.cpu().tolist()

        filtered_clients = list(set(ours_filtered_clients) | set(mk_filtered_clients))
        selected_clients = [i for i in range(num_clients) if i not in filtered_clients]

        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)

        selected_grad = client_grad[selected_clients]
        selected_client_sample_num = client_sample_num[selected_clients]
        selected_client_weight = \
            selected_client_sample_num.float() / selected_client_sample_num.sum()

        agg_client_grad = torch.matmul(selected_client_weight, selected_grad)
        agg_client_item_grad = agg_client_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_client_grad[client_item_grad.shape[1]:]

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()

        return filter_precision, filter_recall, len(filtered_clients)


class NormBoundUNION(UNION):
    @torch.no_grad()
    def agg(self):
        client_sample_num = torch.tensor(self.client_sample_num).to(self.device)
        client_item_grad = torch.stack(self.client_item_grad, dim=0)
        client_other_grad = torch.stack(self.client_other_grad, dim=0)
        client_grad = torch.cat([client_item_grad, client_other_grad], dim=-1)
        num_clients = len(self.attacker_list)
        sample_iid = self.rng.choice(self.args.NUM_ITEMS, self.args.AGG_SAMPLE_NUM)
        eps = 1e-8
        beta1, beta2 = (0.9, 0.999)

        current_item_emb = \
            self.server_model.item_model.item_embedding.weight.clone().detach()[sample_iid]
        total_item_emb = current_item_emb.unsqueeze(dim=0).expand(num_clients, -1, -1)
        total_item_grad = torch.stack(self.client_item_grad,
                                      dim=0).reshape(num_clients, self.args.NUM_ITEMS,
                                                     self.args.EMBDIM)[:, sample_iid]

        item_emb_param = self.server_model.item_model.item_embedding.weight
        item_emb_state = self.server_optimizer.state.get(item_emb_param, {})
        if len(item_emb_state) == 0:
            step = 0
            exp_avg = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
            exp_avg_sq = torch.zeros_like(current_item_emb, memory_format=torch.preserve_format)
        else:
            step = item_emb_state['step']
            exp_avg = item_emb_state['exp_avg'][sample_iid]
            exp_avg_sq = item_emb_state['exp_avg_sq'][sample_iid]
        exp_avg = exp_avg.unsqueeze(dim=0).expand_as(total_item_grad)
        exp_avg_sq = exp_avg_sq.unsqueeze(dim=0).expand_as(total_item_grad)
        step += 1
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step
        exp_avg = exp_avg.mul(beta1).add(total_item_grad, alpha=1 - beta1)
        exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(total_item_grad,
                                                   total_item_grad,
                                                   value=1 - beta2)
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
        step_size = self.args.LR / bias_correction1
        total_item_emb = total_item_emb.addcdiv(exp_avg, denom, value=-step_size)

        total_uniformity = self.cal_uniformity(total_item_emb).reshape(-1, 1)
        more_than_one, two_cluster_labels = GapStatistics(total_uniformity, self.args, self.cache_data)

        cluster_idx = {
            0: torch.where(two_cluster_labels == 0)[0],
            1: torch.where(two_cluster_labels == 1)[0]
        }
        abnormal_idx = 0 if len(cluster_idx[0]) < len(cluster_idx[1]) else 1
        selected_idx = 1 - abnormal_idx

        if more_than_one or len(cluster_idx[abnormal_idx]) <= self.args.ATTACKER_PER_ROUND:
            filtered_clients = cluster_idx[abnormal_idx]
            selected_clients = cluster_idx[selected_idx]
        else:
            selected_clients = list(range(num_clients))
            filtered_clients = []

        attacker_list = torch.tensor(self.attacker_list).to(self.device)
        total_attacker_num = attacker_list.sum()
        filter_precision = attacker_list[filtered_clients].sum() / (len(filtered_clients) + 1e-12)
        filter_recall = attacker_list[filtered_clients].sum() / (total_attacker_num + 1e-12)

        selected_grad = client_grad[selected_clients]
        selected_grad_norm = (selected_grad**2).sum(dim=-1, keepdim=True).sqrt()
        clip_coef = torch.clamp(selected_grad_norm / self.args.NORM_BOUND, min=1.0)
        selected_grad /= clip_coef
        selected_client_sample_num = client_sample_num[selected_clients]
        selected_client_weight = \
            selected_client_sample_num.float() / selected_client_sample_num.sum()

        agg_client_grad = torch.matmul(selected_client_weight, selected_grad)
        agg_client_item_grad = agg_client_grad[:client_item_grad.shape[1]]
        agg_client_other_grad = agg_client_grad[client_item_grad.shape[1]:]

        vector_to_grad(self.client_user_grad, self.server_model.user_model)
        vector_to_grad(agg_client_item_grad, self.server_model.item_model)
        vector_to_grad(agg_client_other_grad, self.server_model.predictor)
        self.server_optimizer.step()
        self._reinit()

        return filter_precision, filter_recall, len(filtered_clients)


@torch.no_grad()
def GapStatistics(metrics, args, cache_data):
    rng = np.random.default_rng()
    low, high = metrics.min(), metrics.max()
    normalized_metrics = (metrics - low) / (high - low)
    gap, s = [], []
    for k in range(1, 3):
        cluster_centroids, cluster_labels = kmeans(X=normalized_metrics,
                                                   num_clusters=k,
                                                   init='kmeans++',
                                                   tol=1e-7,
                                                   verbose=False,
                                                   seed=1)
        if k == 2:
            two_cluster_labels = cluster_labels
        V_k = ((normalized_metrics - cluster_centroids[cluster_labels])**2).sum().cpu().numpy()
        V_kb = rng.choice(cache_data[k], size=args.GAP_SAMPLE, replace=False)
        gap_k = V_kb.mean() - np.log(V_k)
        V_k_std = V_kb.std() * np.sqrt((1 + args.GAP_SAMPLE) / args.GAP_SAMPLE)
        gap.append(gap_k)
        s.append(V_k_std)
    return gap[0] < gap[1] - s[1], two_cluster_labels
