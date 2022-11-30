import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TrainDataset, ContrastiveDataset
import utils
from utils import grad_to_vector


class Client:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.rng = random.Random(13)
        self.iid_set = set(range(args.NUM_ITEMS))

    def update(self, server_model, client_id, client_data):
        optimizer = optim.SGD(server_model.parameters(), lr=self.args.LR)

        client_dataset = TrainDataset(client_id, client_data)
        client_dataloader = DataLoader(client_dataset,
                                       shuffle=True,
                                       batch_size=self.args.BATCH_SIZE)
        client_sample_num = len(client_dataset)

        client_gradient_vec = 0
        client_loss, client_acc = [], []

        if 'UNION' in self.args.AGG_TYPE:
            pos_iids = client_data[:, 0].tolist()
            candidate_iid_list = list(self.iid_set - set(pos_iids))
            candidates, labels = [], []
            for idx in range(len(pos_iids)):
                label = self.rng.randint(0, self.args.K)
                pos_sample = self.rng.sample(pos_iids[:idx] + pos_iids[idx + 1:], k=1)
                neg_sample = self.rng.sample(candidate_iid_list, k=self.args.K)
                candidate = neg_sample[:label] + pos_sample + neg_sample[label:]
                candidates.append(candidate)
                labels.append(label)
            pos_iids = np.array(pos_iids).reshape(-1, 1)
            candidates = np.array(candidates)
            labels = np.array(labels)
            training_data = np.concatenate([pos_iids, candidates], axis=-1)
            contrastive_dataset = ContrastiveDataset(training_data, labels)
            contrastive_dataloader = DataLoader(contrastive_dataset,
                                                shuffle=True,
                                                batch_size=self.args.BATCH_SIZE)

            for (uid, iid), (contrastive_iid, label) in zip(client_dataloader,
                                                            contrastive_dataloader):
                uid = uid.to(self.device, non_blocking=True)
                iid = iid.to(self.device, non_blocking=True)
                contrastive_iid = contrastive_iid.to(self.device, non_blocking=True)
                label = label.to(self.device, non_blocking=True)
                total_item_emb = server_model.item_model(contrastive_iid)  # bz, 1+1+K, emb_dim
                anchor = total_item_emb[:, 0, :]  # bz, emb_dim
                candidate = total_item_emb[:, 1:, :]  # bz, 1+K, emb_dim
                scores = torch.bmm(candidate, anchor.unsqueeze(dim=-1)).squeeze(dim=-1)  # bz, 1+K
                contrastive_loss = F.cross_entropy(scores, label)

                y_hat, bz_loss = server_model(uid, iid)
                bz_acc = utils.acc(y_hat)
                total_loss = bz_loss + self.args.ALPHA * contrastive_loss
                optimizer.zero_grad()
                total_loss.backward()

                batch_sample_num = len(uid)
                client_gradient_vec += grad_to_vector(server_model) * (batch_sample_num /
                                                                       client_sample_num)

                client_loss.append(bz_loss)
                client_acc.append(bz_acc)
        else:
            for uid, iid in client_dataloader:
                uid = uid.to(self.device, non_blocking=True)
                iid = iid.to(self.device, non_blocking=True)

                y_hat, bz_loss = server_model(uid, iid)
                bz_acc = utils.acc(y_hat)
                optimizer.zero_grad()
                bz_loss.backward()

                batch_sample_num = len(uid)
                client_gradient_vec += grad_to_vector(server_model) * (batch_sample_num /
                                                                       client_sample_num)

                client_loss.append(bz_loss)
                client_acc.append(bz_acc)

        user_grad_param = self.args.NUM_USERS * self.args.EMBDIM
        item_grad_param = self.args.NUM_ITEMS * self.args.EMBDIM
        client_user_grad = client_gradient_vec[:user_grad_param].reshape(
            self.args.NUM_USERS, self.args.EMBDIM)
        user_grad_mask = torch.zeros(self.args.NUM_USERS, dtype=torch.float32).to(self.device)
        user_grad_mask[client_id] = 1.0
        client_user_grad = (client_user_grad * user_grad_mask.reshape(-1, 1)).reshape(-1)
        client_item_grad = client_gradient_vec[user_grad_param:user_grad_param + item_grad_param]
        client_other_grad = client_gradient_vec[user_grad_param + item_grad_param:]

        return client_user_grad, client_item_grad, client_other_grad, \
            client_sample_num, client_loss, client_acc
