import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TrainDataset
from utils import grad_to_vector


class FedAttack():
    def __init__(self, attacker_user_data, args, device):
        self.attacker_user_data = attacker_user_data
        self.attacker_id_list = list(self.attacker_user_data.keys())
        self.args = args
        self.device = device

    def prepare(self, *args):
        pass

    def update(self, server_model, client_id):
        optimizer = optim.SGD(server_model.parameters(), lr=self.args.LR)

        client_data = self.attacker_user_data[client_id]
        client_sample_num = len(client_data)
        with torch.no_grad():
            user_emb = server_model.user_model.user_embedding.weight[client_id]
            sim_score = torch.matmul(server_model.item_model.item_embedding.weight, user_emb)
            sim_iid_list = torch.argsort(sim_score,
                                         descending=False).detach().cpu().numpy().tolist()

        pos_iid = sim_iid_list[0]
        neg_iid_list = sim_iid_list[-client_sample_num:]
        random.shuffle(neg_iid_list)

        client_data = np.array([[pos_iid, neg] for neg in neg_iid_list])

        client_dataset = TrainDataset(client_id, client_data)
        client_dataloader = DataLoader(client_dataset,
                                       shuffle=True,
                                       batch_size=self.args.BATCH_SIZE)

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
        client_item_grad = client_gradient_vec[user_grad_param:user_grad_param + item_grad_param]
        client_other_grad = client_gradient_vec[user_grad_param + item_grad_param:]

        return client_user_grad, client_item_grad, client_other_grad, client_sample_num
