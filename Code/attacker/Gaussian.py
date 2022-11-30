import random
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal

from dataset import TrainDataset
from utils import grad_to_vector


class Gaussian():
    def __init__(self, attacker_user_data, args, device):
        self.attacker_user_data = attacker_user_data
        self.attacker_id_list = list(self.attacker_user_data.keys())
        self.args = args
        self.device = device

    def prepare(self, server_model, *args):
        total_item_grad, total_other_grad = [], []

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
                                                   item_grad_param]
            client_other_grad = client_gradient_vec[user_grad_param + item_grad_param:]
            total_item_grad.append(client_item_grad)
            total_other_grad.append(client_other_grad)

        with torch.no_grad():
            total_other_grad = torch.stack(total_other_grad, dim=0)
            other_grad_mean = torch.mean(total_other_grad, dim=0)
            other_grad_std = torch.std(total_other_grad, dim=0)
            self.other_grad_sampler = Normal(loc=other_grad_mean, scale=other_grad_std)

            total_item_grad = torch.stack(total_item_grad, dim=0)
            item_grad_mean = torch.mean(total_item_grad, dim=0)
            item_grad_std = torch.std(total_item_grad, dim=0)
            self.item_grad_sampler = Normal(loc=item_grad_mean, scale=item_grad_std)

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
        client_user_grad = client_gradient_vec[:user_grad_param].reshape(
            self.args.NUM_USERS, self.args.EMBDIM)
        user_grad_mask = torch.zeros(self.args.NUM_USERS, dtype=torch.float32).to(self.device)
        user_grad_mask[client_id] = 1.0
        client_user_grad = (client_user_grad * user_grad_mask.reshape(-1, 1)).reshape(-1)
        client_item_grad = self.item_grad_sampler.sample()
        client_other_grad = self.other_grad_sampler.sample()

        return client_user_grad, client_item_grad, client_other_grad, client_sample_num
