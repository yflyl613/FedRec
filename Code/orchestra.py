import torch
import numpy as np

from client import Client
from attacker import *
from agg import *


class Orchestra():
    def __init__(self, server_model, user_data, attacker_id_list, args, device):
        self.args = args
        self.device = device
        self.user_data = user_data
        self.attacker_id_list = attacker_id_list
        self.uid_list = list(self.user_data.keys())
        self.agg = eval(args.AGG_TYPE)(server_model, args, device)
        self.client = Client(self.args, self.device)
        self.rng = np.random.default_rng(self.args.SEED)
        if self.args.ATTACKER_STRAT is not None:
            self.attacker = eval(
                args.ATTACKER_STRAT)({_id: user_data[_id]
                                      for _id in self.attacker_id_list}, self.args, device)

    def update_one_round(self, step):
        total_client_loss, total_client_acc = [], []
        select_uid = self.rng.choice(self.uid_list, size=self.args.USER_SAMPLE_NUM,
                                     replace=False).tolist()

        if self.args.ATTACKER_STRAT is not None:
            num_attacker_one_round = len(
                [_id for _id in select_uid if _id in self.attacker_id_list])
        else:
            num_attacker_one_round = 0

        if num_attacker_one_round > 0:
            attacker_loss = self.attacker.prepare(self.agg.server_model, step)
        else:
            attacker_loss = None

        attacker_grad_norm, total_grad_norm = [], []

        for uid in select_uid:
            if num_attacker_one_round > 0 and uid in self.attacker_id_list:
                attacker_user_grad, attacker_item_grad, attacker_other_grad, \
                    attacker_sample_num = self.attacker.update(self.agg.server_model, uid)
                self.agg.collect_client_update(attacker_user_grad, attacker_item_grad,
                                               attacker_other_grad, attacker_sample_num, True)

                grad_norm = ((attacker_item_grad**2).sum() + (attacker_other_grad**2).sum()).sqrt()
                attacker_grad_norm.append(grad_norm)
                total_grad_norm.append(grad_norm)
            else:
                client_user_grad, client_item_grad, client_other_grad, client_sample_num, \
                    client_loss, client_acc = self.client.update(self.agg.server_model,
                                                                 uid,
                                                                 self.user_data[uid])
                self.agg.collect_client_update(client_user_grad, client_item_grad,
                                               client_other_grad, client_sample_num, False)
                total_client_loss.extend(client_loss)
                total_client_acc.extend(client_acc)
                total_grad_norm.append(
                    ((client_item_grad**2).sum() + (client_other_grad**2).sum()).sqrt())

        filter_stat = self.agg.agg()
        average_client_loss = sum(total_client_loss) / len(total_client_loss)
        average_client_acc = sum(total_client_acc) / len(total_client_acc)

        if num_attacker_one_round > 0:
            attacker_grad_norm = torch.stack(attacker_grad_norm, dim=0).mean()
        else:
            attacker_grad_norm = None
        total_grad_norm = torch.stack(total_grad_norm, dim=0).mean()

        return average_client_loss, average_client_acc, num_attacker_one_round, \
            attacker_grad_norm, total_grad_norm, filter_stat, attacker_loss
