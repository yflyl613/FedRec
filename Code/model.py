import torch
import torch.nn as nn
import torch.nn.functional as F


class UserModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.user_embedding = nn.Embedding(args.NUM_USERS, args.EMBDIM)
        self.dropout = args.DROPOUT
        nn.init.xavier_uniform_(self.user_embedding.weight)

    def forward(self, uid):
        return F.dropout(self.user_embedding(uid),
                         p=self.dropout,
                         training=self.training,
                         inplace=True)


class ItemModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.item_embedding = nn.Embedding(args.NUM_ITEMS, args.EMBDIM)
        self.dropout = args.DROPOUT
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, iid):
        return F.dropout(self.item_embedding(iid),
                         p=self.dropout,
                         training=self.training,
                         inplace=True)


class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, user_vec, item_vec):
        '''
        user_vec: batch_size, emb_dim
        item_vec: batch_size, candidate_num, emb_dim
        '''
        score = torch.matmul(item_vec, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1)
        return score


def BPR_Loss(score):
    '''
        score: batch_size, 2    (#0:pos, #1:neg)
    '''
    pos_score = score[:, 0]
    neg_score = score[:, 1]
    loss = -((pos_score - neg_score).sigmoid() + 1e-8).log().mean()
    return loss


def L2_Regularizer(named_parameters):
    regularization_loss = 0
    for name, param in named_parameters:
        if 'bias' not in name:
            regularization_loss += torch.sum(param**2)
    return regularization_loss / 2


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.user_model = UserModel(args)
        self.item_model = ItemModel(args)
        self.predictor = Predictor(args)
        self.loss_fn = BPR_Loss
        self.weight_decay = args.WEIGHT_DECAY
        self.regularizer = L2_Regularizer

    def forward(self, uid, iid):
        """
        uid: batch_size
        iid: batch_size, 2  (#0: pos, #1: neg)
        """
        user_vec = self.user_model(uid)
        item_vec = self.item_model(iid)
        score = self.predictor(user_vec, item_vec)
        loss = self.loss_fn(score) + self.weight_decay * self.regularizer(self.named_parameters())
        return score, loss
