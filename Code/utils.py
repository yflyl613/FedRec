import torch
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def acc(y_hat, label=None):
    '''
        if label is None:
            y_hat: batch_size, 2 (#0:pos, #1: neg)
        else:
            y_hat: batch_size, candidate_num
            label: batch_size
    '''
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_hat.shape[0]
    if label is None:
        hit = (y_hat == 0).sum()
    else:
        hit = (y_hat == label).sum()
    return hit.data.float() * 1.0 / tot


def hr_score(y_true, y_score, K=[5, 10, 20]):
    '''
    y_true: batch_size
    y_score: batch_size * candidate_num
    K: list of k
    '''
    order = torch.argsort(y_score, dim=-1, descending=True)
    HR = []
    for k in K:
        hr = (order[:, :k] == y_true.reshape(-1, 1)).any(-1).sum()
        HR.append(hr)
    return torch.tensor(HR)


def dcg_score(y_true, y_score, K=[5, 10, 20]):
    '''
    y_true: batch_size
    y_score: batch_size * candidate_num
    K: list of k
    '''
    order = torch.argsort(y_score, dim=-1, descending=True)
    DCG = []
    for k in K:
        hit = (order[:, :k] == y_true.reshape(-1, 1))
        gains = 2**hit - 1
        discounts = torch.log2(torch.arange(k, device=order.device) + 2.0)
        dcg = torch.sum(gains / discounts, dim=-1).sum()
        DCG.append(dcg)
    return torch.tensor(DCG)


def ndcg_score(y_true, y_score, K=[5, 10, 20]):
    '''
    y_true: batch_size
    y_score: batch_size * candidate_num
    K: list of k
    '''
    idcg = 1
    DCG = dcg_score(y_true, y_score, K)
    return DCG / idcg


@torch.no_grad()
def grad_to_vector(model):
    vec = []
    for param in model.parameters():
        vec.append(param.grad.detach().view(-1))
    return torch.cat(vec)


@torch.no_grad()
def vector_to_grad(vec, model):
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()
        pointer += num_param
