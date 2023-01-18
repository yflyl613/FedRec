# %%
import os
import pickle
import torch
import argparse
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import TestDataset
from utils import hr_score, ndcg_score
from model import Model

assert '1.7.1' in torch.__version__


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_NAME", type=str, default=None)
    parser.add_argument("--MODEL_TYPE", type=str, default='MF')
    parser.add_argument("--DATA", type=str, default='ml', choices=['ml', 'gowalla'])
    parser.add_argument("--CACHE_DIR", type=str, default="../CacheData")
    parser.add_argument("--DROPOUT", type=float, default=0.2)
    parser.add_argument("--EMBDIM", type=int, default=64)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=256)
    parser.add_argument("--MAX_ROUND", type=int, default=6000)
    parser.add_argument("--SAVE_ROUND", type=int, default=200)
    parser.add_argument("--SEED", type=int, default=0)

    args = parser.parse_args()
    args.MODEL_DIR = f"../model_all/{args.EXP_NAME}/seed{args.SEED}"
    if args.DATA == 'ml':
        args.DATA_PATH = '../Data/ml-1m/ratings.dat'
    else:
        args.DATA_PATH = '../Data/gowalla_10core.tsv'
    return args


args = parse_args()
os.makedirs(args.MODEL_DIR, exist_ok=True)
os.makedirs(args.CACHE_DIR, exist_ok=True)

# %%
with open(args.DATA_PATH, 'r', encoding='utf-8') as f:
    ratings = f.readlines()

print("Number of ratings:", len(ratings))
user_data = {}
uid_remap, iid_remap = {}, {}
for line in tqdm(ratings):
    if args.DATA == 'ml':
        uid, iid, rate, timestamp = line.strip('\n').split('::')
        timestamp = int(timestamp)
    else:
        uid, iid, timestamp = line.strip('\n').split('\t')
        timestamp = int(timestamp)

    if uid not in uid_remap:
        uid_remap[uid] = len(uid_remap)
    if iid not in iid_remap:
        iid_remap[iid] = len(iid_remap)
    uid = uid_remap[uid]
    iid = iid_remap[iid]

    if uid not in user_data:
        user_data[uid] = [(iid, timestamp)]
    else:
        user_data[uid].append((iid, timestamp))

args.NUM_USERS = len(uid_remap)
args.NUM_ITEMS = len(iid_remap)
print("Number of users:", args.NUM_USERS)
print("Number of items:", args.NUM_ITEMS)

# %%
cache_path = os.path.join(args.CACHE_DIR, f'{args.DATA}-seed{args.SEED}.pkl')
while not os.path.exists(cache_path):
    time.sleep(30)
with open(cache_path, 'rb') as f:
    cache_data = pickle.load(f)
    val_user_data = cache_data['val_user_data']
    test_user_data = cache_data['test_user_data']
print('Loading cache data from', cache_path)

print('Number of validation data:', len(val_user_data['label']))
print('Number of test data:', len(test_user_data['label']))

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
server_model = Model(args).to(device)

# %%
val_dataset = TestDataset(val_user_data)
val_dataloader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, num_workers=4)
test_dataset = TestDataset(test_user_data)
test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, num_workers=4)
total_val_num = len(val_dataset)
total_test_num = len(test_dataset)

# %%
checked_ckpt = []
total_num = args.MAX_ROUND // args.SAVE_ROUND
while len(checked_ckpt) < total_num:
    current_ckpt = os.listdir(args.MODEL_DIR)
    current_ckpt.sort(key=lambda x: int(x.split('.')[0].split('-')[-1]))
    if len(checked_ckpt) == len(current_ckpt):
        time.sleep(30)
    else:
        for ckpt in current_ckpt:
            if ckpt not in checked_ckpt:
                checked_ckpt.append(ckpt)
                idx = int(ckpt.split('.')[0].split('-')[-1])
                ckpt_path = os.path.join(args.MODEL_DIR, ckpt)
                print('Loading ckpt from', ckpt_path)
                server_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

                server_model.eval()
                torch.set_grad_enabled(False)

                total_item_embedding = server_model.item_model.item_embedding.weight.unsqueeze(
                    dim=0)

                # val
                total_HR, total_nDCG = 0, 0
                with torch.no_grad():
                    for uid, mask, label in tqdm(val_dataloader):
                        uid = uid.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        label = label.to(device, non_blocking=True)
                        user_embedding = server_model.user_model(uid)
                        bz_score = server_model.predictor(
                            user_embedding, total_item_embedding.expand(len(uid), -1, -1))
                        bz_score = bz_score * (~mask) + (bz_score.min(dim=-1, keepdim=True).values -
                                                         1) * mask
                        total_HR += hr_score(label, bz_score, K=[5, 10, 20])
                        total_nDCG += ndcg_score(label, bz_score, K=[5, 10, 20])

                HR5, HR10, HR20 = (total_HR / total_val_num).cpu().tolist()
                nDCG5, nDCG10, nDCG20 = (total_nDCG / total_val_num).cpu().tolist()

                print(
                    '[Val] Round: {}, HR@5: {:.5f}, nDCG@5: {:.5f}, HR@10: {:.5f}, nDCG@10: {:.5f}, HR@20: {:.5f}, nDCG@20: {:.5f}'
                    .format(idx, HR5, nDCG5, HR10, nDCG10, HR20, nDCG20))

                # test
                total_HR, total_nDCG = 0, 0
                with torch.no_grad():
                    for uid, mask, label in tqdm(test_dataloader):
                        uid = uid.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        label = label.to(device, non_blocking=True)
                        user_embedding = server_model.user_model(uid)
                        bz_score = server_model.predictor(
                            user_embedding, total_item_embedding.expand(len(uid), -1, -1))
                        bz_score = bz_score * (~mask) + (bz_score.min(dim=-1, keepdim=True).values -
                                                         1) * mask
                        total_HR += hr_score(label, bz_score, K=[5, 10, 20])
                        total_nDCG += ndcg_score(label, bz_score, K=[5, 10, 20])

                HR5, HR10, HR20 = (total_HR / total_test_num).cpu().tolist()
                nDCG5, nDCG10, nDCG20 = (total_nDCG / total_test_num).cpu().tolist()

                print(
                    '[Test] Round: {}, HR@5: {:.5f}, nDCG@5: {:.5f}, HR@10: {:.5f}, nDCG@10: {:.5f}, HR@20: {:.5f}, nDCG@20: {:.5f}'
                    .format(idx, HR5, nDCG5, HR10, nDCG10, HR20, nDCG20))
