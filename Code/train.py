# %%
import os
import random
import math
import numpy as np
import pickle
import torch
import argparse
from tqdm import tqdm

from orchestra import Orchestra
from model import Model

assert "1.7.1" in torch.__version__


# %%
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--EXP_NAME", type=str, default=None)
    parser.add_argument("--MODEL_TYPE", type=str, default="MF")
    parser.add_argument("--DATA", type=str, default="ml", choices=["ml", "gowalla"])
    parser.add_argument("--CACHE_DIR", type=str, default="../CacheData")
    parser.add_argument("--DROPOUT", type=float, default=0.2)
    parser.add_argument("--EMBDIM", type=int, default=64)
    parser.add_argument("--LR", type=float, default=1e-3)
    parser.add_argument("--WEIGHT_DECAY", type=float, default=1e-5)
    parser.add_argument("--BATCH_SIZE", type=int, default=512)
    parser.add_argument(
        "--AGG_TYPE",
        type=str,
        choices=["FedAdam", "MultiKrumUNION", "NormBoundUNION"],
        default="FedAdam",
    )
    parser.add_argument("--ATTACKER_RATIO", type=float, default=0)
    parser.add_argument(
        "--ATTACKER_STRAT",
        type=str,
        default=None,
        choices=["ClusterAttack"],
    )
    parser.add_argument("--USER_SAMPLE_NUM", type=int, default=50)
    parser.add_argument("--ATTACKER_SAMPLE_NUM", type=int, default=50)
    parser.add_argument("--NORM_BOUND", type=int, default=0.1)
    parser.add_argument("--MAX_ROUND", type=int, default=6000)
    parser.add_argument("--SAVE_ROUND", type=int, default=200)
    parser.add_argument("--LOG_ROUND", type=int, default=100)
    parser.add_argument("--NUM_CLUSTER", type=int, default=2)
    parser.add_argument("--MAX_CLUSTER", type=int, default=50)
    parser.add_argument("--MIN_CLUSTER", type=int, default=1)
    parser.add_argument("--SEED", type=int, default=0)
    parser.add_argument("--SCALE", type=float, default=3)
    parser.add_argument("--DECAY_ROUND", type=int, default=100)
    parser.add_argument("--DECAY_RATE", type=float, default=0.999)
    parser.add_argument("--ALPHA", type=float, default=1)
    parser.add_argument("--K", type=int, default=15)
    parser.add_argument("--AGG_SAMPLE_NUM", type=float, default=500)
    parser.add_argument("--GAP_SAMPLE", type=int, default=50)

    args = parser.parse_args()
    args.MODEL_DIR = f"../model_all/{args.EXP_NAME}/seed{args.SEED}"
    args.GAP_CACHE = "../CacheData/gap_cache.pkl"
    args.ATTACKER_PER_ROUND = math.ceil(args.USER_SAMPLE_NUM * args.ATTACKER_RATIO)
    if args.DATA == "ml":
        args.DATA_PATH = "../Data/ml-1m/ratings.dat"
    else:
        args.DATA_PATH = "../Data/gowalla_10core.tsv"
    return args


args = parse_args()
os.makedirs(args.MODEL_DIR, exist_ok=True)
os.makedirs(args.CACHE_DIR, exist_ok=True)

# %%
with open(args.DATA_PATH, "r", encoding="utf-8") as f:
    ratings = f.readlines()

print("Number of ratings:", len(ratings))
user_data = {}
uid_remap, iid_remap = {}, {}
for line in tqdm(ratings):
    if args.DATA == "ml":
        uid, iid, rate, timestamp = line.strip("\n").split("::")
        timestamp = int(timestamp)
    else:
        uid, iid, timestamp = line.strip("\n").split("\t")
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

for uid in user_data:
    user_data[uid] = sorted(user_data[uid], key=lambda x: x[-1])

# %%
full_attacker_list_path = os.path.join(
    args.CACHE_DIR, f"{args.DATA}-full-attacker-list.pkl"
)

if os.path.exists(full_attacker_list_path):
    with open(full_attacker_list_path, "rb") as f:
        full_attacker_list = pickle.load(f)
    print("Loading full attacker list from", full_attacker_list_path)
else:
    full_attacker_list = random.sample(
        range(args.NUM_USERS), k=int(0.05 * args.NUM_USERS)
    )
    with open(full_attacker_list_path, "wb") as f:
        pickle.dump(full_attacker_list, f)
    print("Dumping full attacker list to", full_attacker_list_path)

attacker_list_path = os.path.join(
    args.CACHE_DIR, f"{args.DATA}-attacker-list-{args.ATTACKER_RATIO}.pkl"
)
if os.path.exists(attacker_list_path):
    with open(attacker_list_path, "rb") as f:
        attacker_id_list = pickle.load(f)
    print("Loading attacker list from", attacker_list_path)
else:
    attacker_id_list = random.sample(
        full_attacker_list, k=int(args.ATTACKER_RATIO * args.NUM_USERS)
    )
    with open(attacker_list_path, "wb") as f:
        pickle.dump(attacker_id_list, f)
    print("Dumping attacker list to", attacker_list_path)

cache_path = os.path.join(args.CACHE_DIR, f"{args.DATA}-seed{args.SEED}.pkl")
if os.path.exists(cache_path):
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
        train_user_data = cache_data["train_user_data"]
        val_user_data = cache_data["val_user_data"]
        test_user_data = cache_data["test_user_data"]
    print("Loading cache data from", cache_path)
else:
    train_user_data = {}
    val_user_data = {"uid": [], "label": [], "mask": []}
    test_user_data = {"uid": [], "label": [], "mask": []}
    iid_set = set(range(args.NUM_ITEMS))

    for uid in tqdm(user_data):
        pos_iid = [x[0] for x in user_data[uid]]
        train_pos_iid = pos_iid[:-2]
        val_pos_iid = pos_iid[-2]
        test_pos_iid = pos_iid[-1]

        candidate_iid_list = list(iid_set - set(pos_iid))

        # train
        for iid in train_pos_iid:
            neg_iid = random.choice(candidate_iid_list)
            if uid not in train_user_data:
                train_user_data[uid] = [[iid, neg_iid]]  # pos, neg
            else:
                train_user_data[uid].append([iid, neg_iid])

        train_user_data[uid] = np.array(train_user_data[uid])

        if uid not in full_attacker_list:
            # val
            val_user_data["uid"].append(uid)
            val_user_data["label"].append(val_pos_iid)
            label_mask = np.zeros(args.NUM_ITEMS, dtype=np.bool)
            label_mask[pos_iid[:-2]] = True
            val_user_data["mask"].append(label_mask)
            # test
            test_user_data["uid"].append(uid)
            test_user_data["label"].append(test_pos_iid)
            label_mask = np.zeros(args.NUM_ITEMS, dtype=np.bool)
            label_mask[pos_iid[:-1]] = True
            test_user_data["mask"].append(label_mask)

    val_user_data["uid"] = np.array(val_user_data["uid"])
    val_user_data["label"] = np.array(val_user_data["label"])
    val_user_data["mask"] = np.array(val_user_data["mask"])
    test_user_data["uid"] = np.array(test_user_data["uid"])
    test_user_data["label"] = np.array(test_user_data["label"])
    test_user_data["mask"] = np.array(test_user_data["mask"])

    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "train_user_data": train_user_data,
                "val_user_data": val_user_data,
                "test_user_data": test_user_data,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print("Dumping cache data to", cache_path)

if args.ATTACKER_STRAT is None:
    attacker_id_list = []

print(f"{len(attacker_id_list)} attackers in total")
print(attacker_id_list)
print("Number of training data:", sum(len(train_user_data[x]) for x in train_user_data))
print("Number of validation data:", len(val_user_data["label"]))
print("Number of test data:", len(test_user_data["label"]))

# %%
random.seed(args.SEED)
np.random.seed(args.SEED)
torch.manual_seed(args.SEED)
torch.cuda.manual_seed(args.SEED)
torch.cuda.manual_seed_all(args.SEED)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
server_model = Model(args).to(device)
print(server_model)

# %%
orch = Orchestra(server_model, train_user_data, attacker_id_list, args, device)

# %%
for i in tqdm(range(args.MAX_ROUND)):
    (
        round_loss,
        round_acc,
        round_attacker,
        attacker_grad_norm,
        total_grad_norm,
        filter_stat,
        attacker_loss,
    ) = orch.update_one_round(i)

    if (i + 1) % args.LOG_ROUND == 0:
        print(
            "Round: {}, train_loss: {:.5f}, acc: {:.5f}".format(
                i + 1, round_loss, round_acc
            )
        )

    if (i + 1) % args.SAVE_ROUND == 0:
        ckpt_path = os.path.join(args.MODEL_DIR, f"round-{i + 1}.pt")
        torch.save(orch.agg.server_model.state_dict(), ckpt_path)
        print(f"Model saved to {ckpt_path}")
