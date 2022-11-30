# FedRec
The source code and data for our paper "Untargeted Attack against Federated Recommendation Systems via Poisonous Item Embeddings and the Defense" in AAAI 2023.

## Requirements
- PyTorch == 1.7.1
- pickle
- tqdm

## Get Started
- **Prepare Data**

  You can prepare all the data by running the command `bash prepare.sh` under `Data/`. The script will download and preprocess the MovieLens-1M and Gowalla datasets for experiments.

- **Run Experiments**

  `Code/run.sh` is the script for running experiments.

  For training, you can run the command `bash run.sh train` under `Code/`. You can modify the value of hyper-parameters to change the setting of experiments. For example, by setting `ATTACKER_STRAT` as `ClusterAttack`, `ATTACKER_RATIO` as `0.01`, and `AGG_TYPE` as `NormBoundUNION`, you can train the Federated Recommendation system with our *MultiKrum+UNION* defense mechanism against our *ClusterAttack*. Please refer to `Code/train.py` for more options.

  For testing, you can run `bash run.sh test` under `Code/`.