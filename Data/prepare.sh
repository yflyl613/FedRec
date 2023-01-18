#!/bin.sh

wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
wget https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
wget https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/user_list.txt
wget https://raw.githubusercontent.com/xiangwang1223/neural_graph_collaborative_filtering/master/Data/gowalla/item_list.txt
unzip ml-1m.zip && gzip -d loc-gowalla_totalCheckins.txt.gz
cd ../Code
python preprocess.py
echo "All data is ready"