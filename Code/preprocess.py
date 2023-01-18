import datetime
from tqdm import tqdm

with open('../Data/loc-gowalla_totalCheckins.txt') as f:
    lines = f.readlines()[1:]
with open('../Data/item_list.txt') as f:
    iid_map = f.readlines()[1:]
with open('../Data/user_list.txt') as f:
    uid_map = f.readlines()[1:]

uid_remap, iid_remap = {}, {}
for line in iid_map:
    org_id, remap_id = line.strip('\n').split(' ')
    iid_remap[org_id] = remap_id
for line in uid_map:
    org_id, remap_id = line.strip('\n').split(' ')
    uid_remap[org_id] = remap_id

filter_lines = []
for line in tqdm(lines):
    uid, ts, _, _, iid = line.strip('\n').split('\t')
    if uid not in uid_remap or iid not in iid_remap:
        continue
    timeobj = datetime.datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
    filter_lines.append((uid, iid, int(timeobj.timestamp())))

with open('../Data/gowalla_10core.tsv', 'w') as f:
    for uid, iid, ts in filter_lines:
        f.write('\t'.join([uid, iid, str(ts)]) + '\n')
