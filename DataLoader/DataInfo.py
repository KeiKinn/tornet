import os
import pickle as p

labels = {
    'airport': 0,
    'shopping_mall': 1,
    'metro_station': 2,
    'street_pedestrian': 3,
    'public_square': 4,
    'street_traffic': 5,
    'tram': 6,
    'bus': 7,
    'metro': 8,
    'park': 9
}

machine = '/home/xinjing/Documents/gpu5'
seed_path = machine+'/nas/staff/data_work/Sure/DCASE2021/processed_data/seeds/train.pkl'

with open(seed_path, 'rb') as f:
    seeds = p.load(f)
    # seeds = sorted(seeds, key=lambda x: x[2])
for label in labels:
    labels[label] = [idx for idx, (_, _, lbl) in enumerate(seeds) if lbl == label]

pass