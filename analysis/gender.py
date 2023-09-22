import os
import pickle
import sys
from genbit.genbit_metrics import GenBitMetrics
import json
from args import DATA_DIR

data = pickle.load(open(os.path.join(DATA_DIR, 'chapters.pkl'), 'rb'))
batch_len = 820000
keys = sorted(list(data.keys()))[batch_len * int(sys.argv[1]): batch_len * (int(sys.argv[1]) + 1)]
print(len(keys))
chapters = [y['label'] for x in keys for y in data[x]['chapters']]
print(len(chapters))
genbit_metrics_object = GenBitMetrics('en', context_window=5, distance_weight=0.95, percentile_cutoff=80)
genbit_metrics_object.add_data(chapters, tokenized=False)
metrics = genbit_metrics_object.get_metrics(output_statistics=True, output_word_list=True)
json.dump(metrics, open(os.path.join(DATA_DIR, 'gender.json'), 'w'))
print(metrics['percentage_of_male_gender_definition_words'])
print(metrics['percentage_of_female_gender_definition_words'])
print(metrics['percentage_of_non_binary_gender_definition_words'])
