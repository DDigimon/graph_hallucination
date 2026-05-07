import os
import json
import re
import pickle
from tqdm import tqdm
from bio_sync import person_name
data_id = 1

base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_com'

nodes_num = 500

method = 'path'

types = 'com'
k_ratio = 0.05
p_in = 0.3
p_out = 0.01
train_num_ratio = 1
p = 0.025
data_type = 'soft'
n_layer = 6
temperature = 0.3

with open(os.path.join(base_model_path,f"{nodes_num}_{k_ratio}_{p_in}_{p_out}.pkl"),'rb') as f:
    G = pickle.load(f)
base_model_path = os.path.join(base_model_path,f"{nodes_num}_{k_ratio}_{p_in}_{p_out}")
eval_data = os.path.join(base_model_path,f"test_language")

epoch_path_ratio = []
diverse_dicts = []
acc_list = []
lens_list = []
person_name = list(person_name.values())
for epoch_selected in range(50):

    save_path = os.path.join(eval_data,f'{epoch_selected}_{n_layer}_generated_results.json')
    eval_save_path = os.path.join(eval_data,f'{epoch_selected}_eval.json')
    test_data = []
    
    # print(person_name)
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_data.append(json.loads(line))
    def find_people_positions(text, people_list):
        results = []
        for idx, name in enumerate(people_list):
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, text) and idx not in results:
                results.append(idx)

        return results

    import networkx as nx

    dicts_set = set()
    def check(G, text, person_name):
        compression_length = 0

        answer = find_people_positions(text, person_name)
        flag = False
        avg_composed = []
        for i in range(1, len(answer)):
            dicts_set.add(answer[i])
            flag = True
        lens = len(answer)-1
        for i in range(len(answer)-1):
            if not G.has_edge(answer[i], answer[i+1]):
                compression_length = nx.shortest_path_length(G, source=answer[i], target=answer[i+1])
                # return False, compression_length
                flag = False
                avg_composed.append(compression_length)
            else:
                avg_composed.append(1)
        return flag,lens, len(answer)/sum(avg_composed) if len(avg_composed)!=0 else 0# len(answer)# compression_length

    acc = []
    compression_lengths = []
    lens_data = []
    for item in tqdm(test_data):
        # print(item)
        
        res,lens, compression_length = check(G, item['generated_answer'], person_name)
        acc.append(res)
        # if compression_length != 0:
        compression_lengths.append(compression_length)
        lens_data.append(lens)

    if len(compression_lengths)!=0:
        epoch_path_ratio.append(sum(compression_lengths)/len(compression_lengths))
    else:
        epoch_path_ratio.append(-1)
    diverse_dicts.append(len(dicts_set))
    acc_list.append(sum(acc)/len(acc))
    lens_list.append(sum(lens_data)/len(lens_data))

    print(epoch_selected)
    print("Accuracy:", sum(acc)/len(acc))
    print("Average Compression Length for Incorrect Answers:", epoch_path_ratio[-1])

print(epoch_path_ratio)
print(diverse_dicts)
print(acc_list)
print(lens_list)
