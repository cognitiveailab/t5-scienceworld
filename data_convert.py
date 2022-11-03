# Example usage: python data_convert.py data/goldpaths-all bc
import json
import sys
import os

assert(len(sys.argv)==3)

data_dir = sys.argv[1]
mode = sys.argv[2]
assert(mode in ['bc', 'dt'])
raw_data_list = []

for filename in os.listdir(data_dir):
    with open(os.path.join(data_dir, filename), 'r') as f:
        raw_data_list.append(json.load(f))

train_data = []
val_data = []
test_data = []

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

for raw_data in raw_data_list:
    for task_id in raw_data.keys():
        curr_task = raw_data[task_id]
        for seq_sample in curr_task['goldActionSequences']:
            task_desc = seq_sample['taskDescription']
            steps = seq_sample['path']
            if len(steps) < 2:
                continue
            fold = seq_sample['fold']
            obs = steps[0]['observation']
            action = steps[0]['action']

            for i in range(len(steps)-1):
                curr_step = steps[i]
                next_step = steps[i+1]
                score = curr_step['score']
                returns_to_go = 1.0 - float(score)
                if i != 0:
                    prev_step = steps[i-1]

                    prev_action = curr_step['action']
                    curr_action = next_step['action']
                    prev_obs = prev_step['observation']
                    curr_obs = curr_step['observation']
                    look = curr_step['freelook']
                    inventory = curr_step['inventory']

                    if mode == 'bc':
                        input_str = task_desc + ' </s> ' + curr_obs + ' ' + inventory + ' ' + look + ' </s> <extra_id_0>'\
                            + ' </s> ' + prev_action + ' </s> ' + prev_obs + ' </s>'
                    else:
                        input_str = task_desc + ' </s> ' + str(returns_to_go) + ' </s> ' + curr_obs + ' ' + inventory + ' ' + look\
                            + ' </s> <extra_id_0>' + ' </s> ' + prev_action + ' </s> ' + prev_obs + ' </s>'
                    label = "<extra_id_0> " + curr_action + ' <extra_id_1>'


                else:
                    curr_action = next_step['action']
                    curr_obs = curr_step['observation']
                    look = curr_step['freelook']
                    inventory = curr_step['inventory']
                    if mode == 'bc':
                        input_str = task_desc + ' </s> ' + curr_obs + ' ' + inventory + ' ' + look + ' </s> <extra_id_0>'\
                            + ' </s>' + ' </s> ' + '</s>'
                    else:
                        input_str = task_desc + ' </s> ' + str(returns_to_go) + ' </s> ' + curr_obs + ' ' + inventory + ' '\
                            + look + ' </s> <extra_id_0>' + ' </s>' + ' </s> ' + '</s>'
                    label = "<extra_id_0> " + curr_action + ' <extra_id_1>'

                curr_dat = {'input': clean(input_str), 'target': clean(label)}

                if fold == 'train':
                    train_data.append(curr_dat)
                elif fold == 'dev':
                    val_data.append(curr_dat)
                elif fold == 'test':
                    test_data.append(curr_dat)


with open(f"data/{mode}/sciworld_formatted_train.jsonl", 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + "\n")

with open(f"data/{mode}/sciworld_formatted_val.jsonl", 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + "\n")

with open(f"data/{mode}/sciworld_formatted_test.jsonl", 'w') as f:
    for item in test_data:
        f.write(json.dumps(item) + "\n")
