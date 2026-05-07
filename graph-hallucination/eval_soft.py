import os
os.environ["WANDB_MODE"] = "disabled"
os.environ['TMPDIR']='/egr/research-dselab/daixinna/temp'
os.environ["HF_HOME"] = '/egr/research-dselab/daixinna/shared/huggingface_path'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch.nn.functional as F
import json
import re
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import networkx as nx
from transformers import LlamaConfig, LlamaForCausalLM, DataCollatorForLanguageModeling
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import torch
from typing import Tuple, Set
from transformers import MixtralConfig, MixtralForCausalLM
from transformers import Qwen2Config, Qwen2ForCausalLM,LlamaModel
from peft import PeftModel

def check_hops(G,text,edge_list, failed_cases, lens, failed_path, results):   
        try:
            s = text.split('S ')[1].split()[0]
            e = text.split('E ')[1].split()[0]
            a_str = text.split("PATH")[1].split("END_P")[0].split()# .rstrip(',')
            # print(s, e, a_str)
            flag = True
            edge_marks = []  
            for i in range(len(a_str)-1):
                ans_pairs = (int(a_str[i]), int(a_str[i + 1]))
                if ans_pairs[0] == ans_pairs[1]:
                    flag = False
                    if ans_pairs not in failed_cases:
                        failed_cases[ans_pairs] = 0
                    failed_cases[ans_pairs] += 1
                dist = nx.shortest_path_length(G, source=int(ans_pairs[0]), target=int(ans_pairs[1]))
                if dist > 2:
                    flag = False
                    if ans_pairs not in failed_cases:
                        failed_cases[ans_pairs] = 0
                    failed_cases[ans_pairs] += 1
                else:
                    edge_marks.append(ans_pairs)
            
            # print(s, e)
            if a_str[0] != s or a_str[-1] != e:
                flag = False
                if 'NoEnd' not in failed_cases:
                    failed_cases['NoEnd'] = 0
                failed_cases['NoEnd'] += 1
            if flag:
                for e in edge_marks:
                    edge_list.add(ans_pairs)
                lens.append(len(a_str))
            else:
                failed_path.append(text)
            results.append(a_str)
            return flag, edge_list, failed_cases, lens, failed_path, results
        except:
            if 'NoPath' not in failed_cases:
                failed_cases['NoPath'] = 0
            failed_cases['NoPath'] += 1
            results.append(text)
            return False, edge_list, failed_cases, lens, failed_path, results
        

def process_one_checkpoint(checkpoint_folder: str,
                           base_model_path: str,
                           tokenizer_filename: str,
                           test_set,
                           supposed_avg_length,
                           device_id: int,
                           G,
                           finetune_path=None,
                           sft_dict = None) -> Tuple[str, float, int, Set[Tuple], Set[str]]:
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
    torch.cuda.set_device(device_id) if device.startswith("cuda") else None

    checkpoint_dir = checkpoint_folder
    # print(checkpoint_dir)
    if finetune_path is not None:
        epoch = os.path.basename(finetune_path).split('-')[-1]
    else:
        epoch = os.path.basename(checkpoint_dir).split('-')[1]

    print(os.listdir(base_model_path))
    tok_path = os.path.join(base_model_path, tokenizer_filename)
    print(tok_path)
    tokenizer_path = os.path.join(base_model_path, f"baby_tokenizer.json")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    print('Tokenizer loaded.')
    print('loading checkpoint dir',checkpoint_dir)
    if 'llama' in checkpoint_dir.lower():
        model = LlamaForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
        if finetune_path is not None:
            model = PeftModel.from_pretrained(model, os.path.join(finetune_path), is_trainable=False).to(device)
    if 'mixtral' in checkpoint_dir.lower():
        model = MixtralForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
    elif 'qwen' in checkpoint_dir.lower():   
        model = Qwen2ForCausalLM.from_pretrained(checkpoint_dir, local_files_only=True).to(device)
    print('Model loaded.')
    model.eval()

    lens = []
    acc = []
    edge_list = set()
    failed_path = []
    fails_case = {}
    results = []
    dec_results = []
    suppose_lens = sum(supposed_avg_length) / len(supposed_avg_length) if supposed_avg_length else 0.0
    with torch.inference_mode():
        for test in tqdm(test_set, total=len(test_set)):
            # print('test:', test)
            enc = tokenizer(test, return_tensors="pt", return_token_type_ids=False)
            enc = {k: v.to(device) for k, v in enc.items()}

            outputs = model.generate(
                **enc,
                max_new_tokens=max(supposed_avg_length)*2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            checks, edge_list, fails_case, lens, failed_path, results = check_hops(G, decoded, edge_list, fails_case, lens, failed_path, results)
            # print('fail case', fails_case)
            acc.append(checks)
            dec_results.append(decoded)

    acc_score = sum(acc) / len(acc) if acc else 0.0
    if finetune_path is not None:
        new_train_ratio = sft_dict['new_train_ratio']
        overlap_method = sft_dict['overlap_method']
        model_selected = sft_dict['model_selected']
        if 'ppo' in finetune_path:
            save_path = os.path.join(os.path.dirname(checkpoint_dir), f'{new_train_ratio}_{overlap_method}_{model_selected}_ppoepoch-{epoch}.pkl')
        elif 'finetune' in finetune_path:
            save_path = os.path.join(os.path.dirname(checkpoint_dir), f'{new_train_ratio}_{overlap_method}_{model_selected}_ftepoch-{epoch}.pkl')
    else:
        save_path = os.path.join(os.path.dirname(checkpoint_dir), f'epoch-{epoch}.pkl')
    
    with open(save_path, 'wb') as f:
        pickle.dump(dec_results, f)
        
    if len(lens) == 0:
        avg_lens = 0
        max_lens = 0
    else:
        avg_lens = sum(lens) / len(lens)
        max_lens = max(lens)


    print(acc_score)
    print('lens',max(lens),max(supposed_avg_length),'avg', avg_lens, suppose_lens)
    return save_path, acc_score, max_lens, edge_list, fails_case

def pick_checkpoint(output_dir, which=1):
    checkpoints = []
    epoch_steps = []
    for f in os.listdir(output_dir):
        if f.endswith('.pkl'):
            continue
        if 'lora' in f:
            continue
        if 'checkpoint' not in f:continue
        # epoch_steps.append(int(f.split('-')[-1]))
        full = os.path.join(output_dir, f)
        if os.path.isdir(full):
            checkpoints.append(full)
    # print(checkpoints)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    return checkpoints[which]

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    print('sets')
    n_layer = 6
    seed = 0
    hidden_size = 512

    n = 300
    if n < 500:
        hidden_size = 128

    method = 'path' 

    k_ratio = 0.1
    p_in = 0.1
    p_out = 0.01
    p = 0.2
    train_num_ratio = 1
    types = 'com' # er for intrinsic reasoning, comm for extrinsic reasoning
    
    
    overlap_method = 'none'  # 'none' or 'full' or 'partial'
    new_train_ratio = 1
    model_type = 'ori'
    model_selected = 6
    sft_dict = {'new_train_ratio':new_train_ratio,'overlap_method':overlap_method,'model_selected':model_selected}

    
    backbone_model = 'llama'  #'llama'  #'qwen'  #'mixtral'

    if types == 'com':

        base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_com'
        output_dir = f'/egr/research-dselab/shared/daixinna/graph_scaling/outputs_tmp_{seed}/{method}_{n}_{k_ratio}_{p_in}_{p_out}/{backbone_model}_{n_layer}_{hidden_size}_{train_num_ratio}/'
        print(n_layer, k_ratio, p_in, p_out)
        with open(os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}.pkl"),'rb') as f:
            G = pickle.load(f)
        base_model_path = os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}")
    elif types == 'er':
        base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_er'
        output_dir = f'/egr/research-dselab/shared/daixinna/graph_scaling/outputs_tmp_{seed}/{method}_{p}/{backbone_model}_{n_layer}_{hidden_size}_{train_num_ratio}/'
        with open(os.path.join(base_model_path,f"{n}_{p}.pkl"),'rb') as f:
            G = pickle.load(f)
        base_model_path = os.path.join(base_model_path,f"{n}_{p}")
        print(p)

    with open(os.path.join(f'{base_model_path}',f'{method}_soft_test.pkl'),'rb') as f:
        test = pickle.load(f)

    test_set = []
    test_answer = []
    supposed_avg_length = []
    for idx,p in tqdm(enumerate(test), total=len(test)):
        supposed_avg_length.append(len(p.split(' '))-7)
        # print(p)
        test_path = p.split('PATH')[0]+'PATH'
        if test_path not in test_set:
            test_answer.append(p)
            test_set.append(test_path)
        if model_type == 'sft' or model_type == 'ppo':
            if len(test_set)>=10000:
                break
        if len(test_set)>=500:
                break
    test_set = list(set(test_set))
    # test_set = test_set[:10000]
    print('test set size:', len(test_set))

    checkpoints = []
    for f in os.listdir(output_dir):
        if f.endswith('.pkl'):
            continue
        full = os.path.join(output_dir, f)
        if os.path.isdir(full):
            checkpoints.append(full)

    if not checkpoints:
        print("No checkpoint folders found.")
        raise SystemExit
    print(f"Found {len(checkpoints)} checkpoint folders.")
    print(checkpoints)

    
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    workers = max(1, n_gpus)

    futures = []
    tokenizer_filename = "baby_tokenizer.json"
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, ckpt in enumerate(sorted(checkpoints)):
            device_id = (i % n_gpus) if n_gpus > 0 else -1  # -1 表示 CPU
            fut = ex.submit(
                process_one_checkpoint,
                ckpt,
                base_model_path,
                tokenizer_filename,
                test_set,
                supposed_avg_length,
                device_id,
                G,
                finetune_path=None
            )
            futures.append(fut)

        for fut in as_completed(futures):
            try:
                save_path, acc_score, max_lens, edge_list, fails_case = fut.result()
                # tqdm.write(f"{os.path.basename(ckpt)} | acc={acc_score:.4f}")
                print(f"[DONE] {save_path} | acc={acc_score:.4f} | max_len={max_lens} | edges={len(edge_list)} | fails={len(fails_case)}")
            except Exception as e:
                print(f"[ERROR] {e}")
