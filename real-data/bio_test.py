import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import (
    Qwen3Config,
    Qwen3ForCausalLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from datasets import load_from_disk
import pickle
import torch
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
base_model_path = os.path.join(base_model_path,f"{nodes_num}_{k_ratio}_{p_in}_{p_out}")
eval_data = os.path.join(base_model_path,f"test_language")

base_model_path = os.path.join(base_model_path, "train_language")
tokenizer_dir = os.path.join(base_model_path, "tokenizer_qwen_like")
ds_path = os.path.join(base_model_path, "pretrain_ds")
output_dir = os.path.join(base_model_path, f"qwen3_{n_layer}_{train_num_ratio}")


def pick_checkpoint(which=1):
    checkpoints = []
    epoch_steps = []
    for f in os.listdir(output_dir):
        if f.endswith('.pkl'):
            continue
        if 'lora_finetune' in f:
            continue
        # epoch_steps.append(int(f.split('-')[-1]))
        full = os.path.join(output_dir, f)
        if os.path.isdir(full):
            checkpoints.append(full)
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))
    print("Available checkpoints:")
    print(len(checkpoints))
    return checkpoints[which]

lens = int(1000/20)
temperature = 0.3
for epoch_selected in range(25,50):
    print(epoch_selected,'/',lens)
    checkpoint_dir = pick_checkpoint(which=epoch_selected)
    print(f"Loading checkpoint from: {checkpoint_dir}")
    # exit()



    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tokenizer
    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"))
    tok.pad_token = "<pad>"
    tok.bos_token = "<bos>"
    tok.eos_token = "<eos>"

    # model
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    model.config.pad_token_id = tok.pad_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.generation_config.bos_token_id = tok.bos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    def generate_text(prompt, max_new_tokens=100):
        enc = tok(prompt, return_tensors="pt")
        model_inputs = {
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc.get("attention_mask", None),
        }
        if model_inputs["attention_mask"] is not None:
            model_inputs["attention_mask"] = model_inputs["attention_mask"].to(device)

        with torch.no_grad():
            out = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )
        return tok.decode(out[0], skip_special_tokens=True)

    from bio_sync import person_name
    person_name = list(person_name.values())
    import json
    from tqdm import tqdm

    save_path = os.path.join(eval_data,f'{epoch_selected}_{n_layer}_generated_results.json')
    json_list = []
    for i in tqdm(range(len(person_name)), total=len(person_name)):
        
        tetxt = generate_text(person_name[i]+' ')
        data_json = {
            'generated_answer': tetxt,
            }
        json_list.append(data_json)

    with open(save_path,'w') as f:
        for item in json_list:
            f.write(json.dumps(item)+'\n')
        print(f'Saved generated results to {save_path}')
        