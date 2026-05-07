import os
os.environ["WANDB_MODE"] = "disabled"
os.environ['TMPDIR']='/egr/research-dselab/daixinna/temp'
os.environ["HF_HOME"] = '/egr/research-dselab/daixinna/shared/huggingface_path'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import pickle
import os
import json
import time
import networkx as nx
from tqdm import tqdm
from copy import deepcopy
from transformers import PreTrainedTokenizerFast, TrainerCallback
from torch.utils.data import Dataset
from datasets import Dataset, DatasetDict
import transformers, accelerate
import time
print("transformers version:", transformers.__version__)
print("accelerate version:", accelerate.__version__)

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
n_layer = 6

types = 'com'
print(types)
data_type = 'soft'
edge_ratio = 0
condition_random_sample = 1
max_length = 128
    
n = 500
backbone_model = 'qwen'
method = 'path'
if n < 500:
    hidden_size = 128
else:
    hidden_size = 512

lr = 5e-4
block_size = 128
dyn_save = False


k_ratio = 0.1
p_in = 0.1
p_out = 0.01
train_num_ratio = 1
p = 0.4


if 'condition' in data_type:
    print(condition_random_sample, train_num_ratio)
if types == 'com':
    base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_com/'
    with open(os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}.pkl"),'rb') as f:
        G = pickle.load(f)
    if  'condition' in data_type:
        output_dir = f'/egr/research-dselab/shared/daixinna/graph_scaling/outputs_tmp_{seed}/{method}_{data_type}_{n}_{k_ratio}_{p_in}_{p_out}/{backbone_model}_{n_layer}_{hidden_size}_{train_num_ratio}/'
    else:
        output_dir = f'/egr/research-dselab/shared/daixinna/graph_scaling/outputs_tmp_{seed}/{method}_{n}_{k_ratio}_{p_in}_{p_out}/{backbone_model}_{n_layer}_{hidden_size}_{train_num_ratio}/'
    base_model_path = os.path.join(base_model_path,f"{n}_{k_ratio}_{p_in}_{p_out}")
    
    
elif types == 'er':
    base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_er/'
    with open(os.path.join(base_model_path,f"{n}_{p}.pkl"),'rb') as f:
        G = pickle.load(f)
    if 'condition' in data_type:
        output_dir = f'/egr/research-dselab/shared/daixinna/graph_scaling/outputs_tmp_{seed}/{method}_{data_type}_{n}_{p}/{backbone_model}_{n_layer}_{hidden_size}_{train_num_ratio}/'
    else:
        output_dir = f'/egr/research-dselab/shared/daixinna/graph_scaling/outputs_tmp_{seed}/{method}_{n}_{p}/{backbone_model}_{n_layer}_{hidden_size}_{train_num_ratio}/'
    base_model_path = os.path.join(base_model_path,f"{n}_{p}")
    
if 'condition' in data_type:
    tokenizer_path = os.path.join(base_model_path, f"condition_baby_tokenizer.json")
    with open(os.path.join(f'{base_model_path}',f'condition_train.pkl'),'rb') as f:
        training_corpus = pickle.load(f)
else:
    tokenizer_path = os.path.join(base_model_path, f"baby_tokenizer.json")
    print(base_model_path)

    with open(os.path.join(f'{base_model_path}',f'{method}_soft_train.pkl'),'rb') as f:
        training_corpus = pickle.load(f)

train_num = len(training_corpus)
train_num = int(train_num*train_num_ratio)
training_corpus = training_corpus[:train_num]
valid_num = int(train_num*0.1)
valid_corpus = training_corpus[:valid_num]

print(training_corpus[:5])


tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
vocab_dict = tokenizer.get_vocab()

print('vocab size: ',len(set(vocab_dict)), tokenizer.vocab_size)

import numpy as np
from transformers import LlamaConfig, LlamaForCausalLM, DataCollatorForLanguageModeling
from transformers import MixtralConfig, MixtralForCausalLM
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers import Trainer, TrainingArguments

vocab_size = len(set(tokenizer.get_vocab()))

if backbone_model == 'llama':
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size*2.8),      # ≈2.8x
        num_hidden_layers=n_layer,
        num_attention_heads=16,      # head_dim=64
        num_key_value_heads=8,       # GQA
        max_position_embeddings=max_length,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        hidden_act="silu",           # SwiGLU 家族
        use_cache=False,             # 训练期关闭
    )

    model = LlamaForCausalLM(cfg)

elif backbone_model == 'mixtral':
    cfg = MixtralConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size*2.8),      # 4x
        num_hidden_layers=n_layer,
        num_attention_heads=16,      # head_dim=64
        num_key_value_heads=8,       # GQA
        max_position_embeddings=max_length,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        use_cache=False,    
        # 训练期关闭
        num_local_experts=8,           # 专家数量（按需可改小 8/16，节省显存）
        num_experts_per_tok=2,          # ✅ top-4 gating（默认Mixtral是2）
        router_jitter=0.0,              # 需要可加噪声稳路由
        router_aux_loss_coef=1e-2,      # 负载均衡loss系数（视训练稳定性可微调）
        router_typ="topk",              # top-k router
        output_router_logits=False, 
    )

    model = MixtralForCausalLM(cfg)

elif backbone_model == 'qwen':
    cfg = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=int(hidden_size*2.8),      # 4x
        num_hidden_layers=n_layer,
        num_attention_heads=16,      # head_dim=64
        num_key_value_heads=8,       # GQA
        max_position_embeddings=max_length,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        use_cache=False,    
        # 训练期关闭
    )

    model = Qwen2ForCausalLM(cfg)
    
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"≈ {total_params/1e6:.2f}M parameters")
print(f"≈ {total_params/1e9:.2f}B parameters")

iter_num = 0

loss_dicts={}
loss_dicts['train']={}
loss_dicts['val']={}

ds = DatasetDict({
    "train": Dataset.from_dict({"text": training_corpus}),
    "validation": Dataset.from_dict({"text": valid_corpus})
})

def tok_fn(batch):
    return tokenizer(batch["text"])

def group_texts(examples):
    concat = {k: sum(examples[k], []) for k in examples.keys()}
    total = len(concat["input_ids"]) // block_size * block_size
    result = {
        k: [t[i:i+block_size] for i in range(0, total, block_size)]
        for k, t in concat.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


import math

tokenized = ds.map(tok_fn, batched=True, remove_columns=["text"])
lm_ds = tokenized.map(group_texts, batched=True)

N = len(lm_ds["train"])
batch_size = 2048
if  'condition' in data_type:
    train_epoch = 42
    save_epoch = 3
    steps_per_epoch = int(save_epoch*(math.ceil(N / batch_size)))
    eval_steps = 100# steps_per_epoch 
else:
    train_epoch = 49
    save_epoch = 7
    steps_per_epoch = save_epoch*(math.ceil(N / batch_size))
    eval_steps = max(int(steps_per_epoch / 10), 10)




class PrintInfoCallback(TrainerCallback):
    def __init__(self, steps_per_epoch: int, output_dir: str, device: str = "cuda"):
        self.steps_per_epoch = steps_per_epoch
        self.output_dir = output_dir
        self.device = device
        self.train_losses = []  # list of dicts: {"step": int, "epoch": float|None, "loss": float}
        self.eval_losses  = []  # list of dicts: {"step": int, "epoch": float|None, "loss": float}
        self._last_seen_idx = 0

        self._last_train_loss = None
        self._last_eval_loss = None
    
    def save_losses(self, prefix="loss"):
        with open(os.path.join(self.output_dir, f"{prefix}_train.json"), "w") as f:
            json.dump(self.train_losses, f, indent=2)

        with open(os.path.join(self.output_dir, f"{prefix}_eval.json"), "w") as f:
            json.dump(self.eval_losses, f, indent=2)

    def _consume_new_logs(self, state):
        logs = state.log_history
        if not logs:
            return

        for i in range(self._last_seen_idx, len(logs)):
            item = logs[i]
            step = item.get("step", state.global_step)
            epoch = item.get("epoch", None)

            if "loss" in item or "train_loss" in item:
                loss = item.get("loss", item.get("train_loss"))
                if loss is not None:
                    self.train_losses.append({"step": step, "epoch": epoch, "loss": float(loss)})
                    self._last_train_loss = float(loss)

            # eval loss
            if "eval_loss" in item:
                ev = item.get("eval_loss")
                if ev is not None:
                    self.eval_losses.append({"step": step, "epoch": epoch, "loss": float(ev)})
                    self._last_eval_loss = float(ev)

        self._last_seen_idx = len(logs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._consume_new_logs(state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 0:
            return control

        self._consume_new_logs(state)

        if state.global_step % self.steps_per_epoch != 0:
            return control

        gpu_mem = 0.0
        if torch.cuda.is_available() and (self.device.startswith("cuda")):
            gpu_mem = torch.cuda.memory_allocated() / 1024**3

        print(self.steps_per_epoch, self.output_dir)
        msg = (
            f"[{time.strftime('%H:%M:%S')}] Step {state.global_step} "
            f"| train_loss: {self._last_train_loss if self._last_train_loss is not None else 'NA'} "
            f"| eval_loss: {self._last_eval_loss if self._last_eval_loss is not None else 'NA'} "
            f"| GPU: {gpu_mem:.2f} GB"
        )
        print(msg)
        return control

cb = PrintInfoCallback(steps_per_epoch=steps_per_epoch, output_dir=output_dir)



collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

start = time.perf_counter()


training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=train_epoch,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    learning_rate=lr,
    weight_decay=0.1,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,

    eval_strategy="steps",
    eval_steps=eval_steps,

    save_steps=steps_per_epoch,
    save_total_limit=None,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_ds["train"],
    eval_dataset=lm_ds["validation"],
    data_collator=collator,
    callbacks=[cb]
)

trainer.train()
end = time.perf_counter()
cb.save_losses()

all_train = cb.train_losses
all_eval  = cb.eval_losses

train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in trainer.state.log_history:
    if "loss" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    elif "train_loss" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["train_loss"])

    # eval loss
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

import json
loss_dict = {
    "train": list(zip(train_steps, train_losses)),
    "eval": list(zip(eval_steps, eval_losses)),
}
print(all_train)
print(all_eval)
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print(f"Training completed at {end_time}, total time: {(end - start)/60:.2f} minutes")
print(output_dir)
with open(f"{output_dir}/loss_all.json", "w") as f:
    json.dump((loss_dict,all_train,all_eval), f)