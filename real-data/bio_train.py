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

k_ratio = 0.05
p_in = 0.3
p_out = 0.01
nodes_num = 500
n_layer = 4
train_num_ratio = 1

types = 'sync'
if types == 'doc':
    base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/doc'
    txt_data_path = base_model_path
else:
    base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_com'
    base_model_path = os.path.join(base_model_path,f"{nodes_num}_{k_ratio}_{p_in}_{p_out}")
    base_model_path = os.path.join(base_model_path,f"train_language")
    txt_data_path = os.path.join(base_model_path,f'data_txt')

tokenizer_dir = os.path.join(base_model_path, "tokenizer_qwen_like")
ds_path = os.path.join(base_model_path, "pretrain_ds")
output_dir = os.path.join(base_model_path, f"qwen3_{n_layer}_{train_num_ratio}")
print(output_dir)
# exit()

# -------------------------
# load dataset + split
# -------------------------
ds_all = load_from_disk(ds_path)
val_ratio = 0.005  # 0.5%
split = ds_all.train_test_split(test_size=val_ratio, seed=42)
train_ds = split["train"]
val_ds = split["test"]

# -------------------------
# tokenizer
# -------------------------
tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"))
tok.pad_token = "<pad>"
tok.bos_token = "<bos>"
tok.eos_token = "<eos>"

# -------------------------
# model config
# -------------------------
config = Qwen3Config(
    vocab_size=tok.vocab_size,
    max_position_embeddings=1024,
    hidden_size=1024,
    intermediate_size=2816,
    num_hidden_layers=n_layer,
    num_attention_heads=16,
    num_key_value_heads=8,   # ✅ 16 % 8 == 0
)

model = Qwen3ForCausalLM(config)

model.config.pad_token_id = tok.pad_token_id
model.config.bos_token_id = tok.bos_token_id
model.config.eos_token_id = tok.eos_token_id
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.bos_token_id = tok.bos_token_id
model.generation_config.eos_token_id = tok.eos_token_id

model.config.use_cache = False
model.gradient_checkpointing_enable()

collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# -------------------------
# callback: dump train/eval loss
# -------------------------
class LossLoggerCallback(TrainerCallback):
    def __init__(self, out_file: str):
        self.out_file = out_file
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        if os.path.exists(out_file):
            os.remove(out_file)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # logs 里可能有: loss / learning_rate / grad_norm / epoch / eval_loss ...
        record = {"step": state.global_step, **logs}
        with open(self.out_file, "a", encoding="utf-8") as f:
            f.write(str(record) + "\n")

loss_log_path = os.path.join(output_dir, "train_val_loss.log")

# -------------------------
# training args: eval + logging
# -------------------------
if types == 'doc':
    max_steps = 1000
    eval_step = 20
else:
    max_steps = 1000
    eval_step = 20
args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=3e-4,
    warmup_steps=10,
    max_steps=max_steps,
    bf16=True,

    save_steps=eval_step,
    logging_steps=10,

    eval_strategy="steps",    
    eval_steps=eval_step,     
    do_eval=True,

    logging_first_step=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    tokenizer=tok,
    callbacks=[LossLoggerCallback(loss_log_path)],
)

trainer.train()

trainer.save_model(args.output_dir)
tok.save_pretrained(args.output_dir)

train_losses = [(x["step"], x["loss"]) for x in trainer.state.log_history if "loss" in x]
eval_losses  = [(x["step"], x["eval_loss"]) for x in trainer.state.log_history if "eval_loss" in x]
print(train_losses)
print(eval_losses)
print("num train loss points:", len(train_losses))
print("num eval loss points:", len(eval_losses))
print("loss log saved to:", loss_log_path)
