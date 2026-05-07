from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

import glob
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import os


def train_bpe_tokenizer(
    data_base_path,
    data_glob="data/*.txt",
    out_dir="tokenizer_qwen_like",
    vocab_size=50000,
    min_frequency=2,
):
    out_dir = os.path.join(data_base_path, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    files = glob.glob(data_glob)
    if not files:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    special_tokens = ["<pad>", "<unk>", "<bos>", "<eos>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.train(files, trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )

    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    print(f"Saved tokenizer to: {out_dir}/tokenizer.json")


def load_lines(files):
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield {"text": line}

def make_dataset(
    base_model_path,
    data_glob="data/*.txt",
    tokenizer_dir="tokenizer_qwen_like",
    out_dir="pretrain_ds",
    block_size=2048,
):
    files = glob.glob(data_glob)
    if not files:
        raise FileNotFoundError(f"No files matched: {data_glob}")
    out_dir = os.path.join(base_model_path, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tok = PreTrainedTokenizerFast(tokenizer_file=os.path.join(tokenizer_dir, "tokenizer.json"))
    tok.pad_token = "<pad>"
    tok.unk_token = "<unk>"
    tok.bos_token = "<bos>"
    tok.eos_token = "<eos>"

    ds = Dataset.from_generator(lambda: load_lines(files))

    def tokenize(batch):
        return tok(batch["text"], add_special_tokens=True, truncation=False)

    tokenized = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=4)

    # pack into fixed-length blocks
    def group_texts(examples):
        # concat
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // block_size) * block_size
        concatenated = concatenated[:total_length]
        input_ids = [concatenated[i:i+block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": input_ids}

    packed = tokenized.map(group_texts, batched=True, remove_columns=tokenized.column_names, num_proc=4)

    packed.save_to_disk(out_dir)
    print(f"Saved packed dataset to: {out_dir}")


if __name__ == "__main__":
    method = 'path'
    types = 'com'
    k_ratio = 0.05
    p_in = 0.3
    p_out = 0.01
    train_num_ratio = 1
    p = 0.025
    data_type = 'soft'
    nodes_num = 500
    
    type = 'doc'
    if type == 'doc':
        base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/doc'
        txt_data_path = base_model_path
    else:
        base_model_path=f'/egr/research-dselab/shared/daixinna/graph_scaling/sync_com'
        base_model_path = os.path.join(base_model_path,f"{nodes_num}_{k_ratio}_{p_in}_{p_out}")
        base_model_path = os.path.join(base_model_path,f"train_language")
        txt_data_path = os.path.join(base_model_path,f'data_txt')
    

    
    train_bpe_tokenizer(base_model_path, data_glob=os.path.join(txt_data_path,'*.txt'))
    make_dataset(base_model_path, data_glob=os.path.join(txt_data_path,'*.txt'), tokenizer_dir=os.path.join(base_model_path,'tokenizer_qwen_like'), out_dir='pretrain_ds', block_size=512)