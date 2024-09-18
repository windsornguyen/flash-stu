import os
import multiprocessing as mp

import torch
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


"""Adapted from https://github.com/karpathy/build-nanogpt/blob/master/fineweb.py"""

# Configuration
local_dir = "data/fineweb-edu-10B"
remote_name = "sample-10BT"
shard_size = int(1e8)  # 100M tokens per shard

# Create the cache directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# Initialize the tokenizer
enc = tiktoken.get_encoding("o200k_base")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens = torch.tensor(tokens, dtype=torch.int32)
    return tokens

def write_datafile(file, tokens):
    torch.save(tokens, file + '.pt')

# Tokenize all documents and write output shards
nprocs = 8
with mp.Pool(nprocs) as pool:
    shard_idx = 0
    all_tokens_tensor = torch.empty((shard_size,), dtype=torch.int32)
    token_count = 0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):
        # If there is enough space in current shard for new tokens,
        if token_count + len(tokens) < shard_size:
            all_tokens_tensor[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit=" toks", desc=f"Shard {shard_idx}")
            progress_bar.update(len(tokens))
        else:
            # Else, write the current shard and start a new one
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb-edu-10B_{split}_{shard_idx:06d}")

            # Split the document into whatever fits in this shard, remainder goes to next one
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_tensor[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_tensor)
            shard_idx += 1
            progress_bar = None

            # Populate the next shard with the leftovers of the current doc
            all_tokens_tensor[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # Write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_idx == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb-edu-10B_{split}_{shard_idx:06d}")
        write_datafile(filename, all_tokens_tensor[:token_count])
