import logging
import os

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_tokens(filename):
    try:
        ext = os.path.splitext(filename)[1]
        
        if ext == ".npy":
            npt = np.load(filename)
            npt = npt.astype(np.int32)
            ptt = torch.tensor(npt, dtype=torch.long)
            return ptt
        elif ext == ".pt":
            return torch.load(filename, weights_only=True)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
    
    except Exception as e:
        logger.error(f"Error loading file {filename}: {str(e)}")
        raise

class DistributedDataloader:
    def __init__(
        self,
        bsz: int,
        seq_len: int,
        rank: int,
        world_size: int,
        dataset: str,
        split: str,
        main_process: bool = False,
    ):
        self.bsz = bsz
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        assert split in {'train', 'val', 'test'}, f"Invalid split: {split}"

        data_root = dataset
        shards = [s for s in os.listdir(data_root) if split in s and (s.endswith('.pt') or s.endswith('.npy'))]
        self.shards = [os.path.join(data_root, s) for s in sorted(shards)]
        assert len(self.shards) > 0, f'No shards found for split {split}'
        if main_process:
            logger.info(f'Found {len(self.shards)} shards for split {split}')

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = (self.bsz * self.seq_len * self.rank) % len(self.tokens)

    def set_epoch(self, epoch):
        self.generator = torch.Generator()
        self.generator.manual_seed(epoch)
        self.shard_order = torch.randperm(len(self.shards), generator=self.generator).tolist()
        self.current_shard = self.shard_order[self.rank % len(self.shards)]
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = (self.bsz * self.seq_len * self.rank) % len(self.tokens)

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_position + (self.bsz * self.seq_len + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = (self.bsz * self.seq_len * self.rank) % len(self.tokens)

        buf = self.tokens[self.current_position : self.current_position + self.bsz * self.seq_len + 1]
        x = buf[:-1].view(self.bsz, self.seq_len)
        y = buf[1:].view(self.bsz, self.seq_len)
        
        self.current_position += self.bsz * self.seq_len * self.world_size
        return x, y.to(torch.long)
