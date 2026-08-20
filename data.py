"""
데이터 파이프라인.

두 가지 모드:
  1) bin    : prepare_data.py 로 미리 토크나이즈한 uint16 memmap
              -> 세 regime 이 "완전히 동일한 데이터 순서"를 보게 됨 (권장)
  2) stream : HuggingFace streaming 을 그대로 packing
              -> 디스크 불필요하지만 순서 재현성이 약함 (worker/rank 수에 의존)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


class MemmapBlockDataset(Dataset):
    """uint16 토큰 스트림 -> (block_size,) x, y 쌍. 순서는 seed 로 고정된 permutation."""

    def __init__(self, bin_path, block_size, seed=1234, shuffle=True, max_blocks=None):
        self.bin_path = str(bin_path)
        self.block_size = int(block_size)

        data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        self.n_tokens = int(len(data))
        del data

        self.n_blocks = (self.n_tokens - 1) // self.block_size
        if self.n_blocks <= 0:
            raise ValueError(f"{bin_path}: 토큰 수가 block_size 보다 적습니다.")

        idx = np.arange(self.n_blocks, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        if max_blocks is not None:
            idx = idx[:max_blocks]
        self.index = idx
        self._data = None  # worker 별 lazy open

    def rotate(self, offset_blocks: int):
        """resume 용: 이미 소비한 block 수만큼 순서를 회전시킨다(결정론 유지)."""
        if len(self.index) == 0:
            return
        self.index = np.roll(self.index, -int(offset_blocks % len(self.index)))

    def _mm(self):
        if self._data is None:
            self._data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        return self._data

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        d = self._mm()
        s = int(self.index[i]) * self.block_size
        chunk = np.asarray(d[s: s + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class StreamingPackedDataset(IterableDataset):
    """HF streaming 데이터셋을 토크나이즈해서 block_size 단위로 packing."""

    def __init__(
        self,
        path="HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        tokenizer_name="gpt2",
        block_size=1024,
        rank=0,
        world_size=1,
        skip_docs=0,
        text_key="text",
        shuffle_buffer=0,
        seed=1234,
    ):
        self.path, self.name, self.split = path, name, split
        self.tokenizer_name = tokenizer_name
        self.block_size = block_size
        self.rank, self.world_size = rank, world_size
        self.skip_docs = skip_docs
        self.text_key = text_key
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

    def __iter__(self):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        wi = get_worker_info()
        nw = wi.num_workers if wi is not None else 1
        wid = wi.id if wi is not None else 0
        shard_id = self.rank * nw + wid
        num_shards = self.world_size * nw

        tok = AutoTokenizer.from_pretrained(self.tokenizer_name)
        tok.model_max_length = int(1e9)
        eos = tok.eos_token_id

        while True:  # 데이터가 끝나면 처음부터 반복
            ds = load_dataset(self.path, name=self.name, split=self.split, streaming=True)
            if self.shuffle_buffer:
                ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

            buf = []
            for i, ex in enumerate(ds):
                if i < self.skip_docs:
                    continue
                if (i - self.skip_docs) % num_shards != shard_id:
                    continue
                buf.extend(tok(ex[self.text_key])["input_ids"] + [eos])
                while len(buf) >= self.block_size + 1:
                    chunk = np.asarray(buf[: self.block_size + 1], dtype=np.int64)
                    buf = buf[self.block_size:]
                    yield torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def cycle(loader, sampler=None):
    """무한 반복 iterator. DistributedSampler(shuffle=False)면 순서는 항상 동일."""
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
