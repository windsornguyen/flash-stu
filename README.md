# ⚡️ Flash STU ⚡️

<div align="center">
  <img src="docs/flash-stu.webp" alt="Flash STU Logo" width="720">
</div>

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Configuration](#configuration)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Introduction

This repository complements the [Flash STU: Fast Spectral Transform Units](https://arxiv.org/abs/2409.10489) paper and contains an optimized, open-source PyTorch implementation of the Spectral Transform Unit (STU) as proposed in [*Spectral State Space Models*](https://arxiv.org/abs/2312.06837) by Agarwal et al. (2024).

The [STU](stu.py) module is a fast and flexible building block that can be adapted into a wide range of neural network architectures, especially those that aim to solve tasks with long-range dependencies.

## Features

- ⚡️ Fast convolutions using [Flash FFT](https://github.com/HazyResearch/flash-fft-conv)
- 🚀 Fast, local attention using (sliding window) [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- 🌐 Support for distributed training using [DDP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [FSDP](https://pytorch.org/docs/stable/fsdp.html)

## Installation

> **Note**: CUDA is required to run code from this repository.

This repository was tested with:
- Python 3.12.5
- PyTorch 2.4.0
- Triton 3.0.0
- CUDA 12.4

and may be incompatible with other versions.

1. Install PyTorch with CUDA support:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    ```

2. Install Flash Attention:
   ```bash
   pip install ninja
   MAX_JOBS=4 pip install flash-attn --no-build-isolation
   ```

3. Install Flash FFT:
   ```bash
    pip install git+https://github.com/HazyResearch/flash-fft-conv.git#subdirectory=csrc/flashfftconv
    pip install git+https://github.com/HazyResearch/flash-fft-conv.git
    ```

4. Install the rest of the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

An example training script for pretraining a large language model is provided in [`example.py`](example.py).

If your compute cluster does not have internet access, you will need to pre-download the entire dataset before running the example script.

To download the dataset, run:
```bash
python data.py
```

We provide an LLM pretraining script, [`example.py`](example.py), for you to test out the repository. To begin training, run the following commands in your terminal:

```bash
chmod +x job.slurm
./job.slurm
```

If you are in a compute cluster that uses Slurm, you can submit a job using the following command:
```bash
sbatch job.slurm
```

Be sure to adjust the configurations of the [Slurm job](job.slurm) based on your cluster's constraints.

> **Note**: The FineWeb-Edu 10B-token sample is a relatively large dataset. It can be swapped out for something smaller, e.g. [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (476.6M tokens).


## Configuration

Training configurations can be adjusted as needed in [`config.json`](config.json).

> **Note**: PyTorch's `torch.compile` currently does not have great support for distributed wrapper modules like DDP or FSDP. If you encounter errors during training, try disabling `torch.compile`. For more information on `torch.compile`, see this [informal manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab).


## Contributing

Contributions are welcomed! Writing performant distributed code is always tricky. We welcome contributors to:

- Submit pull requests
- Report issues
- Help improve the project overall

## License

Apache 2.0 License

You can freely use, modify, and distribute the software, **even in proprietary products**, as long as you:
- Include proper attribution
- Include a copy of the license
- Mention any changes made

It also provides an express grant of patent rights from contributors.

See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Huge thanks to (in no particular order):
- Elad Hazan and the authors of the [Spectral State Space Models](https://arxiv.org/abs/2312.06837) paper
- Isabel Liu, Yagiz Devre, Evan Dogariu
- The Flash Attention team
- The Flash FFT team
- The PyTorch team
- Princeton Research Computing and Princeton Language and Intelligence, for supplying compute
- Andrej Karpathy, for his awesome [NanoGPT](https://github.com/karpathy/build-nanogpt) repository
