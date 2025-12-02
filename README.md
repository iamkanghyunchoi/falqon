# falqon
Official repository of paper [FALQON: Accelerating LoRA Fine-tuning with Low-Bit Floating-Point Arithmetic, NeurIPS 2025]

## License
This codebase is provided under the MIT license, following the original QLoRA repository.

## Installation

Our implementation relies on customized versions of the following libraries: **transformers**, **torchao**, **accelerate**, **peft**, and **lm-eval-harness**. To set up these modified libraries, install each library in editable mode (`pip install -e .`) from their respective directories. Detailed dependencies are listed in the provided `requirements.txt`.
Also, we removed directory 'eval/' of the original QLoRA repo, due to file size constraint. 

First, create and activate a conda environment:

```bash
conda create -n <ENV_NAME> python=3.11.10
conda activate <ENV_NAME>
```

Next, install all required dependencies by running:

```bash
./install.sh
```

## How to Run

Fine-tuning scripts are located under the `scripts/` directory. You can execute the scripts using the following format:

```bash
./scripts/finetune_7b.sh <QUANT_TYPE> <BATCH_SIZE> <LORA_RANK> <DATASET> <NUM_STEPS> <SEED> <LEARNING_RATE> <TOP_K>
```

- `<QUANT_TYPE>`: Choose among `falqon` (our method), `nf4` (QLoRA), `fp8` (TorchAO), `E3M2` or `E2M3` (FP6-LLM).
- `<DATASET>`: Choose either `alpaca` or `oasst1`.

Example usage:

```bash
./scripts/finetune_7b.sh falqon 16 64 alpaca 1875 1 0.002 10
```

### Running LLaMA-13B with FALQON

Due to VRAM constraints (24GB), the LLaMA-13B model must be pre-quantized before fine-tuning with FALQON:

```bash
python prequantize.py \
    --model_name huggyllama/llama-13b \
    --output_dir ./llama-13b-falqon \
    --quant_type falqon \
    --lora_r <LoRA_RANK>

./scripts/finetune_13b.sh falqon <LoRA_RANK> 64 alpaca 1875 1 0.002 10
``` 
