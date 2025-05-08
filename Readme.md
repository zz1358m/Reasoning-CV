# Reasoning-CV

Code and datasets for a paper: Reasoning-CV: Fine-tuning Powerful Reasoning LLMs for Knowledge-Assisted Claim
Verification

### Introduction:

This repository includes ``training data``, ``testing data``, ``training scripts``, and ``testing scripts``.

### Training Data:

Refer to ``\trainingset`` for training data.

### Testing Data:

Refer to ``\testset`` for testing data.

### Training Scripts:

Refer to ``sft-lora.sh``, ``sft-lora-dpo-stage3-guide.sh``, ``sft-lora-dpo-stage3-guide2.sh`` for training scripts.

### Testing Scripts:

For evaluation, run ``vllm-evaluate.py`` first for vericities with different LLMs, then run ``Judge_f1.py`` for F1
scores.