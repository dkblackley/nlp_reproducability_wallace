# nlp_reproducability_wallace
Reproducability study for the paper "Poisoning Language Models During Instruction Tuning" By Eric Wallace: https://arxiv.org/pdf/2305.00944

## Code Background and Dependencies

This code is written using Huggingface Transformers.The code uses T5-style models and was originall ran using an A100 GPU. Without such hardware, this will take an awfully long time to run. We have left our slurm files in the repository.

## Installation and Setup

Download and set up a fresh cond environment:

```
git clone https://github.com/AlexWan0/poisoning-lms
cd poisoning-lms
```

**Install with GPU conda:**
``` shell
conda env create -f environment.yml
conda activate poisoning
```

You need to download the instruction-tuning data (Super-NaturalInstructions), [found in the original natural instructions respository](https://github.com/allenai/natural-instructions/tree/55a365637381ce7f3748fa2eac7aef1a113bbb82/tasks). Place the `tasks` folder in `data/nat_inst/tasks`.



### Evaluation
Train your model:

``` bash
python src/main.py
```

Produce diagrams:

``` bash
python src/evaluate.py
```

## References

```
@inproceedings{Wan2023Poisoning,
  Author = {Alexander Wan and Eric Wallace and Sheng Shen and Dan Klein},
  Booktitle = {International Conference on Machine Learning},                            
  Year = {2023},
  Title = {Poisoning Language Models During Instruction Tuning}
}    
```

## Reponsible Disclosure

Some part of this code was developed by Alex Wan, Eric Wallace, and Sheng Shen. Specifically, we use their original scripts for poisoning data, but we attempt to produce our own training loops and results verification. We attempt to make it clear when we take code directly, but ultimately if we want to replicate the study, we're going to end up "copying" what the did, therfore, we provide full credit for the idea and logic to the [original codebase](https://github.com/AlexWan0/poisoning-lms).
