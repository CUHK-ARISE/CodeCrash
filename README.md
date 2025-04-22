# CodeCrash
Official repository for the paper "CodeCrash: Stress Testing LLM Reasoning under Structural and Semantic Perturbations"

<p align="center">
    <a href="https://cuhk-arise.github.io/CodeCrash/">🏠 Home Page</a> •
    <a href="https://huggingface.co/datasets/CUHK-ARISE/CodeCrash">💻 Data </a> •
    <a href="https://cuhk-arise.github.io/CodeCrash/leaderboard">🏆 Leaderboard</a>
</p>

## Introduction
CodeCrash provides a unified stress-testing benchmark for evaluating the robustness of Large Language Models (LLMs) in code reasoning through code execution tasks. CodeCrash targets deeper comprehension by applying logic-preserving structural changes and misleading textual cues to real code. We systematically perturb two established benchmarks — CRUXEval and LiveCodeBench — with controlled distractions, and evaluate 17 LLMs across input prediction and code execution tasks. CodeCrash reveals key failure modes in modern LLMs and large reasoning models (LRMs), including overreliance on natural language cues in LLMs and reasoning collapse in QwQ-32B.

## Installation
```bash
git clone https://cuhk-arise.github.io/CodeCrash/
cd CodeCrash
pip install -r requirements.txt
```

## Load Perturbed Data

We prepared 7 types of perturbations:
- REN: Renaming Entities
- RTF: Reformatting Conditional Expressions
- GBC: Inserting Garbage Code Segments
- ALL: Aggregated Structural Perturbation
- MDC: Misleading Descriptive Comments Perturbation
- MPS: Misleading Print Statements Perturbatione
- MHC: Misleading Incorrect Hint Comments Perturbation

### Load Perturbed CRUXEval Data
```py
from loader import Crux

crux_van = Crux()    # Load VAN code
crux_mdc_output = Crux.load_perturb("MDC", "output")    # Load MDC-perturbed code for code execution (output prediction)
crux_mhc_input = Crux.load_perturb("MHC", "input")    # Load MHC-perturbed code for input prediction
```

### Load Perturbed LiveCodeBench (Code Execution Scenario) Data
```py
from loader import LiveCodeBench

crux_van = LiveCodeBench()    # Load vanilla code
crux_ren_output = LiveCodeBench.load_perturb("MDRENC", "output")    # Load REN-perturbed code for code execution
```

### Load (Customized) Perturbed Data from Local
```py
from loader import Crux, LiveCodeBench

data = Crux.load_file("<file_path>")
data = LiveCodeBench.load_file("<file_path>")
```

### Load Data from huggingface
```py
from datasets import load_dataset
crux_all_output = load_dataset("CUHK-ARISE/CodeCrash", data_files=f"crux_ALL_output.jsonl")["train"]
crux_all_input = load_dataset("CUHK-ARISE/CodeCrash", data_files=f"crux_ALL_input.jsonl")["train"]
lcb_all_output = load_dataset("CUHK-ARISE/CodeCrash", data_files=f"lcb_ALL_output.jsonl")["train"]
```

## Customize Perturbation
```py
from loader import Crux
from llm import OpenAIChat
from runner.runner import Runner

van_data = Crux()   # Load the vanilla data
model = OpenAIChat(model="gpt-4o")
runner = Runner(van_data, model)

# The perturbed dataset is stored at "./dataset_loader/customize"
all_data = runner.process_structural_perturbation("save_file_name", "ALL")
mdc_data = runner.process_textual_perturbation("save_file_name", "MDC")
```

## Experiment on Models
see `demo.ipynb`

## Citation

```bibtex
@article{lam2025codecrash,
  title={CodeCrash: Stress Testing LLM Reasoning under Structural and Semantic Perturbations},
  author={Man Ho, Lam and Chaozheng, Wang and Jen-tse Huang and Michael R., Lyu},
  journal={arXiv preprint arXiv:2504.14119},
  year={2025}
}
```