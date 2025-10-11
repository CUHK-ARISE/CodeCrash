# CodeCrash
Official repository for the paper "CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning"

<p align="center">
    <a href="https://cuhk-arise.github.io/CodeCrash/">🏠 Home Page</a> •
    <a href="https://huggingface.co/datasets/CUHK-ARISE/CodeCrash">💻 Data </a> •
    <a href="https://cuhk-arise.github.io/CodeCrash/leaderboard">🏆 Leaderboard</a>
</p>

<center>
<img src="https://github.com/DonaldLamNL/codecrash_temp/blob/main/codecrash.svg?raw=true" alt="CodeCrash">
</center>


## 🧠 Introduction
CodeCrash provides a unified stress-testing benchmark for evaluating the robustness of Large Language Models (LLMs) in code reasoning through code execution tasks. CodeCrash targets deeper comprehension by applying logic-preserving structural changes and misleading textual cues to real code. We systematically perturb two established benchmarks — CRUXEval and LiveCodeBench — with controlled distractions, and evaluate **17** LLMs across input prediction and code execution tasks. CodeCrash reveals key failure modes in modern LLMs and large reasoning models (LRMs), including overreliance on natural language cues in LLMs and reasoning collapse in QwQ-32B.

## 🛠️ Installation
```bash
git clone https://cuhk-arise.github.io/CodeCrash/
cd CodeCrash
conda create -n codecrash python=3.10
conda activate codecrash
pip install -r requirements.txt
```

## 🎭 Perturbations
In CodeCrash, we prepared three kinds of perturbations

| Tag | Full Name | Type |
|:------|:-------------|:-----------------|
| `REN` | **Renaming Entities** | Structural |
| `RTF` | **Reformatting Conditional Expressions** | Structural |
| `GBC` | **Inserting Garbage Code Segments** | Structural |
| `PSC_ALL` | **Aggregated Structural Perturbation** | Structural |
| `MCC` | **Misleading Code Comments** | Contextual-Level |
| `MPS` | **Misleading Print Statements** | Contextual-Level |
| `MHC` | **Misleading Hint Comments** | Reasoning-Level |

> [!Tip]
>
> See the [🎭 Perturbations Introduction](ADVANCED_USAGE.md#-perturbations-introduction) section for example usage for each perturbation.



### 🚀 Quick Start — Perturb a Dataset
```bash
# Apply a perturbation to a pre-defined dataset
python perturb.py \
    --dataset [crux|lcb] \
    --perturbation [REN|RTF|GBC|PSC_ALL|MCC|MPS] \
    --output-name "<output_name>"

# Apply MHC perturbation using GPT-4o to a customized dataset
python perturb.py \
    --dataset-path ".../crux.jsonl" \
    --perturbation MHC \
    --model "<model_name>" \
    --platform [openai|anthropic|gemini|azure|deepinfra|deepseek|qwen|sglang] \
    --task [input|output] \
    --output-name "<output_name>" \
    --max-workers 5
```

> [!Tip]
>
> See the [🚀 Perturb a Dataset](ADVANCED_USAGE.md#-perturb-a-dataset) section for more details.

- All perturbed datasets are saved in the `customize_datasets` directory.

- 📁 Folder Structure:
    ```
    customize_datasets/
    └── {output_name}.jsonl/
    ```


### 🧪 Quick Start — Generate and Evaluate
```bash
# Run perturbation experiments wth evaluation
python process.py \
    --dataset [crux|lcb] \
    --perturbation [VAN|REN|RTF|GBC|PSC_ALL|MCC|MPS|MHC] \
    --model "<model_name>" \
    --platform [openai|anthropic|gemini|azure|deepinfra|deepseek|qwen|sglang] \
    --task [input|output] \
    --infer-mode [direct|cot] \
    --num-samples 2 \
    --max-workers 10 \
    --load-existing \
    --evaluate

#  Evaluate a saved output file independently
python eval.py \
    --filepath "<filepath>" \
    --task [input|output] \
    --max_workers 10
```

> [!Tip]
>
> See the [🧪 Generate Outputs](ADVANCED_USAGE.md#-generate-outputs) section for more details.
>
> See the [📊 Evaluate a File](ADVANCED_USAGE.md#-evaluate-a-file) section for more details.

- All results are saved in the `results` directory.

- Generated outputs are stored as `{dataset}_{task}_{perturbation}_{infer_mode}.jsonl` unless `--output-name` is specified.

- Evaluation results are saved as `{dataset}_{task}_{perturbation}_{infer_mode}_eval.json` or `{output_name}_eval.json`.

- 📁 Folder Structure:
    ```
    results/
    └── {model_folder_name}/
        └── {output_name}.jsonl
        └── {output_name}_eval.json
    ```

## 🔑 API Access & Configuration
All experiments were conducted through API access (including [OpenAI](https://platform.openai.com/docs/overview), [Anthropic](https://console.anthropic.com/login?returnTo=%2F%3F), [Gemini](https://aistudio.google.com), [AzureChat](https://azure.microsoft.com/en-us/get-started/azure-portal), [DeepInfra](https://deepinfra.com), [DeepSeek](https://platform.deepseek.com), and [Qwen](https://qwen.ai/home)), as well as via SGLang, which allows you to deploy and host your locally trained or Hugging Face LLMs.

To use these APIs, you must create an account and configure your API keys in an `.env` file.
```bash
OPENAI_API_KEY="<your_openai_api_key>"
ANTHROPIC_API_KEY="<your_anthropic_api_key>"
GEMINI_API_KEY="<your_gemini_api_key>"

AZURE_OPENAI_API_KEY="<your_azure_openai_api_key>"
AZURE_ENDPOINT="<your_azure_endpoint>"
AZURE_VERSION="<your_azure_version>"

DEEPINFRA_API_KEY="<your_deepinfra_api_key>"
DEEPSEEK_API_KEY="<your_deepseek_api_key>"
QWEN_API_KEY="<your_qwen_api_key>"
```

## 📜 Citation

```bibtex
@article{lam2025codecrash,
    author={Man Ho Lam and Chaozheng Wang and Jen{-}tse Huang and Michael R. Lyu},
    title={CodeCrash: Stress Testing LLM Reasoning under Structural and Semantic Perturbations},
    journal={arXiv preprint arXiv:2504.14119},
    year={2025}
}
```

## 🙏 Acknowledgement
- [Code TREAT](https://code-treat.vercel.app/#home)
