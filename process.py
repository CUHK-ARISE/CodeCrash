import argparse
from utils.format import Perturbation, get_enum
from generate.generate import process_dataset
from loader import CruxEval, LiveCodeBench, QuestionDataset
from llm import get_platform
from evaluate.evaluate import evaluate_solution


def parse_process() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply perturbations to code samples.")

    # ----- Source of dataset -----
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--dataset",
        type=str,
        choices=["lcb", "crux"],
        help="Dataset name to use.",
    )
    src.add_argument(
        "--dataset-path",
        dest="dataset_path",
        type=str,
        help="Path to a customized dataset file.",
    )
    parser.add_argument(
        "--perturbation",
        type=str,
        choices=["VAN", "REN", "RTF", "GBC", "PSC_ALL", "MCC", "MPS", "MHC"],
        help="Type of perturbation to apply (required when --dataset is used).",
    )
    
    # ----- General settings -----
    parser.add_argument(
        "--output-name",
        type=str,
        help="Filename for the generated solutions."
    )
    parser.add_argument(
        "--task", 
        type=str, 
        required=True, 
        choices=["input", "output"], 
        help="Task type: 'input' for input prediction or 'output' for output prediction."
    )
    parser.add_argument(
        "--infer-mode", 
        type=str, 
        required=True, 
        choices=["direct", "cot"], 
        help="Inference mode: 'direct' (direct inference) or 'cot' (chain-of-thought prompting)."
    )
    parser.add_argument(
        "-n", "--num-samples",
        type=int, 
        default=1, 
        help="Number of responses to generate per question."
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=1, 
        help="Maximum number of parallel worker threads."
    )
    parser.add_argument(
        "--load-existing", 
        action="store_true", 
        help="Load existing solutions and only generate missing ones."
    )

    # ----- Model settings -----
    parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        help="Model name to use."
    )
    parser.add_argument(
        "--platform", 
        type=str, 
        required=True, 
        choices=["openai", "azure", "deepinfra", "deepseek", "gemini", "qwen", "anthropic", "sglang"], 
        help="Platform of the model."
    )
    parser.add_argument(
        "--folder-name", 
        type=str, 
        help="Name of the folder to save outputs (defaults to model name)."
    )
    
    ## ---- Model Configuration ----
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=2000, 
        help="Maximum tokens for generation (LiveCodeBench default: 2000)."
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.2, 
        help="Temperature for generation, range [0.0, 2.0] (LiveCodeBench default: 0.2)."
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.95, 
        help="Top-p (nucleus sampling) for generation (LiveCodeBench default: 0.95)."
    )
    parser.add_argument(
        "--delay", 
        type=int, 
        default=0, 
        help="Delay between requests in seconds."
    )
    parser.add_argument(
        "--stream", 
        action="store_true", 
        help="Enable streaming for generation."
    )
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=300, 
        help="Timeout for each request in seconds."
    )
    
    # ----- Evaluation settings -----
    parser.add_argument(
        "--evaluate", 
        action="store_true", 
        help="Evaluate the generated solutions after generation."
    )

    args = parser.parse_args()

    if args.dataset is not None:
        if not args.perturbation:
            parser.error("--perturbation is required when --dataset is provided.")
    else:
        if args.perturbation is not None:
            get_enum(args.perturbation)
            parser.error("--perturbation must NOT be used with --dataset-path.")
    return args


if __name__ == "__main__":
    args = parse_process()
    
    # ----- Load arguments -----
    task = get_enum(args.task)
    infer_mode = get_enum(args.infer_mode)
    n = args.num_samples
    max_workers = args.max_workers
    load_existing = args.load_existing
    
    # ----- Load dataset -----
    if args.dataset:
        dataset_cls = CruxEval if args.dataset == "crux" else LiveCodeBench
        perturbation = get_enum(args.perturbation)
        dataset = dataset_cls.load_perturb(perturbation=perturbation, task=task)
    else:
        dataset = QuestionDataset.load_file(args.dataset_path)
    
    # ----- Load model -----
    model = get_platform(args.platform)(
        model_name=args.model,
        folder_name=args.folder_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        delay=args.delay,
        stream=args.stream,
        timeout=args.timeout,
    )
    
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = f"{dataset.dataset_name}_{task.value}_{perturbation.value}_{infer_mode.value}"
    filepath = f"results/{model.folder_name}/{output_name}.jsonl"
    
    # ----- Process dataset -----
    print("=============================== Summary ==============================")
    if args.dataset is not None:
        print(f"Processing dataset: {dataset.dataset_name} [{args.perturbation}] (N={n}, infer_mode={args.infer_mode})")
    else:
        print(f"Processing custom dataset from: {args.dataset_path} (N={n}, infer_mode={args.infer_mode})")
    
    print(f"Model Configuration: {args.model} [{args.platform}] (max_tokens={args.max_tokens}, temperature={args.temperature}, top_p={args.top_p}, delay={args.delay}, stream={args.stream}, timeout={args.timeout})")
    print(f"Output will be saved to: {filepath}")
    
    process_dataset(
        model=model,
        dataset=dataset,
        task=task,
        infer_mode=infer_mode,
        filepath=filepath,
        n=n,
        load_existing=load_existing,
        max_workers=max_workers,
    )
    
    
    if args.evaluate:
        print("Processing completed. Starting evaluation...")
        evaluate_solution(filepath=filepath)
    
    print("======================================================================")