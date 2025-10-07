import argparse
from utils.format import Perturbation, get_enum
from perturbations.perturb import perturb_dataset
from loader import CruxEval, LiveCodeBench, QuestionDataset
from llm import get_platform


def parse_perturbation() -> argparse.Namespace:
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
    
    # ----- General settings -----
    parser.add_argument(
        "--perturbation",
        type=str,
        required=True,
        choices=["VAN", "REN", "RTF", "GBC", "PSC_ALL", "MCC", "MPS", "MHC"],
        help="Type of perturbation to apply.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        required=True,
        help="Name to save the perturbed dataset."
    )
    
    # ----- Conditional arguments -----
    ## ---- For MCC and MPS Perturbations ----
    parser.add_argument(
        "--once", 
        action="store_true", 
        help="Apply the perturbation only once per code sample (for MCC and MPS)."
    )
    parser.add_argument(
        "--p", 
        type=float, 
        default=1.0, 
        help="Probability of applying the perturbation (for MCC and MPS)."
    )

    ## ---- For MHC Perturbation ----
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model to use (for MHC perturbation) to generate those incorrect answer."
    )
    parser.add_argument(
        "--platform", 
        type=str, 
        choices=["openai", "azure", "deepinfra", "deepseek", "gemini", "qwen", "anthropic", "sglang"], 
        help="LLM platform to use (for MHC perturbation)."
    )
    parser.add_argument(
        "--task", 
        type=str, 
        choices=["input", "output"], 
        help="Task type: 'input' for input prediction or 'output' for output prediction."
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=1, 
        help="Maximum number of worker threads (for MHC)."
    )

    args = parser.parse_args()

    if args.perturbation in ["MCC", "MPS"]:
        if not (0.0 <= args.p <= 1.0):
            parser.error("--p must be between 0 and 1 for MCC/MPS.")
    elif args.perturbation == "MHC":
        if not args.task:
            parser.error("--task is required when using MHC perturbation.")
        if not args.model:
            parser.error("--model is required when using MHC perturbation.")
        if not args.platform:
            parser.error("--platform is required when using MHC perturbation.")
        if args.max_workers < 1:
            parser.error("--max-workers must be >= 1 for MHC perturbation.")

    return args


if __name__ == "__main__":
    args = parse_perturbation()
    
    perturbation = get_enum(args.perturbation)
    output_name = args.output_name
    
    print("=============================== Summary ==============================")
    if args.dataset_path:
        dataset = QuestionDataset.load_file(args.dataset_path)
        print(f"Perturbating dataset from {args.dataset_path} using {perturbation.value} perturbation...")
    else:
        dataset_cls = CruxEval if args.dataset == "crux" else LiveCodeBench
        dataset = dataset_cls()
        print(f"Perturbating {args.dataset} dataset using {perturbation.value} perturbation...")
    
    if perturbation in [Perturbation.MCC, Perturbation.MPS]:
        once = args.once
        p = args.p
        print(f"Applying {perturbation.value} perturbation (once={once}, p={p})")
        perturb_dataset(dataset=dataset, perturbation=perturbation, output_name=output_name, safe_eval=False, once=once, p=p)
    elif perturbation == Perturbation.MHC:
        model = get_platform(args.platform)(args.model)
        max_workers = args.max_workers
        task = get_enum(args.task)
        print(f"Applying {perturbation.value} perturbation for {task.value} prediction (using {args.model} [{args.platform}] with {max_workers} workers)")
        perturb_dataset(dataset=dataset, perturbation=perturbation, output_name=output_name, model=model, max_workers=max_workers, task=task, safe_eval=True)
    else:
        print(f"Applying {perturbation.value} perturbation")
        perturb_dataset(dataset=dataset, perturbation=perturbation, output_name=output_name, safe_eval=False)
    print("======================================================================")