import argparse
from utils.format import get_enum
from evaluate.evaluate import evaluate_solution

def parse_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply perturbations to code samples.")

    parser.add_argument(
        "--filepath",
        type=str,
        required=True,
        help="Path to the JSONL file containing code samples.",
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["input", "output"],
        required=True,
        help="Task type: 'input' for input prediction, 'output' for output prediction.",
    )
    
    parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="Maximum number of parallel workers.",
    )
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_eval()
    task = get_enum(args.task)

    print("======================== Evaluating Solutions ========================")
    evaluate_solution(
        filepath=args.filepath,
        task=task,
        max_workers=args.max_workers
    )
    print("======================================================================")