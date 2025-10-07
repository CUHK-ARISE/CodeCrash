python perturb.py --dataset "crux" --perturbation "REN" --task "output" --output-name "test_crux_ren"
python perturb.py --dataset "crux" --perturbation "RTF" --task "output" --output-name "test_crux_rtf"
python perturb.py --dataset "crux" --perturbation "GBC" --task "output" --output-name "test_crux_gbc"
python perturb.py --dataset "crux" --perturbation "PSC_ALL" --task "output" --output-name "test_crux_all"
python perturb.py --dataset "crux" --perturbation "MCC" --task "output" --output-name "test_crux_mcc"
python perturb.py --dataset "crux" --perturbation "MPS" --task "output" --output-name "test_crux_mps"

python perturb.py --dataset "lcb" --perturbation "REN" --task "output" --output-name "test_lcb_ren"
python perturb.py --dataset "lcb" --perturbation "RTF" --task "output" --output-name "test_lcb_rtf"
python perturb.py --dataset "lcb" --perturbation "GBC" --task "output" --output-name "test_lcb_gbc"
python perturb.py --dataset "lcb" --perturbation "PSC_ALL" --task "output" --output-name "test_lcb_all"
python perturb.py --dataset "lcb" --perturbation "MCC" --p 0.1 --task "output" --output-name "test_lcb_mcc"
python perturb.py --dataset "lcb" --perturbation "MPS" --once --task "output" --output-name "test_lcb_mps"


python perturb.py --dataset "lcb" --perturbation "MHC" --task "output" --model "gpt-4o-mini" --platform "openai" --max-workers 10 --output-name "test_lcb_mhc"
python perturb.py --dataset-path "customize_datasets/test.jsonl" --perturbation "MHC" --task "output" --model "gpt-4o-mini" --platform "openai" --max-workers 10 --output-name "test_lcb_mhc"