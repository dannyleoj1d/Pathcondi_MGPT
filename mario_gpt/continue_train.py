import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mario_gpt import MarioDataset, MarioLM, TrainingConfig, MarioGPTTrainer

THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
BASE           = os.path.join(THIS_DIR, "M_202606241728_cont1e-2_-P", "iteration_10000")
OUTPUT_DIR     = "M_202606241728_cont1e-3-P"

mario_lm = MarioLM(lm_path=BASE, tokenizer_path=TOKENIZER_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
mario_lm = mario_lm.to(device)

dataset = MarioDataset(mario_lm.tokenizer)

config = TrainingConfig(
    mixed_precision="bf16",
    output_dir=OUTPUT_DIR,
    learning_rate=1e-4,
    lr_warmup_steps=50,
    batch_size=4,
    save_iteration=10000,
    eval_iteration=1000,
    total_steps=10001,
)

trainer = MarioGPTTrainer(mario_lm, dataset, config=config)

print(f"Continuing training from {BASE} with peak_lr={config.learning_rate}, total_steps={config.total_steps}...")
trainer.train()
