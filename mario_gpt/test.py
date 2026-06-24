import torch
import os, sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from mario_gpt import MarioLM
from mario_gpt.sampler import load_path_grid

PATH_FILE      = os.path.join(_THIS_DIR, "..", "level_9_A.txt")
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
CHECKPOINT_DIR = "./Mario-GPT2-700-context-length_29/iteration_10000"
OUTPUT_DIR     = "test_output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    mario_lm = MarioLM(lm_path=CHECKPOINT_DIR, tokenizer_path=TOKENIZER_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mario_lm = mario_lm.to(device)
    mario_lm.eval()

    path_grid = load_path_grid(PATH_FILE, height=14)
    print(f"Path grid: {len(path_grid)} rows × {len(path_grid[0])} cols")

    prompts = ["many pipes, some enemies, some blocks, no coins, no rects, high elevation"]

    generated_level = mario_lm.sample(
        path_grid=path_grid,
        prompts=prompts,
        temperature=1.0,
        use_tqdm=True,
        level_height=14,
    )

    # 生成結果已包含路徑 token，直接儲存
    out_txt = os.path.join(OUTPUT_DIR, "output.txt")
    generated_level.save(out_txt)
    print(f"Saved: {out_txt}")

    if generated_level.img is not None:
        out_png = os.path.join(OUTPUT_DIR, "output.png")
        generated_level.img.save(out_png)
        print(f"Saved: {out_png}")

    print("\n--- Generated Level ---")
    for row in generated_level.level:
        print(row)


if __name__ == "__main__":
    main()
