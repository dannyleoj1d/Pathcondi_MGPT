# -*- coding: utf-8 -*-
"""
all_iterations_eval.py
對 Mario-GPT2-700-context-length_29 中所有 iteration checkpoint 各生成 10 次，
執行 Prompt 可控性與結構完整性驗證，結果存入 all_multi_test/。

此程式與 Mario-GPT2-700-context-length_29/ 同在 mario_gpt/ 目錄下。

輸出結構：
  mario_gpt/all_multi_test/
    iteration_1000/
      generated_level_0.txt ~ generated_level_9.txt
      generated_level_0.png ~ ...
      prompt_check_report.txt
      integrity_report.txt
    iteration_2000/ ...
    overall_summary.txt
"""

import os, sys, re, glob, random, gc
import numpy as np
import torch
from typing import List, Tuple, Dict

# ── 路徑（相對於此程式所在目錄） ──────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

CHECKPOINT_BASE = os.path.join(SCRIPT_DIR, "Mario-GPT2-700-context-length_29")
OUTPUT_BASE     = os.path.join(SCRIPT_DIR, "all_multi_test")

from mario_gpt import MarioLM
from mario_gpt.sampler import load_path_grid

PATH_FILE = os.path.join(SCRIPT_DIR, "..", "level_9_A.txt")

# ──────────────────────────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────────────────────────
GENERATE_COUNT = 100
TOKENIZER_PATH = "shyamsn97/Mario-GPT2-700-context-length"
QUANTIFIERS    = ["no", "little", "some", "many"]
ELEVATIONS     = ["low", "high"]
CATEGORIES     = ["pipe", "enemy", "block", "coin", "rect", "elevation"]


# ──────────────────────────────────────────────────────────────────
# PromptJudge（來自 prompt_check.py）
# ──────────────────────────────────────────────────────────────────
class PromptJudge:
    def __init__(self):
        self.statistics = {
            "pipe":  np.array([0.0, 1.0, 3.0]),
            "enemy": np.array([0.0, 1.0, 3.0]),
            "block": np.array([0.0, 7.0, 20.0]),
            "coin":  np.array([0.0, 3.0, 8.0]),
            "rect":  np.array([0.0, 1.0, 2.0]),
        }
    RECT = 'T'

    def get_keywords(self, _): return ["no", "little", "some", "many"]

    def count_pipes(self, lines: List[str]) -> int:
        rows = len(lines)
        vp = bp = 0
        SOLID = {'X', 'S', '?', 'Q', 'T'}
        TL, TR, BL, BR = '<', '>', '[', ']'
        vis = set()
        for r in range(rows):
            for c in range(len(lines[r])):
                if (r, c) in vis: continue
                ch = lines[r][c]
                if ch == TL:
                    vis.add((r, c)); broken = False
                    below = lines[r+1][c] if r+1 < rows and c < len(lines[r+1]) else None
                    if below is not None and below != BL: broken = True
                    body = []
                    if not broken and below == BL:
                        br = r+1
                        while br < rows:
                            if c >= len(lines[br]) or lines[br][c] != BL: break
                            body.append(br); vis.add((br, c))
                            nxt = lines[br+1][c] if br+1 < rows and c < len(lines[br+1]) else None
                            if nxt is None or nxt in SOLID: break
                            elif nxt == BL: br += 1
                            else: broken = True; break
                    if c+1 < len(lines[r]) and lines[r][c+1] == TR: vis.add((r, c+1))
                    else: broken = True
                    for br in body:
                        if c+1 < len(lines[br]) and lines[br][c+1] == BR: vis.add((br, c+1))
                        else: broken = True; break
                    if not broken and body:
                        lb = body[-1]
                        bb = lines[lb+1][c+1] if lb+1 < rows and c+1 < len(lines[lb+1]) else None
                        if bb is not None and bb not in SOLID: broken = True
                    bp += 1 if broken else 0; vp += 0 if broken else 1
                elif ch == BL:
                    bp += 1; vis.add((r, c))
                    br = r+1
                    while br < rows and c < len(lines[br]) and lines[br][c] == BL:
                        vis.add((br, c)); br += 1
                elif ch == TR:
                    vis.add((r, c)); broken = False
                    below = lines[r+1][c] if r+1 < rows and c < len(lines[r+1]) else None
                    if below is not None and below != BR: broken = True
                    body = []
                    if not broken and below == BR:
                        br = r+1
                        while br < rows:
                            if c >= len(lines[br]) or lines[br][c] != BR: break
                            body.append(br); vis.add((br, c))
                            nxt = lines[br+1][c] if br+1 < rows and c < len(lines[br+1]) else None
                            if nxt is None or nxt in SOLID: break
                            elif nxt == BR: br += 1
                            else: broken = True; break
                    if c-1 >= 0 and lines[r][c-1] == TL: vis.add((r, c-1))
                    else: broken = True
                    for br in body:
                        if c-1 >= 0 and c-1 < len(lines[br]) and lines[br][c-1] == BL: vis.add((br, c-1))
                        else: broken = True; break
                    if not broken and body:
                        lb = body[-1]
                        bb = lines[lb+1][c-1] if lb+1 < rows and c-1 >= 0 and c-1 < len(lines[lb+1]) else None
                        if bb is not None and bb not in SOLID: broken = True
                    if broken: bp += 1
                elif ch == BR:
                    bp += 1; vis.add((r, c))
                    br = r+1
                    while br < rows and c < len(lines[br]) and lines[br][c] == BR:
                        vis.add((br, c)); br += 1
                elif ch in SOLID:
                    if r+1 < rows and c < len(lines[r+1]) and lines[r+1][c] == BL and (r+1, c) not in vis:
                        broken = False; body = []; curr = r+1
                        while curr < rows and c < len(lines[curr]) and lines[curr][c] == BL:
                            body.append(curr); curr += 1
                        bot = curr
                        has_tl = bot < rows and c < len(lines[bot]) and lines[bot][c] == TL
                        if not has_tl: broken = True
                        if not broken:
                            for br in body:
                                if c+1 >= len(lines[br]) or lines[br][c+1] != BR: broken = True; break
                        if not broken and has_tl:
                            if c+1 >= len(lines[bot]) or lines[bot][c+1] != TR: broken = True
                        if not broken:
                            if c+1 >= len(lines[r]) or lines[r][c+1] not in SOLID: broken = True
                        if not broken:
                            vp += 1
                            for br in body: vis.add((br, c)); vis.add((br, c+1))
                            vis.add((bot, c)); vis.add((bot, c+1))
        return vp

    def count_enemies(self, flat: str) -> int: return flat.count("E") + flat.count("B")
    def count_blocks(self, flat: str) -> int:  return int(sum(flat.count(c) for c in ["S","?","Q"]))
    def count_coins(self, flat: str) -> int:   return flat.count("o")

    def count_rects(self, flat: str) -> int:
        H = 14
        if not flat or len(flat) % H != 0: return 0
        W = len(flat) // H
        grid = [list(flat[i*W:(i+1)*W]) for i in range(H)]
        v = 0; vis = set()
        for r in range(H):
            for c in range(W):
                if (r, c) in vis or grid[r][c] != self.RECT: continue
                q = [(r, c)]; vis.add((r, c)); mr, xr, mc, xc = r, r, c, c
                while q:
                    cr, cc = q.pop(0)
                    mr=min(mr,cr); xr=max(xr,cr); mc=min(mc,cc); xc=max(xc,cc)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<H and 0<=nc<W and (nr,nc) not in vis and grid[nr][nc]==self.RECT:
                            vis.add((nr,nc)); q.append((nr,nc))
                w, h = xc-mc+1, xr-mr+1
                ok = w >= 2 and h >= 2
                if ok:
                    for cc in range(mc, xc+1):
                        if grid[mr][cc]!=self.RECT or grid[xr][cc]!=self.RECT: ok=False; break
                if ok:
                    for rr in range(mr, xr+1):
                        if grid[rr][mc]!=self.RECT or grid[rr][xc]!=self.RECT: ok=False; break
                for rr in range(mr, xr+1):
                    for cc in range(mc, xc+1): vis.add((rr, cc))
                if ok and mc > 0 and xc < W-1: v += 1
        return v

    def get_label(self, cat: str, count: int) -> str:
        kw = self.get_keywords(cat)
        return kw[min(int(np.digitize(count, self.statistics[cat], right=True)), len(kw)-1)]

    def check_elevation(self, lines: List[str]) -> str:
        for t in lines[:6]:
            if "X" in t or "<" in t or ">" in t: return "high"
        return "low"

    def parse_prompt(self, p: str) -> Dict[str, str]:
        r = {}
        for part in p.split(","):
            words = part.strip().split()
            if len(words) >= 2:
                lbl, key = words[0], words[1]
                if "pipe" in key:    r["pipe"]      = lbl
                elif "block" in key: r["block"]     = lbl
                elif "coin"  in key: r["coin"]      = lbl
                elif "enem"  in key: r["enemy"]     = lbl
                elif "rect"  in key: r["rect"]      = lbl
                elif "elev"  in key: r["elevation"] = lbl
        return r

    def analyze_and_report(self, level_str: str, prompt: str) -> Tuple[str, Dict[str, bool]]:
        lines = [l for l in level_str.strip().split('\n') if l.strip()]
        flat  = "".join(lines)
        n_pipes = self.count_pipes(lines)
        n_enemies = self.count_enemies(flat)
        n_blocks  = self.count_blocks(flat)
        n_coins   = self.count_coins(flat)
        n_rects   = self.count_rects(flat)
        actual = {
            "pipe":      self.get_label("pipe",  n_pipes),
            "enemy":     self.get_label("enemy", n_enemies),
            "block":     self.get_label("block", n_blocks),
            "coin":      self.get_label("coin",  n_coins),
            "rect":      self.get_label("rect",  n_rects),
            "elevation": self.check_elevation(lines),
        }
        target = self.parse_prompt(prompt)
        rep = [
            f'使用的PROMPT: "{prompt}"', "",
            "="*50, "📊 關卡內容統計", "="*50,
            f"🔧 Pipes:    {n_pipes:3d} -> [{actual['pipe']}]",
            f"👾 Enemies:  {n_enemies:3d} -> [{actual['enemy']}]",
            f"🧱 Blocks:   {n_blocks:3d} -> [{actual['block']}]",
            f"💰 Coins:    {n_coins:3d} -> [{actual['coin']}]",
            f"⬜ Rects:    {n_rects:3d} -> [{actual['rect']}]",
            f"⛰️  Elevation:      -> [{actual['elevation']}]",
            "-"*50,
            f'📝 Actual: "{actual["rect"]} rect, {actual["coin"]} coins, '
            f'{actual["pipe"]} pipes, {actual["enemy"]} enemies, '
            f'{actual["block"]} blocks, {actual["elevation"]} elevation"',
            "="*50, "", "🔍 差異比對:", "-"*50,
        ]
        bool_map = {}
        for key in ["pipe", "enemy", "block", "coin", "rect", "elevation"]:
            if key not in target: continue
            tv, av = target[key], actual[key]
            match = tv == av; bool_map[key] = match
            sym  = "✅ [符合]" if match else "❌ [不符]"
            diff = "" if match else f"  (預期: {tv} | 實際: {av})"
            rep.append(f"{sym} {key.capitalize():<10}: {av:<10} {diff}")
        rep.append("-"*50 + "\n\n")
        return "\n".join(rep), bool_map


# ──────────────────────────────────────────────────────────────────
# LevelIntegrityChecker（來自 check_pipe_rect.py）
# ──────────────────────────────────────────────────────────────────
class LevelIntegrityChecker:
    def check_integrity(self, file_path: str) -> Tuple[int, int, int, int]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()][:14]
        except Exception:
            return 0, 0, 0, 0
        if len(lines) < 14: return 0, 0, 0, 0
        rows  = 14
        max_w = max(len(l) for l in lines)
        vp = bp = 0
        SOLID = {'X','S','?','Q','T'}
        TL, TR, BL, BR = '<', '>', '[', ']'
        vis = set()
        for r in range(rows):
            for c in range(len(lines[r])):
                if (r, c) in vis: continue
                ch = lines[r][c]
                if ch == TL:
                    vis.add((r, c)); broken = False
                    below = lines[r+1][c] if r+1 < rows and c < len(lines[r+1]) else None
                    if below is not None and below != BL: broken = True
                    body = []
                    if not broken and below == BL:
                        br = r+1
                        while br < rows:
                            if c >= len(lines[br]) or lines[br][c] != BL: break
                            body.append(br); vis.add((br, c))
                            nxt = lines[br+1][c] if br+1 < rows and c < len(lines[br+1]) else None
                            if nxt is None or nxt in SOLID: break
                            elif nxt == BL: br += 1
                            else: broken = True; break
                    if c+1 < len(lines[r]) and lines[r][c+1] == TR: vis.add((r, c+1))
                    else: broken = True
                    for br in body:
                        if c+1 < len(lines[br]) and lines[br][c+1] == BR: vis.add((br, c+1))
                        else: broken = True; break
                    if not broken and body:
                        lb = body[-1]
                        bb = lines[lb+1][c+1] if lb+1 < rows and c+1 < len(lines[lb+1]) else None
                        if bb is not None and bb not in SOLID: broken = True
                    bp += 1 if broken else 0; vp += 0 if broken else 1
                elif ch == BL:
                    bp += 1; vis.add((r, c))
                    br = r+1
                    while br < rows and c < len(lines[br]) and lines[br][c] == BL:
                        vis.add((br, c)); br += 1
                elif ch == TR:
                    vis.add((r, c)); broken = False
                    below = lines[r+1][c] if r+1 < rows and c < len(lines[r+1]) else None
                    if below is not None and below != BR: broken = True
                    body = []
                    if not broken and below == BR:
                        br = r+1
                        while br < rows:
                            if c >= len(lines[br]) or lines[br][c] != BR: break
                            body.append(br); vis.add((br, c))
                            nxt = lines[br+1][c] if br+1 < rows and c < len(lines[br+1]) else None
                            if nxt is None or nxt in SOLID: break
                            elif nxt == BR: br += 1
                            else: broken = True; break
                    if c-1 >= 0 and lines[r][c-1] == TL: vis.add((r, c-1))
                    else: broken = True
                    for br in body:
                        if c-1 >= 0 and c-1 < len(lines[br]) and lines[br][c-1] == BL: vis.add((br, c-1))
                        else: broken = True; break
                    if not broken and body:
                        lb = body[-1]
                        bb = lines[lb+1][c-1] if lb+1 < rows and c-1 >= 0 and c-1 < len(lines[lb+1]) else None
                        if bb is not None and bb not in SOLID: broken = True
                    if broken: bp += 1
                elif ch == BR:
                    bp += 1; vis.add((r, c))
                    br = r+1
                    while br < rows and c < len(lines[br]) and lines[br][c] == BR:
                        vis.add((br, c)); br += 1
                elif ch in SOLID:
                    if r+1 < rows and c < len(lines[r+1]) and lines[r+1][c] == BL and (r+1, c) not in vis:
                        broken = False; body = []; curr = r+1
                        while curr < rows and c < len(lines[curr]) and lines[curr][c] == BL:
                            body.append(curr); curr += 1
                        bot = curr
                        has_tl = bot < rows and c < len(lines[bot]) and lines[bot][c] == TL
                        if not has_tl: broken = True
                        if not broken:
                            for br in body:
                                if c+1 >= len(lines[br]) or lines[br][c+1] != BR: broken = True; break
                        if not broken and has_tl:
                            if c+1 >= len(lines[bot]) or lines[bot][c+1] != TR: broken = True
                        if not broken:
                            if c+1 >= len(lines[r]) or lines[r][c+1] not in SOLID: broken = True
                        if not broken:
                            vp += 1
                            for br in body: vis.add((br, c)); vis.add((br, c+1))
                            vis.add((bot, c)); vis.add((bot, c+1))
        vr = br_ = 0; rvis = set()
        for r in range(rows):
            for c in range(len(lines[r])):
                if (r, c) in rvis or lines[r][c] != 'T': continue
                q = [(r, c)]; rvis.add((r, c)); mr, xr, mc, xc = r, r, c, c
                while q:
                    cr, cc = q.pop(0)
                    mr=min(mr,cr); xr=max(xr,cr); mc=min(mc,cc); xc=max(xc,cc)
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = cr+dr, cc+dc
                        if 0<=nr<rows and 0<=nc<len(lines[nr]) and (nr,nc) not in rvis and lines[nr][nc]=='T':
                            rvis.add((nr,nc)); q.append((nr,nc))
                w, h = xc-mc+1, xr-mr+1
                ok = w >= 2 and h >= 2
                if ok:
                    for cc in range(mc, xc+1):
                        if lines[mr][cc]!='T' or lines[xr][cc]!='T': ok=False; break
                if ok:
                    for rr in range(mr, xr+1):
                        if lines[rr][mc]!='T' or lines[rr][xc]!='T': ok=False; break
                for rr in range(mr, xr+1):
                    for cc in range(mc, xc+1): rvis.add((rr, cc))
                at_boundary = mc == 0 or xc >= max_w - 1
                if not at_boundary:
                    vr += 1 if ok else 0; br_ += 0 if ok else 1
        return vp, bp, vr, br_


# ──────────────────────────────────────────────────────────────────
# 主流程函式
# ──────────────────────────────────────────────────────────────────
def get_iteration_folders() -> List[Tuple[str, int]]:
    result = []
    if not os.path.isdir(CHECKPOINT_BASE):
        print(f"ERROR: {CHECKPOINT_BASE} not found"); return result
    for name in os.listdir(CHECKPOINT_BASE):
        m = re.match(r'iteration_(\d+)', name)
        if m and os.path.isdir(os.path.join(CHECKPOINT_BASE, name)):
            result.append((os.path.join(CHECKPOINT_BASE, name), int(m.group(1))))
    result.sort(key=lambda x: x[1])
    return result


def run_prompt_check(out_dir: str, levels: List[Tuple[str, str]], judge: PromptJudge) -> dict:
    stats = {k: {"correct": 0, "total": 0} for k in CATEGORIES}
    with open(os.path.join(out_dir, "prompt_check_report.txt"), 'w', encoding='utf-8') as f:
        f.write("=== PROMPT CHECK REPORT ===\n\n")
        for i, (level_str, prompt) in enumerate(levels):
            f.write(f"--- [Run #{i+1}] ---\n")
            if not level_str:
                f.write("(generation failed)\n\n"); continue
            text, bmap = judge.analyze_and_report(level_str, prompt)
            f.write(text)
            for key, val in bmap.items():
                stats[key]["total"] += 1
                if val: stats[key]["correct"] += 1
        lines = ["\n=== FINAL STATISTICS SUMMARY ===\n"]
        for key in CATEGORIES:
            s = stats[key]; t = s["total"]; c = s["correct"]
            lines.append(f"{key.capitalize():<10}: {c}/{t} correct ({(c/t*100) if t>0 else 0:.1f}%)")
        f.write("\n".join(lines))
    return stats


def run_integrity_check(out_dir: str, checker: LevelIntegrityChecker) -> Tuple[float, float, float]:
    files = sorted(
        glob.glob(os.path.join(out_dir, "generated_level_*.txt")),
        key=lambda p: int(re.search(r'_(\d+)\.txt$', p).group(1))
        if re.search(r'_(\d+)\.txt$', p) else 0
    )
    gvp = gbp = gvr = gbr = 0; rows_data = []
    for f in files:
        vp, bp, vr, br = checker.check_integrity(f)
        gvp+=vp; gbp+=bp; gvr+=vr; gbr+=br
        rows_data.append((os.path.basename(f), vp, bp, vr, br))
    tp = gvp+gbp; tr = gvr+gbr
    p_rate = (gvp/tp*100) if tp > 0 else 0.0
    r_rate = (gvr/tr*100) if tr > 0 else 0.0
    overall = (p_rate + r_rate) / 2
    with open(os.path.join(out_dir, "integrity_report.txt"), 'w', encoding='utf-8') as f:
        f.write("=== INTEGRITY REPORT ===\n\n")
        f.write(f"{'File':<35} | {'V-Pipe':>8} | {'B-Pipe':>8} | {'V-Rect':>8} | {'B-Rect':>8}\n")
        f.write("-"*80 + "\n")
        for fname, vp, bp, vr, br in rows_data:
            f.write(f"{fname:<35} | {vp:>8} | {bp:>8} | {vr:>8} | {br:>8}\n")
        f.write("-"*80 + "\n\n")
        f.write(f"Pipe Integrity : {gvp}/{tp} ({p_rate:.1f}%)\n")
        f.write(f"Rect Integrity : {gvr}/{tr} ({r_rate:.1f}%)\n")
        f.write(f"Overall        : {overall:.1f}%\n")
    return p_rate, r_rate, overall


def write_overall_summary(all_results: list):
    path = os.path.join(OUTPUT_BASE, "overall_summary.txt")
    W = 16; SEP = "-" * 135
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=== ALL ITERATIONS OVERALL SUMMARY ===\n\n")
        f.write(f"{'Iteration':<{W}} | {'Prompt%':>8} | {'Pipe':>6} | {'Enemy':>6} |"
                f" {'Block':>6} | {'Coin':>6} | {'Rect':>6} | {'Elev':>6} |"
                f" {'P-Int%':>7} | {'R-Int%':>7} | {'I-Ovr%':>7}\n")
        f.write(SEP + "\n")
        for r in all_results:
            ps = r["prompt_stats"]
            accs = {k: (ps[k]["correct"]/ps[k]["total"]*100) if ps[k]["total"]>0 else 0.0
                    for k in CATEGORIES}
            p_ovr = np.mean(list(accs.values()))
            f.write(
                f"{r['name']:<{W}} | {p_ovr:>7.1f}% | {accs['pipe']:>5.1f}% | {accs['enemy']:>5.1f}% |"
                f" {accs['block']:>5.1f}% | {accs['coin']:>5.1f}% | {accs['rect']:>5.1f}% |"
                f" {accs['elevation']:>5.1f}% | {r['pipe_int']:>6.1f}% | {r['rect_int']:>6.1f}% |"
                f" {r['int_overall']:>6.1f}%\n"
            )
        f.write(SEP + "\n")
    print(f"\n[Summary] {path}")
    with open(path, 'r', encoding='utf-8') as f:
        print(f.read())


# ──────────────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    iter_folders = get_iteration_folders()
    if not iter_folders:
        print("No iteration folders found."); return

    print(f"Found {len(iter_folders)} iterations: {[x[1] for x in iter_folders]}")
    print(f"Output: {OUTPUT_BASE}\n")

    judge     = PromptJudge()
    checker   = LevelIntegrityChecker()
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path_grid = load_path_grid(PATH_FILE, height=14)
    all_res   = []

    for iter_path, iter_num in iter_folders:
        iter_name = f"iteration_{iter_num}"
        out_dir   = os.path.join(OUTPUT_BASE, iter_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n{'='*60}\n[{iter_name}] Loading from {iter_path}")

        try:
            mario_lm = MarioLM(lm_path=iter_path, tokenizer_path=TOKENIZER_PATH)
            mario_lm = mario_lm.to(device)
            mario_lm.eval()
        except Exception as e:
            print(f"  ERROR loading model: {e}"); continue

        levels = []
        for i in range(GENERATE_COUNT):
            prompt = (f"{random.choice(QUANTIFIERS)} pipes, "
                      f"{random.choice(QUANTIFIERS)} enemies, "
                      f"{random.choice(QUANTIFIERS)} blocks, "
                      f"{random.choice(QUANTIFIERS)} coins,  "
                      f"{random.choice(QUANTIFIERS)} rect, "
                      f"{random.choice(ELEVATIONS)} elevation")
            try:
                gen = mario_lm.sample(
                    path_grid=path_grid,
                    prompts=[prompt],
                    temperature=1.0,
                    use_tqdm=False,
                    level_height=14,
                )
                level_str = '\n'.join(gen.level)
            except Exception as e:
                print(f"  Gen {i} failed: {e}"); level_str = ""; gen = None

            levels.append((level_str, prompt))
            with open(os.path.join(out_dir, f"generated_level_{i}.txt"), 'w', encoding='utf-8') as f:
                f.write(level_str + f'\n\n使用的PROMPT: "{prompt}"')
            if level_str and gen is not None and hasattr(gen, 'img') and gen.img is not None:
                try: gen.img.save(os.path.join(out_dir, f"generated_level_{i}.png"))
                except Exception: pass
            del gen; gen = None  # 釋放 SampleOutput（含 PIL Image / tensor）
            print(f"  [{i+1}/{GENERATE_COUNT}] {prompt[:55]}")

        print("  Running prompt check...")
        ps = run_prompt_check(out_dir, levels, judge)

        print("  Running integrity check...")
        p_int, r_int, i_overall = run_integrity_check(out_dir, checker)

        accs = {k: (ps[k]["correct"]/ps[k]["total"]*100) if ps[k]["total"]>0 else 0.0
                for k in CATEGORIES}
        print(f"  Prompt={np.mean(list(accs.values())):.1f}%  Enemy={accs['enemy']:.1f}%  "
              f"P-Int={p_int:.1f}%  R-Int={r_int:.1f}%")

        all_res.append({"name": iter_name, "prompt_stats": ps,
                        "pipe_int": p_int, "rect_int": r_int, "int_overall": i_overall})
        del levels, mario_lm
        torch.cuda.empty_cache()
        gc.collect()

    write_overall_summary(all_res)
    print(f"\n[Done] All results saved to {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
