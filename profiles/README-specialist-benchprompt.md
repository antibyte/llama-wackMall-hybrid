# Specialist expert profile (bench prompt)

- **File:** `specialist-profile.csv`
- **SHA-256:** `e246161c7c516efe0a911a0165d5c5a934d82dbf215b535929defec478c7bdf4`
- **Source:** session usage on `phase1-placement-s33-mtp2` prompt, 512 decode tokens
- **Mean top-33 mass coverage:** 72.87% (general corpus profile: 51.64%)
- **Measured (2026-08-09, skip-sentinel winner stack, S=33, MTP-2, q4_0/q4_0):**
  - 3×256: +11.64% decode vs general
  - 3×2k: +2.00% decode vs general
- **Caveat:** prompt-matched / overfit. For multi-domain chat, prefer the general corpus profile or rebuild from real session usage CSVs.

Rebuild:

```bash
# after SAVE_EXPERT_USAGE=1 collection run(s)
python3 tools/aggregate_expert_profiles.py \
  --input path/to/run.usage.csv --weight 1 \
  --model "$HOME/models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  --output specialist-profile.csv \
  --normalization per-layer --scale 10000 --top-k 33
```
