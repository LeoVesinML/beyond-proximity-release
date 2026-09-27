# Instruction-Agent Benchmark

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Queries: 150 total / 125 quality-eligible
- Device: `cpu`

| Method | hit@1 | hit@3 | MRR | Views | Calls | Input tokens | Output tokens | ms/query | Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_instruction | 0.736 | 0.912 | 0.812 | 19.40 | 1.00 | 2221.9 | 19.4 | 20215.0 | 1.000 |
| graph_instruction | 0.672 | 0.856 | 0.748 | 10.59 | 2.00 | 1692.7 | 17.4 | 16927.7 | 1.000 |

- Graph input-token reduction: 23.82%
- Graph views reduction: 45.40%
- Paired hit@1 difference: -0.064 (95% CI [-0.128, 0.0])
- Paired hit@3 difference: -0.056 (95% CI [-0.12, 0.0])

Token values are exact native tokenizer counts for the rendered chat prompts and generated responses.
