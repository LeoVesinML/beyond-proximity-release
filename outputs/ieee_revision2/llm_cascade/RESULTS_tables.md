### A2 Qwen/Qwen2.5-7B-Instruct (bfloat16, MPS) -- test75

| Condition | k | hit@1 | hit@3 | MRR | MRR@3 | target in C | retained | scored/query | input tok/query | calls/query | median ms | mean ms | median e2e ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (i) exhaustive Qwen, all views | all | 0.627 | 0.973 | 0.796 | 0.793 | 1.000 | 19.40 | 19.40 | 2236.2 | 1.00 | 5682 | 5520 | 5682 |
| (i) exhaustive Qwen, all views | 3 | 0.627 | 0.973 | 0.793 | 0.793 | 1.000 | 3.00 | 19.40 | 2236.2 | 1.00 | 5682 | 5520 | 5682 |
| (ii) flat BGE top-N_q -> Qwen (N_q=|C(q)|) | all | 0.653 | 0.960 | 0.800 | 0.796 | 0.987 | 9.48 | 9.48 | 1089.9 | 1.00 | 2867 | 2685 | 2886 |
| (ii) flat BGE top-N_q -> Qwen (N_q=|C(q)|) | 3 | 0.653 | 0.960 | 0.796 | 0.796 | 0.987 | 2.95 | 9.48 | 1089.9 | 1.00 | 2867 | 2685 | 2886 |
| (iii) Alg. 1 (BGE) C(q) -> Qwen | all | 0.653 | 0.947 | 0.793 | 0.789 | 0.973 | 9.48 | 9.48 | 1090.1 | 1.00 | 2868 | 2649 | 2874 |
| (iii) Alg. 1 (BGE) C(q) -> Qwen | 3 | 0.653 | 0.947 | 0.789 | 0.789 | 0.973 | 2.95 | 9.48 | 1090.1 | 1.00 | 2868 | 2649 | 2874 |
| sens. Alg. 1 max_leaf_proxy C(q) -> Qwen | all | 0.693 | 0.960 | 0.819 | 0.816 | 0.987 | 13.47 | 13.47 | 1549.6 | 1.00 | 3889 | 3812 | 3896 |
| extra: Alg. 1 BGE budget-3 -> Qwen | all | 0.640 | 0.960 | 0.784 | 0.784 | 0.960 | 2.95 | 2.95 | 337.6 | 1.00 | 846 | 861 | 860 |
| extra: flat BGE top-3 -> Qwen | all | 0.640 | 0.960 | 0.784 | 0.784 | 0.960 | 3.00 | 3.00 | 344.7 | 1.00 | 802 | 824 | 814 |
| (iv') LLM-only two-branch gate, rerun | all | 0.653 | 0.880 | 0.757 | 0.753 | 0.907 | 11.15 | 11.15 | 1767.2 | 2.00 | 4338 | 4351 | 4338 |

Paired bootstrap (10,000 resamples, seed 20260901), estimate [95% CI]:

| Comparison | hit@1 | hit@3 | MRR |
|---|---:|---:|---:|
| alg1_bge__full - exhaustive__full | +0.027 [-0.040, +0.107] | -0.027 [-0.067, +0.000] | -0.003 [-0.045, +0.039] |
| alg1_bge__full - flat_bge_topN_matched__full | +0.000 [-0.067, +0.067] | -0.013 [-0.040, +0.000] | -0.007 [-0.043, +0.029] |
| alg1_bge__top3 - exhaustive__top3 | +0.027 [-0.040, +0.107] | -0.027 [-0.067, +0.000] | -0.004 [-0.047, +0.038] |
| alg1_bge__top3 - flat_bge_topN_matched__top3 | +0.000 [-0.067, +0.067] | -0.013 [-0.040, +0.000] | -0.007 [-0.042, +0.029] |
| alg1_bge_budget3__full - flat_bge_top3__full | +0.000 [-0.040, +0.040] | +0.000 [+0.000, +0.000] | +0.000 [-0.020, +0.020] |
| alg1_bge__full - gate_llm__full | +0.000 [-0.080, +0.080] | +0.067 [-0.013, +0.147] | +0.036 [-0.033, +0.111] |

### A2 Qwen/Qwen2.5-7B-Instruct (bfloat16, MPS) -- all125

| Condition | k | hit@1 | hit@3 | MRR | MRR@3 | target in C | retained | scored/query | input tok/query | calls/query | median ms | mean ms | median e2e ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| (i) exhaustive Qwen, all views | all | 0.712 | 0.968 | 0.835 | 0.831 | 1.000 | 19.40 | 19.40 | 2235.1 | 1.00 | 5651 | 5514 | 5651 |
| (i) exhaustive Qwen, all views | 3 | 0.712 | 0.968 | 0.831 | 0.831 | 1.000 | 3.00 | 19.40 | 2235.1 | 1.00 | 5651 | 5514 | 5651 |
| (ii) flat BGE top-N_q -> Qwen (N_q=|C(q)|) | all | 0.720 | 0.952 | 0.832 | 0.827 | 0.984 | 9.42 | 9.42 | 1080.4 | 1.00 | 2836 | 2695 | 2844 |
| (ii) flat BGE top-N_q -> Qwen (N_q=|C(q)|) | 3 | 0.720 | 0.952 | 0.827 | 0.827 | 0.984 | 2.94 | 9.42 | 1080.4 | 1.00 | 2836 | 2695 | 2844 |
| (iii) Alg. 1 (BGE) C(q) -> Qwen | all | 0.720 | 0.936 | 0.824 | 0.819 | 0.968 | 9.42 | 9.42 | 1080.6 | 1.00 | 2732 | 2615 | 2752 |
| (iii) Alg. 1 (BGE) C(q) -> Qwen | 3 | 0.720 | 0.936 | 0.819 | 0.819 | 0.968 | 2.94 | 9.42 | 1080.6 | 1.00 | 2732 | 2615 | 2752 |
| sens. Alg. 1 max_leaf_proxy C(q) -> Qwen | all | 0.752 | 0.952 | 0.845 | 0.840 | 0.984 | 13.18 | 13.18 | 1516.4 | 1.00 | 3736 | 3728 | 3744 |
| extra: Alg. 1 BGE budget-3 -> Qwen | all | 0.688 | 0.920 | 0.793 | 0.793 | 0.920 | 2.94 | 2.94 | 336.2 | 1.00 | 847 | 855 | 860 |
| extra: flat BGE top-3 -> Qwen | all | 0.696 | 0.928 | 0.801 | 0.801 | 0.928 | 3.00 | 3.00 | 344.5 | 1.00 | 804 | 824 | 815 |
| (iv') LLM-only two-branch gate, rerun | all | 0.704 | 0.872 | 0.784 | 0.777 | 0.912 | 11.11 | 11.11 | 1762.6 | 2.00 | 4354 | 4464 | 4354 |

Paired bootstrap (10,000 resamples, seed 20260901), estimate [95% CI]:

| Comparison | hit@1 | hit@3 | MRR |
|---|---:|---:|---:|
| alg1_bge__full - exhaustive__full | +0.008 [-0.040, +0.064] | -0.032 [-0.064, -0.008] | -0.011 [-0.045, +0.021] |
| alg1_bge__full - flat_bge_topN_matched__full | +0.000 [-0.048, +0.040] | -0.016 [-0.040, +0.000] | -0.008 [-0.038, +0.018] |
| alg1_bge__top3 - exhaustive__top3 | +0.008 [-0.040, +0.064] | -0.032 [-0.064, -0.008] | -0.012 [-0.045, +0.020] |
| alg1_bge__top3 - flat_bge_topN_matched__top3 | +0.000 [-0.048, +0.040] | -0.016 [-0.040, +0.000] | -0.008 [-0.037, +0.019] |
| alg1_bge_budget3__full - flat_bge_top3__full | -0.008 [-0.048, +0.024] | -0.008 [-0.024, +0.000] | -0.008 [-0.032, +0.012] |
| alg1_bge__full - gate_llm__full | +0.016 [-0.048, +0.080] | +0.064 [+0.008, +0.120] | +0.040 [-0.013, +0.096] |

### (iv) Saved Qwen2.5-1.5B traces (CPU fp32, Windows) restricted to test75

| Method | hit@1 | hit@3 | MRR@3 | views checked | input tok/query | calls/query | median ms (CPU) | mean ms (CPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_instruction | 0.733 | 0.907 | 0.807 | 19.40 | 2236.2 | 1.00 | 20607 | 20340 |
| graph_instruction | 0.667 | 0.867 | 0.744 | 10.80 | 1727.8 | 2.00 | 17224 | 17172 |

graph - flat: hit@1 -0.067 [-0.147, +0.013]; hit@3 -0.040 [-0.133, +0.040]; MRR@3 -0.062 [-0.131, -0.000]

### (iv) Saved Qwen2.5-1.5B traces (CPU fp32, Windows) restricted to all125

| Method | hit@1 | hit@3 | MRR@3 | views checked | input tok/query | calls/query | median ms (CPU) | mean ms (CPU) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| flat_instruction | 0.736 | 0.912 | 0.812 | 19.40 | 2235.1 | 1.00 | 20679 | 20236 |
| graph_instruction | 0.672 | 0.856 | 0.748 | 10.90 | 1738.2 | 2.00 | 17459 | 17261 |

graph - flat: hit@1 -0.064 [-0.128, +0.000]; hit@3 -0.056 [-0.120, +0.008]; MRR@3 -0.064 [-0.120, -0.013]

### B10 answer selection, Qwen/Qwen2.5-7B-Instruct (bfloat16, MPS) -- test75

| Context | accuracy | valid ID rate | target in context | records | input tok/query | median ms | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_records | 0.587 | 0.867 | 1.000 | 19.40 | 1343.1 | 3485 | 3568 |
| alg1_bge_top3 | 0.720 | 0.973 | 0.960 | 2.95 | 261.8 | 931 | 954 |
| flat_bge_top3 | 0.707 | 0.960 | 0.960 | 3.00 | 266.2 | 907 | 943 |
| all_records__constrained | 0.613 | 1.000 | 1.000 | 19.40 | 1343.1 | 3404 | 3364 |
| alg1_bge_top3__constrained | 0.733 | 1.000 | 0.960 | 2.95 | 261.8 | 899 | 923 |
| flat_bge_top3__constrained | 0.733 | 1.000 | 0.960 | 3.00 | 266.2 | 905 | 933 |

| Comparison | accuracy difference [95% CI] |
|---|---:|
| alg1_bge_top3 - all_records | +0.133 [+0.027, +0.240] |
| alg1_bge_top3 - flat_bge_top3 | +0.013 [-0.027, +0.067] |
| flat_bge_top3 - all_records | +0.120 [+0.027, +0.213] |
| alg1_bge_top3__constrained - all_records__constrained | +0.120 [+0.013, +0.227] |
| alg1_bge_top3__constrained - flat_bge_top3__constrained | +0.000 [-0.040, +0.040] |
| flat_bge_top3__constrained - all_records__constrained | +0.120 [+0.013, +0.227] |

### B10 answer selection, Qwen/Qwen2.5-7B-Instruct (bfloat16, MPS) -- all125

| Context | accuracy | valid ID rate | target in context | records | input tok/query | median ms | mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_records | 0.584 | 0.888 | 1.000 | 19.40 | 1343.1 | 3485 | 3513 |
| alg1_bge_top3 | 0.744 | 0.976 | 0.920 | 2.94 | 261.0 | 933 | 950 |
| flat_bge_top3 | 0.720 | 0.968 | 0.928 | 3.00 | 266.1 | 906 | 930 |
| all_records__constrained | 0.624 | 1.000 | 1.000 | 19.40 | 1343.1 | 3409 | 3371 |
| alg1_bge_top3__constrained | 0.752 | 1.000 | 0.920 | 2.94 | 261.0 | 899 | 913 |
| flat_bge_top3__constrained | 0.744 | 1.000 | 0.928 | 3.00 | 266.1 | 902 | 921 |

| Comparison | accuracy difference [95% CI] |
|---|---:|
| alg1_bge_top3 - all_records | +0.160 [+0.072, +0.248] |
| alg1_bge_top3 - flat_bge_top3 | +0.024 [-0.016, +0.064] |
| flat_bge_top3 - all_records | +0.136 [+0.056, +0.216] |
| alg1_bge_top3__constrained - all_records__constrained | +0.128 [+0.032, +0.224] |
| alg1_bge_top3__constrained - flat_bge_top3__constrained | +0.008 [-0.024, +0.048] |
| flat_bge_top3__constrained - all_records__constrained | +0.120 [+0.032, +0.208] |

