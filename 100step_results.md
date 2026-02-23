# 100-Step Results Summary

All runs use `T=1024`, global batch `64`, `num_iterations=100`.

| config | flags | train_loss@100 | val_loss@100 | step_avg@100 | log |
| --- | --- | ---: | ---: | ---: | --- |
| MLP baseline | none | 6.1897 | 6.5421 | 3419.07 ms | `logs/9a8cddb2-f94d-4886-b9b8-cd977ea81df9.txt` |
| SSINF3 weight-matched | `--ssinf3-weight-matched` | 6.2324 | 8.1085 | 15046.16 ms | `logs/38a67bec-0bf6-4bfc-aa02-42796f543794.txt` |
| SSINF3 one-tenth | `--ssinf3-one-tenth` | 6.2275 | 9.4302 | 8527.04 ms | `logs/d10047e9-8f6c-4781-a57e-5af9767da70a.txt` |
| SSINF3 weight-matched optimized (initial grouped LR profile) | `--ssinf3-weight-matched --ssinf3-optimized` | 6.4689 | 8.8487 | 16686.87 ms | `logs/ssinf3_optimized_weightmatched_100step_uncompiled.out` |
| SSINF3 weight-matched optimized (uniform SSINF3 LR=0.02) | `--ssinf3-weight-matched --ssinf3-optimized` | 6.7257 | 8.7247 | 16038.72 ms | `logs/ssinf3_optimized_weightmatched_lr002_100step_uncompiled.out` |
| SSINF3 weight-matched optimized (uniform SSINF3 LR=0.08) | `--ssinf3-weight-matched --ssinf3-optimized` | 7.3702 | 7.9056 | 15952.98 ms | `logs/ssinf3_optimized_weightmatched_lr008_100step_uncompiled.out` |
