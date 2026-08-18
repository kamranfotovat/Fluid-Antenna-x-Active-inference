# OP_V2 results (no figures)

**Operating point:** N=441 (21x21), 2x2 lambda, K=3, M=8, d_min=0.5 lambda, 15 dB, rho=0.9, beta_w=0.25
**Monte-Carlo:** 6 seeds, T=40 slots, second-half averaging. All selection methods honor the >= 0.5 lambda min-spacing constraint.

### T1 — Baselines at the default point (M=8, 2% of ports, beta_w=0.25, d_min=0.5)

| method | rate | % genie | objective | switch/slot |
|---|---:|---:|---:|---:|
| genie (full CSI) | 20.63 | 100% | 6.76 | 13.88 |
| AIF (ours) | 17.32 | 84% | 17.26 | 0.00 |
| AIF (d_min OFF) | 17.36 | 84% | 17.29 | 0.00 |
| naive (no inference) | 17.58 | 85% | 14.83 | 2.82 |
| random (partial CSI) | 16.84 | 82% | 1.47 | 15.27 |

### T2 — Observation budget (M sweep, d_min=0.5, beta_w=0.25)

| M | % ports | genie rate | AIF rate | % genie | AIF obj | naive rate |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.9% | 16.74 | 12.70 | 76% | 12.42 | 12.01 |
| 6 | 1.4% | 19.22 | 15.39 | 80% | 15.49 | 15.16 |
| 8 | 1.8% | 20.63 | 17.32 | 84% | 17.26 | 17.58 |
| 10 | 2.3% | 21.55 | 18.50 | 86% | 18.35 | 18.61 |
| 12 | 2.7% | 22.20 | 19.49 | 88% | 19.29 | 19.81 |

### T3 — Exploration weight (beta_w sweep, M=8, d_min=0.5) — rate vs switching Pareto

| beta_w | rate | % genie | objective | switch/slot |
|---:|---:|---:|---:|---:|
| 0 | 16.55 | 80% | 16.79 | 0.00 |
| 0.1 | 17.32 | 84% | 17.26 | 0.00 |
| 0.25 | 17.32 | 84% | 17.26 | 0.00 |
| 0.5 | 18.21 | 88% | 13.91 | 4.27 |
| 1 | 18.10 | 88% | 5.91 | 12.13 |

_generated in 1573s_
