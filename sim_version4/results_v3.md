# OP_V3 hybrid-beamforming results (no figures)

**Operating point:** N=441 (21x21), 2x2 lambda, K=3, M=10, n_rf=6, d_min=off, 15 dB, rho=0.9, beta_w=0.25
**Monte-Carlo:** 6 seeds, T=40 slots, second-half averaging. Fully-connected unit-modulus analog network; hybrid factorized from the belief-based precoder by coordinate-descent AltMin.

### H1 — RF-chain sweep (M=10 active, K=3, 2K=6, beta_w=0.25)

Rate (bits/slot) vs number of RF chains n_rf. 'digital' = one chain per active port (n_rf=M=10). n_rf >= 2K=6 recovers digital; the loss concentrates at n_rf=K.

| n_rf | note | genie | AIF | naive | AIF % of digital |
|---:|---|---:|---:|---:|---:|
| 10 (digital) | digital | 22.11 | 18.63 | 16.30 | 100% |
| 3 | = K | 21.20 | 16.83 | 14.57 | 90% |
| 4 |  | 22.09 | 18.58 | 16.17 | 100% |
| 5 |  | 22.11 | 18.63 | 16.30 | 100% |
| 6 | = 2K | 22.11 | 18.63 | 16.30 | 100% |
| 8 |  | 22.11 | 18.63 | 16.30 | 100% |
| 10 |  | 22.11 | 18.63 | 16.30 | 100% |

### H2 — Joint budget: AIF rate for (M active ports) x (n_rf RF chains)

| M \ n_rf | 3 | 4 | 6 | 8 | 10 |
|---:|---:|---:|---:|---:|---:|
| 6 | 14.99 | 15.31 | 15.31 | - | - |
| 8 | 16.29 | 17.35 | 17.36 | 17.36 | - |
| 10 | 16.83 | 18.58 | 18.63 | 18.63 | 18.63 |

_generated in 759s_
