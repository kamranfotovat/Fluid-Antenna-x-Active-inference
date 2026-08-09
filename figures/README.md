# Paper figures

**All figures generated at ONE operating point:** N=25, K=3, M=5 (20% obs), 15 dB, sigma_e^2=1e-3, rho=0.9, beta_w=0.25, observe-then-precode.

Headline: AIF gets ~84% of the genie's rate while measuring only 20% of ports, and BEATS the genie on the switching-aware objective (it barely moves the antenna).

## Results figures
- **figA_observation_budget.png** - performance vs observation budget M/N (headline).
- **figR_results_baselines.png** - AIF vs genie/naive/random (rate + objective + switching).
- **figC_protocol_doppler.png** - observe-then-precode vs predict-then-act vs Doppler.
- **figD_exploration_weight.png** - beta_w sweep (sweet spot ~0.1-0.25).
- **figLC_closed_loop_learning.png** - closed-loop learning curve + bars.
- **figE_learning_mismatch.png** - learning R adapts to a non-Jakes channel.

## Diagnostics (mechanism-verification plots)
- diag_step0_channel / diag_step3_belief_calibration / diag_step4_efe_terms / diag_step5_greedy_optimality.

Regenerate: `python sim/make_paper_figures.py`.
