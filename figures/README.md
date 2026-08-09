# Paper figures

Operating point unless noted: N=25 ports (5x5), K=3 users, M=5 activated (20% observation budget), 15 dB, rho=0.9, beta_w=0.25, eta_sw=1, observe-then-precode.

## Headline / results
- **figA_observation_budget.png** - performance vs observation budget M/N (the headline: AIF tracks a high fraction of the genie while measuring a fraction of ports).
- **figR_results_baselines.png** - AIF vs genie / naive / random: rate is comparable but AIF wins the switching-aware objective and barely moves the antenna.
- **figC_protocol_doppler.png** - observe-then-precode reaches ~80-89% of genie and is robust to Doppler; predict-then-act (ablation) collapses.
- **figD_exploration_weight.png** - exploration-weight sweep: sweet spot around beta_w=0.1-0.25.
- **figE_learning_mismatch.png** - learning R from data adapts to a non-Jakes channel (objection-proofing); learned R_hat matches the oracle.
- **figLC_closed_loop_learning.png** - closed-loop learning curve + rate/objective bars.

## Diagnostics (verification plots)
- diag_step0_channel.png - channel generator (Jakes R + AR(1)).
- diag_step3_belief_calibration.png - Kalman belief calibration + CSI aging.
- diag_step4_efe_terms.png - EFE terms (submodular epistemic, conservative pragmatic).
- diag_step5_greedy_optimality.png - greedy vs exhaustive + latency.

Regenerate with `python sim/make_paper_figures.py` (after running the verify_*.py that produce the step plots).
