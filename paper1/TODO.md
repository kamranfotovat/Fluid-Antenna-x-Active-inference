# paper1 — open items

Running list so nothing is lost between sessions. Newest block first.

## Done 2026-08-28

- **Genie now defined as a bound, not a scheme.** Section IV-A states all three
  exemptions (perfect knowledge of all N ports, no pilots spent to acquire it,
  no reconfiguration cost) and says explicitly that no causal pilot-limited
  policy can attain it. Both figure legends carry the short form "no pilot or
  switching cost", and both captions repeat it.
- **Fig. 3 rate-greedy callout removed.** The 66.0% number moved into the
  caption alongside the 83.7% operating point, so nothing was lost.
- **Figs. 2 and 3 restyled** to a conventional journal look: full box frame,
  dotted grid, framed legend, filled markers, black dashed ceiling. No tinted
  callout boxes, curved leader arrows, or shaded gap regions. Both are at
  figsize (3.5, 2.02); the height was the lever used to stay inside five pages
  after the genie prose was added.

## Carried over

- **Fig. 2 baselines.** Agreed plan: overlay CUCB bandit (matches cited Zou WCL
  2024), round-robin `run_naive`, and random-partial on Fig. 2's axes instead of
  adding a fourth figure. All three exist in `sim_version3/` but run fully
  digital at m = M through `run_aif`; they need porting to `run_st` with
  `m_sense` before their numbers can sit beside Table I. DRL baseline dropped —
  needs training plus a full-CSI fairness gate.
- **§II length.** Open question for Kian: relax from 75% of page 2 to ~60% to
  free the ~0.7 column Kamran's extra references need. Intro (533 w) + §II
  (618 w) is 35% of the body.
- **Table I block (a) vs Fig. 2** are the same sweep. Duplication is acceptable
  only until Fig. 2 gains the baselines; if that slips, block (a) is the first
  cut (~0.15 col).
- **refs.bib**: every entry still carries `% VERIFY`; `drl_switching` has an
  empty author field. Kamran is adding roughly six more references.
- Before submission: delete `\todo`/`\note` macros, produce `main.bbl` for arXiv.
- **arXiv endorsement** for eess.SP still needed.
