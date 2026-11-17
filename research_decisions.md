# Research Decisions Log

**Project**: AITR_Lab02_MDE2026006_PlanningDegradation  
**Student**: Priyanshu Raj (MDE2026006)  
**Study**: Planning-Quality Degradation Comparative Study

Evidence labels: **VERIFIED** (from original source / actual execution) | **INFERRED** (plausible, requires further testing) | **UNKNOWN** (missing evidence)

---

## Decision Log

| # | Date | Decision | Reason | Source / Evidence | Status | Next Action |
|---|------|----------|--------|-------------------|--------|-------------|
| D01 | 2026-11-17 | Selected **Gap 1** (no standardized planning-quality curve) over Gaps 2–5 | Gap 1 is most experimentally tractable: varies one independent variable (horizon length), does not require novel memory architecture or new benchmark construction | Review Section 7, Gap 1; APB (Sun et al., 2026) diagnoses planning but reports single accuracy numbers | **VERIFIED** (gap stated in review; APB confirmed by literature) | Confirm no existing work plots quality-vs-horizon curves |
| D02 | 2026-11-17 | Selected **Pathway B** (Controlled Comparative Study) | Pathway B answers the gap directly: comparing architectures under identical conditions with horizon as the independent variable is more informative than replicating one paper | Assignment rubric Pathway B definition; review's recommendation for trajectory-level evaluation | **VERIFIED** | Design controlled protocol |
| D03 | 2026-11-17 | Selected **ReAct, Reflexion, Hierarchical** as comparison architectures | Each represents a distinct mechanism for handling long horizons: no memory (ReAct), reflection+memory (Reflexion), decomposition+context-compression (Hierarchical). Together they span the design space described in the review | Yao et al. (2023) [ReAct]; Shinn et al. (2023) [Reflexion]; Aghzal et al. (2026); Tan et al. (2026) | **VERIFIED** (all four papers are in the 20-paper evidence set) | Implement each from published description |
| D04 | 2026-11-17 | Designed **MSRPB** (synthetic benchmark) rather than using an existing benchmark | WebArena/AgentBench do not support arbitrary horizon control. MSRPB provides: exact horizon control, deterministic evaluation, no API dependencies, free access | AgentBench (Liu et al., 2024) — fixed environments; WebArena (Zhou et al., 2024) — fixed tasks | **INFERRED** (suitability of synthetic environment for calibration is not fully validated) | Document synthetic-environment limitation |
| D05 | 2026-11-17 | Chose **6 metrics** (GAS, PVR, CUS, EAR, TCR, RR) | Directly derived from review's five conceptual dimensions: goal alignment, plan validity, context efficiency, error accumulation, task success, recovery. Each corresponds to a cited mechanism | Review Section 2 (conceptual foundation); APB (Sun et al., 2026); ARC (Yao et al., 2026) | **VERIFIED** (metrics map to review-cited mechanisms) | Freeze metric definitions before viewing results |
| D06 | 2026-11-17 | Set **horizon range 5–30** steps | Below 5: tasks too trivial for meaningful comparison. Above 30: computational cost and task-design validity degrade. The review's 20-paper set covers up to ~40-step trajectories; 30 is a reasonable upper bound | AgentBench (Liu et al., 2024): identifies "long-term" at 10+ steps; WebArena average task ~15 steps | **INFERRED** (exact upper bound is a judgment call) | Justify in paper |
| D07 | 2026-11-17 | Added **Reflexion ablation** (Full / NoMemory / NoReflection / Baseline) as deeper analysis | Reflexion has two distinct components (memory + reflection); separating them tests H3 and provides mechanistic understanding beyond the main comparison | Shinn et al. (2023): identifies both components; review Section 3.4 discusses reflection and memory separately | **VERIFIED** (ablation design is consistent with original paper's component analysis) | Implement ablation mode in ReflexionAgent |
| D08 | 2026-11-17 | **Froze protocol** before viewing results | Research integrity requirement: all metrics, splits, and comparisons defined before observing experimental outcomes | Assignment rubric (Section 2: "freeze evaluation protocol before viewing results"); Prompt 07 | **VERIFIED** | Log protocol freeze timestamp |
| D09 | 2026-11-17 | Used **rule-based simulation** instead of live LLM calls | No API budget; rule-based simulation with literature-calibrated parameters is transparent, reproducible, and sufficient to demonstrate the methodology. Acknowledged as a limitation. | APB (Sun et al., 2026): uses fixed scoring rather than LLM-judged quality in some settings | **INFERRED** (simulation is not equivalent to real LLM behavior; parameters are approximate) | State clearly in limitations |
| D10 | 2026-11-17 | Chose **GAS degradation slope** as primary finding metric | Slope captures the trajectory of decline, not just a point estimate. A steeper slope predicts worse reliability at scale. Directly addresses Gap 1's call for a quality curve | Review Gap 1: "report reliability or planning-quality curves rather than one final success rate" | **VERIFIED** (matches exact language of Gap 1) | Report slope with R² and p-value |

---

## Evidence Ledger Summary

| Claim | Status |
|-------|--------|
| All three architectures degrade significantly with horizon | VERIFIED (statistically, p < 0.05 for all) |
| Hierarchical degrades slowest | VERIFIED (empirical from experiments) |
| Memory > reflection for Reflexion at h=30 | VERIFIED (ablation result, reversed from H3 at short horizons) |
| Synthetic benchmark adequately represents real agent behavior | INFERRED |
| Degradation profiles match real LLM behavior | INFERRED |
| Exact slope values transfer to real systems | UNKNOWN |
