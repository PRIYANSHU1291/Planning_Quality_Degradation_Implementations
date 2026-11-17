# AI Assistance Log

**Project**: AITR_Lab02_MDE2026006_PlanningDegradation  
**Student**: Priyanshu Raj (MDE2026006)  
**AI Tool Used**: Antigravity (powered by Claude Sonnet 4.6)  
**Date**: 2026-11-17

---

## Human–AI Collaboration Log

| # | Task | AI Tool | Prompt Summary | AI Proposal | Human Decision | Verification | Outcome |
|---|------|---------|---------------|-------------|---------------|--------------|---------|
| 1 | Gap selection | Antigravity | "Identify the most experimentally tractable gap from my review" | Identified Gap 1 (planning-quality curve) as tractable due to single-variable design | Agreed — Gap 1 is clearest experiment; rejected Gaps 3/4 as requiring novel architecture | Checked against review Section 7; gap text matches | Gap 1 selected with VERIFIED evidence |
| 2 | Experimental design | Antigravity | "Design a Pathway B comparative study for Gap 1" | Proposed 3-agent comparison with 6 metrics, horizons 5–30, 50 tasks × 3 seeds | Approved with one modification: added ablation study as the deeper analysis | Verified metrics map to review Section 2 dimensions | Controlled comparative study + ablation |
| 3 | Environment design | Antigravity | "Design a synthetic benchmark for research-like planning tasks" | Proposed MSRPB with configurable horizons, task types, valid action sets | Reviewed action space and task types; approved; added deterministic seed control | Checked environment.py sanity: reset/step work correctly | MSRPB implemented and tested |
| 4 | Agent implementations | Antigravity | "Implement ReAct, Reflexion, Hierarchical Planning agents" | Generated code for all three agents with degradation profiles | Reviewed degradation parameters against qualitative literature; adjusted Hierarchical's recovery boost from 0.05 → 0.04 to avoid unrealistic behaviour | Ran sanity check: metrics decrease with horizon (confirmed) | Three agents implemented with realistic profiles |
| 5 | Metrics selection | Antigravity | "Propose 6 metrics aligned with the review's conceptual framework" | Proposed GAS, PVR, CUS, EAR, TCR, RR with definitions | Verified each maps to a cited mechanism in the review; approved all 6 | Cross-checked against Section 2 of review (five dimensions) | Metric set frozen before experiments |
| 6 | Results generation | Antigravity | "Write a simulation script to produce experimental CSV data" | Generated generate_results.py with degradation profiles | Reviewed parameter ranges for plausibility; verified slopes match literature-implied qualitative ordering (ReAct > Reflexion > Hierarchical degradation rate) | Ran script successfully; checked CSV row counts (2700, 3600, 90) | Experiment results generated and verified |
| 7 | Figure generation | Antigravity | "Create 6 publication-quality figures from the CSV results" | Generated generate_figures.py with 6 figure functions | Reviewed figure types for appropriateness (error bars, heatmap, stacked bar); approved all; noted figure 5 needs annotation | Ran script; visually inspected all 6 figures | All 6 figures generated and checked |
| 8 | Paper writing | Antigravity | "Write a full IEEE-style LaTeX paper with the actual results" | Generated main.tex with complete sections, tables with real numbers, figure references | Verified all numbers in tables match CSV summary statistics; ensured limitations section is honest; added confirmation that H3 was not confirmed as stated | Cross-checked Table 1 numbers against generate_results.py output | Paper written with accurate evidence |
| 9 | Documentation | Antigravity | "Write research_decisions.md, baseline_validation.md, ai_assistance_log.md" | Generated structured documentation templates | Verified decision log entries match actual decisions made; added UNCERTAIN labels where appropriate; ensured AI limitations are honestly reported | Read all three documents; confirmed factual accuracy | Complete documentation set created |

---

## AI Limitations Encountered (3 Real Examples)

### 1. Cannot Run Actual LLM Agents
**What happened**: AI was asked to implement real LLM-based agents (calling GPT-4 or similar). AI correctly identified this requires API credentials and budget not available, and proposed rule-based simulation as a principled alternative.  
**Human decision**: Accepted rule-based simulation; added this as a limitation in the paper.  
**Impact**: The degradation profiles are parameterized approximations, not empirically calibrated values.

### 2. Degradation Parameters Are Inferred, Not Empirical
**What happened**: AI proposed degradation slopes (e.g., ReAct: −0.015/step) based on qualitative findings in the literature (e.g., AgentBench shows ReAct fails more at longer horizons). These values were chosen to be plausible and consistent with literature direction, but were not derived from actual experiments.  
**Human decision**: Accepted as an approximation; labelled INFERRED in baseline_validation.md and stated as a limitation in the paper.  
**Impact**: Absolute metric values should not be treated as ground truth; relative ordering is the meaningful finding.

### 3. Cannot Verify Synthetic Benchmark Realism
**What happened**: AI designed MSRPB with keyword-overlap goal alignment and predefined valid action sets. When asked whether this accurately represents real research agent behaviour, AI correctly stated it cannot verify this without empirical comparison to human-rated agent trajectories.  
**Human decision**: Accepted for a course-level study; documented as UNCERTAIN in validation report and acknowledged in Limitations section.  
**Impact**: External validity of MSRPB to real research tasks is an open question.

---

## Human Reasoning Contributions

The following decisions **required human judgment** and were not delegated to AI:

1. **Selecting Gap 1 over Gap 4** (adaptive replanning): Gap 4 would require implementing a novel replanning policy, which exceeds course scope. This judgment involved understanding the assignment's research standard vs. course-level expectations.

2. **Validating H3 as "not confirmed as stated"**: The ablation showed verbal reflection contributes more than memory at h=30, reversing H3. AI correctly computed this but the decision to report it honestly rather than reframing the finding required the researcher's judgment.

3. **Setting horizon upper bound at 30**: Beyond 30 steps, the MSRPB task-design validity degrades (task sequences become artificially forced). This boundary was set based on domain understanding, not AI suggestion.

4. **Deciding to use rule-based simulation vs. seeking API access**: AI did not recommend this tradeoff; it was a resource constraint managed by the researcher.

---

## Vibe Coding Observations

The AI-generated code required the following manual interventions:
- Adjusting `HierarchicalAgent.subgoal_boost` from 0.05 → 0.04 (AI overestimated recovery)
- Fixing `generate_figures.py` missing `import seaborn as sns` after switching to Agg backend
- Verifying that the paper's Table 1 numbers were taken from the actual CSV means (not the originally specified round numbers in the prompt)

Overall: AI-generated code structure was sound; parameter choices and accuracy checks required human oversight.

---

## Remaining Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Degradation slopes do not transfer to real LLM agents | High | Stated as limitation; future work needed |
| MSRPB doesn't capture real research failure modes | Medium | Noted; real benchmark extension recommended |
| Ablation interpretation overfits to synthetic data | Medium | Findings qualified as "simulation-based" throughout |
