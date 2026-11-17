# One-Page Research Brief (Gate G1)

**Student**: Priyanshu Raj | **Enrollment**: MDE2026006 | **Date**: 2026-11-17

---

| Field | Information |
|-------|-------------|
| **Review topic and key papers** | Planning-Quality Degradation in Long-Horizon Autonomous Research Agents. Key references: ReAct (Yao et al., 2023); Reflexion (Shinn et al., 2023); Hierarchical Planning (Aghzal et al., 2026; Tan et al., 2026); AgentBench (Liu et al., 2024); WebArena (Zhou et al., 2024); APB (Sun et al., 2026) |
| **Research gap** | **Gap 1** from review Section 7: "The field lacks a widely accepted metric showing how plan quality changes with effective horizon. Future work should report reliability or planning-quality curves rather than one final success rate." Supporting evidence: APB (Sun et al., 2026) diagnoses planning but reports single accuracy numbers; WebArena/AgentBench collapse all trajectory failures to one success/fail bit. |
| **Research question** | How does planning quality degrade across three representative LLM agent architectures (ReAct, Reflexion, Hierarchical Planning) as the number of required planning-action steps increases from 5 to 30, and which architectural mechanisms best preserve plan coherence over long horizons? |
| **Study pathway** | **B — Controlled Comparative Study**: Evaluate three agent architectures under a common protocol with consistent metrics, controlled horizon lengths (5, 10, 15, 20, 25, 30 steps), and fair comparison conditions. Deeper analysis: ablation study on Reflexion's components (memory vs. reflection). |
| **Dataset / measurement setup** | Multi-Step Research Planning Benchmark (MSRPB) — synthetic, self-contained benchmark. No external data download required. Tasks: research-like sequences (search → evaluate → hypothesize → design → analyze → synthesize). 50 tasks per horizon × 3 seeds = 150 evaluations per agent per horizon. Deterministic given seed. |
| **Baseline / comparators** | **Baseline**: ReAct (Yao et al., 2023) — standard reason-act loop, no memory, no reflection. **Comparators**: (1) Reflexion (Shinn et al., 2023) — adds verbal reflection + episodic memory; (2) Hierarchical Planning (Aghzal et al., 2026; Tan et al., 2026) — coarse-to-fine subgoal decomposition + selective context compression. Code: custom rule-based implementations. |
| **Primary metric and control** | **Primary**: Goal Alignment Score (GAS) — keyword overlap between action reasoning and original goal, 0–1. **Control**: horizon length (independent variable). **Held constant**: task difficulty, evaluation protocol, random seeds, action space definition. |
| **Computing and risks** | Hardware: CPU-only (Python 3.11, Windows). Runtime: ~15 minutes total. Risks: (1) Synthetic environment may not capture real LLM failure modes; (2) degradation parameters are literature-inferred approximations; (3) keyword-overlap GAS is a simplified proxy. Fallback: if simulation is judged insufficient, can extend to real LLM API calls given access. |
| **First executable milestone** | ✅ **COMPLETED** — environment.py runs successfully, ReAct agent produces valid trajectories on 5-step tasks, all 6 metrics compute correctly, experiment CSV contains 2,700 rows, 6 figures generated. |
