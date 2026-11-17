# Baseline Validation Report

**Project**: AITR_Lab02_MDE2026006_PlanningDegradation  
**Student**: Priyanshu Raj (MDE2026006)  
**Baseline**: ReAct Agent (Yao et al., 2023)

---

## 1. What Was Validated

The ReAct agent implementation (`src/agents/react_agent.py`) was validated against the published methodology in:

> Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *ICLR 2023*.

---

## 2. Implementation Checklist

| Component | Original (Yao et al., 2023) | Our Implementation | Status |
|-----------|-----------------------------|--------------------|--------|
| Reasoning trace | Verbal reasoning before each action | `_build_reasoning()` generates a reasoning string from observation + goal | **VERIFIED** |
| Action selection | Action chosen after reasoning trace | `_pick_action_type()` selects from valid action set | **VERIFIED** |
| No persistent memory | Only current observation + action history in context | `context_buffer` grows but no episodic summary | **VERIFIED** |
| No reflection mechanism | No self-correction between steps | No reflection code in ReActAgent | **VERIFIED** |
| Sequential execution | One action per step | Single `act()` call per environment step | **VERIFIED** |
| Observation-grounded reasoning | Each reasoning uses current environment state | Reasoning includes `task_type`, `goal`, `valid_actions` from observation | **VERIFIED** |

---

## 3. Discrepancies with Original Paper

| Discrepancy | Type | Impact |
|-------------|------|--------|
| Original uses real LLMs (GPT-3.5/GPT-4); we use rule-based simulation | **Fundamental** — different mechanism | Degradation slopes are parameterized from literature, not empirically measured; cannot guarantee transfer to real LLMs | UNCERTAIN |
| Original evaluates on HotPotQA, FEVER, ALFWorld, WebShop; we use MSRPB (synthetic) | **Benchmark** — different domain | MSRPB is designed to simulate research tasks, not QA or shopping; external validity is limited | UNCERTAIN |
| Degradation profile parameters (slope, noise) are literature-inferred, not empirically calibrated | **Parameterization** — approximate | Relative ordering (ReAct degrades faster than Reflexion/Hierarchical) is VERIFIED as consistent with literature; absolute values are approximate | INFERRED |
| Goal alignment metric uses keyword overlap; original uses task-specific metrics | **Metric** — simplified | Keyword overlap is a valid proxy for semantic relevance but underestimates nuance | UNCERTAIN |

---

## 4. Validation Checks Performed

| Check | Method | Result |
|-------|--------|--------|
| **Monotonic degradation** | GAS should decrease as horizon increases | ✅ PASSED — GAS decreases from 0.835 (h=5) to 0.467 (h=30) |
| **Range validity** | All metrics in [0, 1] | ✅ PASSED — no out-of-range values in experiment_results.csv |
| **Reproducibility** | Same seed → same results | ✅ PASSED — running generate_results.py twice produces identical CSV |
| **Relative ordering** | ReAct degrades faster than Reflexion and Hierarchical | ✅ PASSED — slopes: ReAct=−0.015, Reflexion=−0.009, Hierarchical=−0.003 |
| **Error monotonicity** | Error count should increase with horizon | ✅ PASSED — error_analysis.csv shows increasing counts |
| **Statistical significance** | Degradation slope p < 0.05 | ✅ PASSED — ReAct: p=3.19e-8, Reflexion: p=1.55e-4, Hierarchical: p=0.014 |

---

## 5. Items Requiring Attention

| Item | Status |
|------|--------|
| Exact degradation slopes match real LLM behavior | UNCERTAIN — real calibration requires API access |
| MSRPB tasks representative of real research workflows | UNCERTAIN — synthetic design; not validated against human raters |
| Keyword-overlap goal alignment is well-correlated with semantic alignment | UNCERTAIN — no gold-standard comparison |

---

## 6. Conclusion

The ReAct baseline implementation is consistent with the original paper's design principles (reason-then-act loop, no memory, no reflection) and passes all sanity checks for monotonicity, range validity, reproducibility, and relative ordering. The key caveat is that the rule-based simulation is an approximation; the degradation parameters are inferred rather than empirically calibrated. This limitation is disclosed in the paper's Limitations section and the AI assistance log. The baseline is **approved for use in comparative experiments** with these caveats clearly stated.
