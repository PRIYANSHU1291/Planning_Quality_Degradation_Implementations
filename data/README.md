# Data

## Multi-Step Research Planning Benchmark (MSRPB)

This study uses a **synthetic benchmark environment** rather than an external dataset. The environment (`src/environment.py`) generates research-like planning tasks with configurable horizon lengths.

### Why synthetic?

The research question concerns how agent architectures handle increasing trajectory lengths. A controlled synthetic environment allows:
1. Precise control over horizon length
2. Deterministic evaluation given random seeds
3. Reproducibility without external data dependencies
4. Fair comparison across architectures

### Task structure

Each task simulates a research planning workflow:
- **Goal**: A research objective (e.g., "Investigate the effect of X on Y")
- **Steps**: Sequential research operations (search, evaluate, synthesize, verify, decide)
- **Evaluation**: Each action is scored on goal alignment, validity, and context utilization

### Generating tasks

Tasks are generated programmatically:
```python
from src.environment import generate_task_suite
tasks = generate_task_suite(
    horizon_lengths=[5, 10, 15, 20, 25, 30],
    n_tasks_per_horizon=50,
    seed=42
)
```

### License
No external dataset is used. All benchmark code is provided under the project license.
