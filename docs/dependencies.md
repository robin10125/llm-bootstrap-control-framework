# Dependencies and portability

The experiment uses one repository-local virtual environment: `.venv/`. Do not invoke a venv from
a sibling experiment directory.

## Python dependencies

The base editable install declares NumPy, JAX, MuJoCo/MJX, Flax, and Optax:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

Optional live Anthropic calls use `.venv/bin/python -m pip install -e '.[llm]'`. The `codex` and
`claude-code` backends invoke their respective command-line tools and require their normal login or
credentials.

For GPU runs, install the JAX CUDA extra that matches the installed NVIDIA driver, for example:

```bash
.venv/bin/python -m pip install -U 'jax[cuda12]'
nvidia-smi
```

CPU-only JAX is sufficient for import checks and small tests, but full MJX/PPO runs are intended for
a GPU. Headless rendering also requires a working EGL/OpenGL stack and `ffmpeg` on `PATH`.

## Repository-owned runtime

The sources formerly loaded from `../bootstrapping` now live in `experiment_runtime/`. The Shadow
Hand XML, meshes, upstream documentation, and license formerly loaded from `../hand-manipulation`
now live in `experiment_runtime/assets/shadow_hand/`. No external source checkout or `PYTHONPATH`
entry is required.

## Validation

```bash
.venv/bin/python -c 'import experiment_runtime, llm_framework, policy_bias_lab'
.venv/bin/python -m pytest
ffmpeg -version
```

Environment construction triggers JAX/XLA initialization and can take several minutes; keep it out
of ordinary import/static checks.
