# Bootstrapping An RL Policy From LLM Completions

The intended loop is:

```text
sample task/state
  -> ask LLM for a primitive policy JSON
  -> validate and execute policy in sim/harness
  -> keep successful traces and informative failures
  -> train a neural policy by behavior cloning
  -> continue with RL using the same primitive action space
```

The LLM should not be in the inner control loop. It is a data generator and repair
mechanism. The learned policy should eventually map observations directly to primitive
actions.

## Action Space

Use the primitive vocabulary as a parameterized action space:

```text
action = {
  primitive_id: discrete,
  args: continuous vector,
  duration_s: continuous
}
```

For the current toy harness, `dataset_from_runs.py` encodes each step as:

```text
[primitive_id, x, y, z, width, force, duration_s]
```

Unused parameters are zero. This is simple enough for behavior cloning, and it can later
be split into a discrete primitive head plus continuous parameter heads.

## Training Stages

1. **LLM demonstrations**
   Generate many policy JSON files from randomized setups. Execute each policy with
   `primitive_policy_runner.py`. Keep successful runs as demonstration data.

2. **Behavior cloning warm start**
   Train a policy to predict the next primitive and parameters from the current
   observation. This gives RL a competent initial policy instead of starting from random
   exploration.

3. **RL fine-tuning**
   Use the same primitive executor as the environment action interface. Reward should
   include sparse task success plus shaped terms such as distance to cube, grasp acquired,
   and final cube height.

4. **DAgger-style repair**
   When the neural policy fails in new states, send the failed state/trace back to the LLM
   for a corrected primitive schedule. Add corrected traces to the dataset and retrain.

## Data Policy

Keep three datasets separate:

- `expert`: successful LLM or hand-authored schedules.
- `repair`: LLM corrections after failed neural rollouts.
- `failure`: failed schedules with diagnostics, useful for critics or preference models.

Do not silently mix invalid completions into expert data. Validate schema and success
checks first.
