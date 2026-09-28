# Procedural Graphs Deep Dive

*Giving an LLM agent an explicit, editable map of "how this kind of task is done" — injecting the relevant corner of that map into its prompt at every step, and letting a second LLM redraw the map from the agent's own successes and failures.*

**The walkthrough itself is the notebook, [`walkthrough.ipynb`](walkthrough.ipynb).** It opens with the conceptual map (roles, how a graph is developed and maintained, and how the refiner is supervised) and then exercises every mechanism with a cell on the HotpotQA scenario: the graph data structure and its two-hop neighbourhood, the environment and an episode driven by hand, the solver loop, the exact text the guidance model sees and where its advice lands in the solver's prompt, an `EditSet` applied and diffed by hand, and finally a real three-round evolution with its log, rejection memory and timeline. Everything there is produced live; most of it needs no API key.

This file holds what a notebook cannot run cheaply: the long-horizon scenario the paper's headline result comes from, and the experiment that says whether any of this helps.

## An aside on long episodes: EnterpriseArena

HotpotQA episodes are three steps. The paper's headline scenario, [EnterpriseArena](https://arxiv.org/abs/2603.23638), is the opposite: the agent is CFO of a lending company for **132 simulated months**, up to 20 information calls a month and exactly one action that closes the month, with the conversation wiped every month so that only notes persist. Hidden growth surges drain cash; fundraising takes 1–6 months to land. An episode is anywhere from about 130 to 800 steps.

Adapting that *without touching the core* needed three small hooks on `Episode`, and they're worth knowing because they show where the interface bends:

- **`pop_context_reset()`** — the episode can ask the agent to restart the solver's conversation (task prompt + a fresh status message). `agent.py` replaces the message history but keeps the step records and the guidance loop going. This is how "memory resets monthly" is reproduced.
- **`pop_extra_steps()`** — when the episode acts on its own (closing a month because the tool budget ran out), it reports that as a step so trajectories stay complete.
- **`Environment.compact_trajectory()`** — 800 steps don't fit in a refiner prompt, so this environment renders **one line per month** from a structured log, collapsing runs of uneventful months into a single `m<first>-m<last> pass x<n>, cash <from>-><to>, tools=<n>` line. A 132-month episode shrinks from ~180,000 characters to a few thousand.

The graphs that come out look different too. This edge was written by the refiner, not by a person:

```json
{"src": "check_cash_in_bank", "rel": "PROVIDES_INPUT_FOR", "dst": "MonthlyRaiseDecision",
 "condition": "Cash, debt and equity balances for the current month are known.",
 "guidance": "Derive burn = previous month's cash minus current cash, and stressed runway = current cash / (1.5-2x that burn), because burn grows over time as the loan book and originations grow. ...",
 "pitfalls": "Do not extrapolate a flat burn: in these episodes burn accelerated until cash fell several million dollars in a single month, so a runway computed from an average of early months is far too optimistic."}
```

That is a *procedure with a threshold and a reason*, written by the refiner after reading trajectories of companies going bankrupt — and it sits in a JSON file you can read, argue with, and edit.

## Did it actually help?

This is where a reference implementation has to be honest, and where the tooling matters as much as the method.

```bash
uv run pg eval hotpotqa --split test --graphs none,graphs/hotpotqa_expert.json,graphs/evolved/hotpotqa/best.json
uv run python -m analysis.paired_comparison runs/<stamp>-eval-hotpotqa
```

[`evaluate.py`](../src/pg/evaluate.py) runs the same tasks under each graph (`none` = no guidance at all); [`analysis/paired_comparison.py`](paired_comparison.py) then compares every graph with the baseline **paired by task**, because task difficulty varies far more than any effect we're hunting: mean paired difference, a bootstrap confidence interval that resamples tasks, and an exact McNemar test on success.

**HotpotQA, 150 held-out test questions** (`paired_comparison` output, some columns and rows trimmed):

```
graph                              metric   tasks   graph  baseline    diff            95% CI        McNemar p
graphs/hotpotqa_expert.json        success    150   0.780     0.787  -0.007  [-0.053, +0.040]   1.0000 (+6/-7)
graphs/evolved/hotpotqa/best.json  success    150   0.760     0.787  -0.027  [-0.067, +0.007]   0.2891 (+2/-6)
```

Nothing. Not the expert graph, not the evolved one — while costing 3.6x (expert) and 6.4x (evolved) as much as running with no graph. And yet validation *rose* from 0.700 to 0.775 during evolution. Both facts are true, and the second one is a trap: the gate keeps whichever candidate scored highest on 50 questions, so an accepted graph's validation score is biased upward by construction. **A rising validation curve is not evidence; only an untouched test split is.** The likelier story here is simply that the base model already solves these questions about as well as the tools allow, so there's no procedure left to teach.

**EnterpriseArena, 20 held-out seeds, 132 months:**

```
graph                               metric           tasks   graph  baseline     diff            95% CI         McNemar p
graphs/enterprisearena_expert.json  success             20   1.000     0.450   +0.550  [+0.350, +0.750]   0.0010 (+11/-0)
graphs/enterprisearena_expert.json  months_survived     20   132.0      78.5    +53.5  [+32.3, +74.8]
graphs/enterprisearena_expert.json  steps               20   135.7     386.7   -251.0  [-351.7, -152.5]
```

Without a graph the agent survives 9 of 20 companies — it typically notices the cash problem only once the first surge is already under way, too late for a raise that takes months to land. With the expert graph, 20 of 20, in a third of the steps. Here there *is* procedure the model doesn't have, and having it is the difference between bankruptcy and survival.

Put together, that's the practical lesson of the two scenarios: **a procedural graph helps when the task has non-obvious procedure and the model lacks it — and costs you tokens for nothing when it doesn't.**

### Can self-evolution find that procedure on its own?

That was the open question, and the answer is yes for this run. A graph was evolved from the edgeless skeleton on 66-month EnterpriseArena episodes (`pg evolve enterprisearena --init scratch`); OpenRouter credits ran out after three effective rounds, at which point validation survival had gone from 5/20 to 17/20. The graph was then run on the untouched 20 test seeds at the full 132-month horizon, alongside three controls, in one pass:

```
graph                                     survived   mean months   steps/episode   $/episode
none                                          1/20            40             203        0.04
skeleton round_0.json (guidance, no edges)    2/20            45             128        0.06
evolved best.json (3 rounds from scratch)    17/19           122             894        1.01
expert (hand-written)                        20/20           132             135        0.11
```

(One evolved-graph episode was lost to a provider timeout while solvent and is excluded as an error.) Paired by seed, `analysis/paired_comparison.py` gives:

- **Evolved vs none:** +16/−0 seeds flipped to survived, exact McNemar p < 0.0001; +81 months, 95% CI [+64, +95].
- **Evolved vs skeleton:** +15/−0, p = 0.0001. The skeleton runs the same guidance machinery with no procedural content and does no better than nothing (2/20 vs 1/20, p = 1.0), so the gain is in what the graph *says*, not in the extra LLM call.
- **Evolved vs expert:** −2/+0, p = 0.5; valuation $135M vs $161M with a CI that includes zero. Statistically indistinguishable on survival.
- **Transfer:** evolved on 66 months, tested on 132; both bankruptcies were at the first surge (months 32–33), none later.
- **The cost the gate never saw:** the evolved graph prescribes a full monthly routine (check cash, recall notes, forecast, act, note) and runs ~7 steps per month against ~1 for the expert graph's "raise whenever nothing is pending", so it is about 9× more expensive per episode. The validation gate scores survival only; if efficiency should count, it has to be in the score.

Two caveats. This is one evolution run and one evaluation pass: it shows that *this* graph helps, not that evolution reliably produces such graphs (that needs repeated evolve runs, ~$45 each). And the no-graph baseline is noisy: the same 20 seeds gave 9/20 five days earlier and 1/20 here; the effect is far larger than that noise, but small differences in the table above should be read as noise.

### Logging

Set `PG_WANDB_PROJECT` and every `evolve` / `eval` run logs to Weights & Biases ([`tracking.py`](../src/pg/tracking.py)): per-round train/val scores, accept/reject, graph size, tokens and cost; a table of rounds with the refiner's rationale; and an `episodes` table with one row of scalars per episode, which is what paired analysis needs. `--experiment <name>` groups an evolution run with its evals. Guidance text and trajectories are deliberately **not** uploaded — they stay in the local JSONL files, which remain the source of truth.

## Where to change things

| You want to… | Go to |
|---|---|
| Add a scenario | Subclass `Environment` / `Episode` in a new `envs/<name>/`; register it in `envs/__init__.py`. Nothing else changes. |
| Change how advice is phrased | `GUIDANCE_SYSTEM` in `guidance.py` |
| Change what the refiner is told | `REFINER_SYSTEM` and `build_prompt` in `refiner.py` |
| Change neighborhood size / history window | `Config.h`, `Config.w` in `config.py` |
| Change the acceptance rule | the `accepted = …` line in `evolve.py` |
| Swap models | `PG_SOLVER_MODEL`, `PG_GUIDANCE_MODEL`, `PG_REFINER_MODEL` in `.env` |
| Hand-edit a graph | it's JSON — edit it, then `pg eval` it against the original |

## In summary

The mechanics turn out to be small. **Localization** is a string comparison between the last tool name and a node id. **Context selection** is a two-hop breadth-first search. **Injection** is one paragraph appended to the system prompt for a single turn and then discarded. **Evolution** is a loop that shows a strong model the best and worst attempts, takes back a typed list of edits, and keeps them only if a held-out set agrees — with a memory of what didn't work.

The compelling part isn't the accuracy numbers, which depend heavily on whether your task has procedure worth learning. It's that the agent's know-how ends up as an **artifact**: a file you can diff between rounds, review like code, roll back when it regresses, and hand to a colleague — and, for every edit, a written rationale and a validation score explaining why it's there. Prompts and weights give you neither. If agents are going to run long, consequential processes, being able to read what they've learned seems like the right place to start.
