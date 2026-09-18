# agentic-procedural-graphs

A small, readable reference implementation of **Procedural Graphs: Self-Evolving Execution Structures for LLM
Agents** (arXiv 2609.09153) on LangGraph.

An agent's procedural knowledge is stored as an explicit, editable graph of `(procedure, relation, procedure)`
triplets. The method has two halves:

- **Online guidance.** Before each solver turn, the agent is localized in the graph by exact match of its last
  tool call. The 2-hop neighborhood around that node (or the whole graph, before the first call) plus the last 3
  steps goes to a guidance model. That model writes step-level guidance, which is injected into the solver's
  system prompt for that turn only.
- **Offline self-evolution.** Roll out the current graph on a train batch, then give the best and worst
  trajectories to a refiner LLM, which proposes add/delete edits. Validate the candidate on a fixed val set and
  accept it if its score is at least the current graph's. Rejected edits go into a rejection memory that later
  refiner prompts include.

## Setup

```bash
uv sync                      # fetches Python 3.13 (LangGraph 1.2 supports <=3.13)
cp .env.example .env         # then set OPENROUTER_API_KEY
uv run pytest                # unit tests, no network
```

All LLM calls go through OpenRouter (`pg.llm.make_llm`). Defaults: solver and guidance `openai/gpt-5.6-luna`,
refiner `anthropic/claude-opus-5`. Override with `PG_SOLVER_MODEL`, `PG_GUIDANCE_MODEL`, `PG_REFINER_MODEL`, and
set `PG_CONCURRENCY` for the number of parallel rollouts. If the refiner id is not served, try `openai/gpt-5.6-luna-pro`.

## Usage

```bash
# data
uv run pg gen-data finance  --seed 0 --n-train 30 --n-val 20 --n-test 30
uv run pg gen-data hotpotqa --seed 0 --n 300          # downloads hotpotqa/hotpot_qa once, caches JSONL
uv sync --extra enterprisearena && uv run pg gen-data enterprisearena   # fetches + verifies CFO-Env, 20/20/20 seeds + probe

# evaluate a split with no graph, one graph, or several at once
uv run pg eval hotpotqa --split val --n 5 --graphs none,graphs/hotpotqa_expert.json

# self-evolution from a skeleton (Start/End + one node per tool, zero edges) or from an existing graph
uv run pg evolve hotpotqa --init scratch --rounds 2 --batch 5 --val-n 10   # cheap smoke test
uv run pg evolve hotpotqa --init scratch --rounds 5 --batch 10               # validates on the whole val split (50)
uv run pg evolve finance  --init scratch --rounds 5 --batch 10               # whole val split (20)

# comparison
uv run pg eval finance --graphs none,graphs/finance_expert.json,graphs/evolved/finance/best.json
```

`eval` prints mean score, success rate, mean steps, solver vs guidance tokens (the guidance overhead), cost,
and counts of episodes stopped early or errored. Trajectories (including every guidance string) and `metrics.json` go to `runs/<timestamp>-.../`.

`evolve` writes to `graphs/evolved/<env>/`:
- `round_k.json` and `best.json`
- `evolution_log.jsonl`: batch score, val before/after, accepted?, edits, rationale, and cost for each round
- `prompts/round_k.txt`: the exact refiner prompt, including rejection memory
- `trajectories/`

It refuses to start if that directory already holds a run (`evolution_log.jsonl` exists); pass `--out <dir>` to
write elsewhere or `--overwrite` to replace it. A validation pass only counts when at least
`Config.val_min_scored` (80%) of its episodes ran without error: below that the baseline aborts the run, and a
candidate round is logged as no data (graph kept, nothing added to rejection memory).

## What is not in this repository

Four paths are gitignored. Everything needed to rebuild them is here; nothing else is missing.

| Path | Rebuild with | Notes |
|---|---|---|
| `data/finance/` | `uv run pg gen-data finance --seed 0` | Fully deterministic from the seed. |
| `data/hotpotqa/` | `uv run pg gen-data hotpotqa --seed 0 --n 300` | Seeded sample of the `hotpotqa/hotpot_qa` distractor validation split on Hugging Face (CC BY-SA 4.0). |
| `third_party/cfo-env/` and `data/enterprisearena/` | `uv sync --extra enterprisearena && uv run pg gen-data enterprisearena` | Downloads the simulator, verifies it, then writes seeded splits (episodes are just simulator seeds). |
| `runs/`, `graphs/evolved/` | `uv run pg eval ...` / `uv run pg evolve ...` | Outputs. LLM calls are not deterministic (temperature is not sent by default), so expect the numbers in the results sections to reproduce in distribution, not exactly. |

**If the CFO-Env mirror disappears.** The simulator code is published only as an anonymous review snapshot
(`anonymous.4open.science/r/CFO-Env-F1B9`), which may be taken down; the Hugging Face dataset `TheFinAI/CFO-Env`
has the data and documents but not the code. It has no license, so it cannot be redistributed here. The adapter
does not care where the copy comes from: put the snapshot's files in `third_party/cfo-env/` (from the authors, or
a later official release of EnterpriseArena, arXiv 2603.23638) and run `pg gen-data enterprisearena` again. It
skips the download when every file matches
[`cfo_env_sha256.json`](src/pg/envs/enterprisearena/cfo_env_sha256.json), and otherwise names the files that
differ. A newer release will not match the 2026-05-07 hashes; the results here were produced with that snapshot,
so treat a different version as a different benchmark (and regenerate the manifest deliberately if you adopt it).
The other two scenarios (finance, HotpotQA) do not depend on it.

*Maintainer's setup.* A private companion repository holds a verified copy of these four paths (the unlicensed
simulator cannot be shared). It is cloned into `private/` and symlinked into place with `sh private/link.sh`;
new runs and evolved graphs are committed there, never here.

### W&B metrics (optional)

```bash
uv sync --extra wandb        # then set PG_WANDB_PROJECT in .env (and `wandb login` or WANDB_API_KEY)
```

When `PG_WANDB_PROJECT` is set, every `eval` / `evolve` command creates one W&B run (metrics only, no LLM
tracing), with the CLI args and `Config` as the run config:
- `evolve` logs one step per round:
  - `train/score`, `train/success_rate`
  - `val/score` (the graph kept after the round) and `val/candidate_score`
  - `accepted` and `rejection_memory`
  - `graph/nodes` and `graph/edges`
  - solver, guidance and refiner tokens
  - `cost/round` and `cost/total`

  At the end it adds a `rounds` table (edits and rationale) and a `graph` artifact with `best.json` and the
  evolution log.
- `eval` logs a per-graph summary (`<graph>/mean_score`, `success_rate`, `guidance_overhead`, `cost`, and
  so on) and an `eval` comparison table.

Without the variable, nothing is imported or sent. Local files stay the source of truth either way.

## Layout

```
src/pg/
  config.py      Config: model ids, h=2, w=3, concurrency, truncation limits; loads .env
  llm.py         make_llm() via OpenRouter; token + cost accounting
  graph.py       Node / Edge / Relation / EditSet, neighborhood(), serialize(), apply()
  trajectory.py  Step, Trajectory, JSONL io, summarize()
  guidance.py    match(), build_context(), generate_guidance()
  agent.py       LangGraph StateGraph: guide -> solver -> tools | text -> guide ...; run_episode(), run_batch()
  refiner.py     refiner prompt (best/worst trajectories + rejection memory) and propose_edits()
  evolve.py      rollout / refine / validate / accept loop
  evaluate.py    compare graphs on a split
  cli.py         `pg` entry point
  envs/base.py   Environment (shared, read-only) and Episode (per-rollout state) interfaces
  envs/finance/  deterministic startup finance simulator
  envs/hotpotqa/ HotpotQA distractor setting with search / lookup / finish
  envs/enterprisearena/  adapter over the EnterpriseArena (CFO-Env) simulator in third_party/ (not redistributed)
graphs/          hand-written expert graphs; evolved/ output
tests/           no-network unit tests
```

The core (`graph`, `guidance`, `agent`, `refiner`, `evolve`, `evaluate`) is task-agnostic. Everything
scenario-specific lives under `envs/<name>/`.

## Scenarios

**Finance simulator.** The agent is CFO for 24 months and must keep cash at or above zero. Each task hides three things:
- a financing lag of 2 to 4 months
- 2 or 3 scheduled crises: a revenue shock, a cost spike, or churn
- the company's starting numbers (it starts with 5 to 8 months of runway)

Only one financing request can be pending at a time. Investors decline requests while runway is above 9 months
and cap a round at 6 months of net burn. The forecast tool ignores crises. Score is the fraction of months
survived, and success means surviving all 24.

The scenario is calibrated with scripted policies over 200 seeds, no LLM:

| scripted policy | survival |
|---|---|
| never raise (just advance) | 0% |
| raise when runway < 3 months | ~30% |
| raise when runway < 5 months | ~60% |
| raise when runway < 7 months | ~85% |
| raise as soon as investors allow (< 9), cut burn on alerts | ~95% |

So the procedural knowledge that matters is to raise early, never stack requests, and react to alerts.
`tests/test_finance_policies.py` pins the two ends of this range.

**HotpotQA.** The distractor setting, using each question's own 10 paragraphs. `search` returns the top-3
paragraphs by token overlap, `lookup` returns a paragraph by title, and `finish` submits the answer. There are 8 steps.
Score is SQuAD-style F1, and success means F1 >= 0.5. A text-only reply is accepted as the final answer.

**EnterpriseArena (CFO-Env).** This is the finance benchmark the paper actually uses (Han et al. 2026,
arXiv 2603.23638). Our synthetic simulator above turned out to be too easy: a no-graph agent survived it by
raising money at month 0.

*Not redistributed.* `pg gen-data enterprisearena` downloads the benchmark authors' anonymous review snapshot
(2026-05-07) into `third_party/cfo-env/` (gitignored) and checks every file against a pinned sha256 manifest.
The code has no license file and the data is CC BY-NC-ND 4.0, so use it for non-commercial research only.
If the mirror is gone, see [What is not in this repository](#what-is-not-in-this-repository).

*How it runs.* The agent is CFO of a consumer-lending fintech for 132 months of real 2015–2025 macro data:
- Each month it may make up to 20 information-tool calls plus free notes, then exactly one action (raise equity or debt, close the books, or pass) closes the month.
- As in the benchmark's own agent, the solver's conversation restarts every month, so only notes carry over.
- The solver is never told how many months the episode lasts; the benchmark's agent isn't either. Note that the
  refiner's environment description does mention the horizon.
- Fundraising approval is probabilistic: it depends on market conditions, drops after each equity round and weakens with leverage. Results arrive 1–6 months later at 70–100% of the amount requested.
- Hidden user-growth surges drain cash through new loans.

Episodes differ only by simulator seed (20/20/20), plus a 5-seed, 36-month `probe` split for cost checks.
`train` and `val` run to 66 months (the first two growth surges) so evolution stays affordable; `test` keeps
the benchmark's full 132-month horizon. A guided full-horizon episode is 500-800 steps, i.e. 1000-1600 model
calls and 1-3 hours, so `Config.episode_timeout_s` defaults to 4 hours and only catches genuine hangs.
Score is the fraction of months survived, and success means cash never went negative.

Scripted policies over 50 seeds, full horizon, no LLM:

| scripted policy | survival | mean months |
|---|---|---|
| never raise | 0% | 33 |
| one $20M equity raise at month 0 | 0% | 57 |
| greedy $20M debt | 0% | 79 |
| designers' equity-debt-equity schedule | 14% | 76 |
| greedy $5M equity | 56% | 105 |
| $20M equity whenever cash < $15M | 74% | 111 |
| greedy $20M equity (whenever nothing is pending) | 100% | 131 |

`graphs/enterprisearena_expert.json` encodes the last row, so it is an expert graph derived from calibration
rather than domain knowledge.

## Results

To be filled in after running with an API key.

| env | graph | split | n | mean score | success | guidance overhead | cost |
|---|---|---|---|---|---|---|---|
| finance | none | test | 30 | | | – | |
| finance | finance_expert | test | 30 | | | | |
| finance | evolved (scratch, 5 rounds) | test | 30 | | | | |
| hotpotqa | none | test | 150 | | | – | |
| hotpotqa | hotpotqa_expert | test | 150 | | | | |
| hotpotqa | evolved (scratch) | test | 150 | | | | |

## Deviations from the paper's text

- `Environment` is split into a shared `Environment` and a per-rollout `Episode` whose tools close over its
  state, so thread-pooled rollouts never share state.
- When the solver replies with text and no tool call, the episode's `on_text` decides whether that text is the
  final answer (HotpotQA) or whether to nudge the solver to keep calling tools (finance, at most
  `Config.max_nudges` times).
- Episodes can report actions they took themselves (`Episode.pop_extra_steps()`), so a month the environment
  closes on its own is still recorded as a step instead of vanishing inside another tool's observation.
- Scores average only over episodes that ran to completion; crashed or timed-out episodes are reported as
  `errors` / `n_scored` rather than counted as task failures.
- `Environment.compact_trajectory()` lets an environment summarize its own trajectories for the refiner prompt
  (EnterpriseArena renders one line per simulated month), so long episodes are not tail-truncated away. The
  default is the raw step list, as in the paper.
- `Episode.pop_context_reset()` lets an environment restart the solver's conversation from a fresh opening
  message. EnterpriseArena uses it for the benchmark's monthly reset. Step records and guidance carry across
  restarts.
- Temperature is not sent by default, because the default OpenRouter solver model does not accept it. Set
  `Config.temperature` for models that do.
- The paper does not publish its refiner prompt. Ours tells the refiner to write rules that transfer to any task
  in the environment and never to copy task-specific entities, queries or answers into the graph. In early smoke
  runs it did copy them. The validation gate is the paper's defence against this, but our validation sets are
  far smaller than the paper's HotpotQA set (50 vs 1,000), so `evolve` validates on the whole val split by default.

## License

MIT (see [LICENSE](LICENSE)) for the code in this repository. It does not cover the EnterpriseArena / CFO-Env
simulator or the HotpotQA data, which are not part of this repository and keep their own terms.
