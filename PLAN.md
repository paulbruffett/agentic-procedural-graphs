# Plan: Lightweight reference implementation of "Procedural Graphs" (arXiv 2609.09153)

## Context

The paper *Procedural Graphs: Self-Evolving Execution Structures for LLM Agents* (Lu, Chen, Wu, Arık, Sep 2026) proposes storing an agent's procedural knowledge as an explicit, editable graph of `(procedure, relation, procedure)` triplets. At inference time the agent is localized in the graph by its last tool call, a 2-hop neighborhood is extracted, and an LLM turns that neighborhood into step-level guidance injected into the solver's prompt. Offline, a refiner LLM compares successful and failed trajectories, proposes graph edits, and a validation gate accepts or rejects them; rejected edits are remembered so they are not re-proposed.

Goal: a small, readable Python/LangGraph implementation of both halves (online guidance + offline self-evolution) plus two scenarios to exercise it: a synthetic **mini enterprise finance simulation** (the paper's clearest long-horizon self-evolution result) and **HotpotQA** (real data, short episodes, cheap to iterate on). Everything scenario-specific sits behind one small `Environment` interface so the core is provably task-agnostic.

Decisions already made with the user:
- All LLM calls via **OpenRouter** through `langchain-openai` (`ChatOpenAI(base_url="https://openrouter.ai/api/v1")`), one `OPENROUTER_API_KEY` in `.env`.
- Solver + guidance default `openai/gpt-5.6-luna` ($0.20/$1.20 per MTok). Refiner defaults to a stronger OpenRouter model (`anthropic/claude-opus-5`, fall back to `openai/gpt-5.6-luna-pro` if that id is not served). All three are config strings.
- Tooling: `uv` + `pyproject.toml`. LangGraph 1.2.x supports Python ≤3.13, the machine has 3.14, so pin `requires-python = ">=3.12,<3.14"` and let uv fetch 3.13.
- Greedy decoding (temperature 0) was the intent, matching the paper. As built, `Config.temperature` defaults to `None` (not sent) because the default solver model rejects the parameter; set it for models that accept it.

## Paper facts the implementation mirrors

| Paper element | Implementation |
|---|---|
| G = (V, R, E, Φ); nodes are tool functions / reasoning steps / task states | `Node(id, type ∈ {ACTION, REASONING, STATE}, description)` |
| Relations: LEADS_TO, TRIGGERS, PROVIDES_INPUT_FOR, CONVERGES_TO | `Relation` enum |
| Edge attributes: condition, guidance, pitfalls | `Edge(src, rel, dst, condition, guidance, pitfalls)` |
| Match: exact match of last tool call to a node, else full graph | `match(last_tool_name)`; `None` → whole graph |
| h = 2 hop neighborhood, w = 3 trajectory window | `Config.h`, `Config.w` |
| Guidance model Ψ is the same LLM as the solver | `guidance_model` config, defaults to solver model |
| Solver sees `Procedural Graph Guidance: ...` in its system prompt | appended to env system prompt each step, not persisted in message history |
| Evolution: rollout → mutate (add/delete; revise = delete+add) → validate on held-out (accept if ≥) → rejection memory | `evolve.py` + `refiner.py` |
| Scratch init = minimal skeleton | Start/End STATE nodes + one ACTION node per tool, zero edges; round 1 refiner discovers the backbone |

## Layout

```
procedural-graphs/
  pyproject.toml  .env.example  README.md
  src/pg/
    config.py        Config dataclass: model ids, h, w, concurrency, paths; loads .env
    llm.py           make_llm(model_id, temperature) -> ChatOpenAI via OpenRouter; temperature only sent when set
    graph.py         ProceduralGraph: Node/Edge/Relation, JSON load/save, neighborhood(node, h),
                     apply(EditSet) -> new graph, serialize(subgraph, active_node) -> text
    trajectory.py    Step(tool, args, observation), Trajectory(task_id, steps, score, usage), JSONL io
    guidance.py      match(), build_context(), generate_guidance(llm, graph, task_text, steps[-w:])
    agent.py         LangGraph StateGraph; run_episode(env, task, graph | None, cfg) -> Trajectory
    refiner.py       EditSet pydantic schema + refiner prompt + propose_edits(llm, graph, traces, rejected)
    evolve.py        evolve(env, init_graph, cfg): rounds of rollout/refine/validate/accept, persists per-round graphs + log
    evaluate.py      run a split with/without graph(s); mean score, success rate, token overhead
    cli.py           argparse entry point `pg`: gen-data | evolve | eval | analyze
    analyze.py       paired statistics over eval run directories (added later; see Build status)
    tracking.py      optional W&B metrics logging (added later; see Build status)
    envs/
      base.py        Environment protocol (below)
      finance/  sim.py  tools.py  tasks.py  env.py
      hotpotqa/ data.py tools.py  scoring.py env.py
      enterprisearena/  data.py tools.py env.py   (added later; see Build status)
  graphs/            finance_expert.json, hotpotqa_expert.json (hand-written); evolved/<env>/round_k.json
  data/              generated task files (gitignored except a tiny sample)
  runs/              trajectories + metrics per run (gitignored)
  tests/             no-LLM unit tests
```

## Core design

### `envs/base.py` — the only scenario-facing surface
```python
class Environment(Protocol):
    name: str
    max_steps: int                       # tool calls per episode
    success_threshold: float             # score >= threshold counts as success
    def tasks(self, split: str) -> list[Task]           # Task(id, prompt, meta)
    def reset(self, task: Task) -> list[BaseTool]       # fresh per-episode state; tools close over it
    def system_prompt(self) -> str
    def is_done(self) -> bool
    def score(self, task: Task) -> dict                 # {"score": float, ...extra metrics}
    def tool_descriptions(self) -> list[tuple[str, str]]  # for skeleton graph + refiner context
```

### `agent.py` — LangGraph loop
State: `messages` (add_messages), `steps: list[Step]`, `guidance: str`, `n_steps: int`.
Nodes and edges:
- `guide`: if no graph → `guidance=""`. Else `match(last step's tool)` → `neighborhood` (or full graph) → `generate_guidance(...)` with last `w` steps.
- `solver`: `SystemMessage(env.system_prompt() + guidance block)` + `messages` → `llm.bind_tools(tools).invoke(...)`.
- `tools`: own small node (not `ToolNode`) so each call is recorded as a `Step` (name, args, truncated observation) in the same pass; appends `ToolMessage`s.
- Routing: `solver` → `tools` if tool_calls else END; `tools` → END if `env.is_done()` or `n_steps ≥ max_steps` else `guide`. `recursion_limit = 3*max_steps + 5`.
- Token usage summed from `AIMessage.usage_metadata`, split solver vs guidance, so the paper's overhead table can be reproduced.

### `graph.py`
- `neighborhood(node_id, h)`: BFS over both edge directions, h hops; returns induced subgraph.
- `serialize()`: the paper's text form, e.g. `Active Node: [search] (ACTION) ...` then each outgoing/incoming edge with condition/guidance/pitfalls.
- `apply(EditSet)`: copy; add nodes, add edges, delete edges, delete nodes (dropping dangling edges). Cycles are allowed (retry loops are legitimate). Returns new graph; original untouched.

### `refiner.py`
```python
class EditSet(BaseModel):
    rationale: str
    add_nodes: list[Node]; add_edges: list[Edge]
    delete_nodes: list[str]; delete_edges: list[tuple[str, str, str]]   # (src, rel, dst)
```
Obtained with `llm.with_structured_output(EditSet)` (OpenRouter reports `structured_outputs` for gpt-5.6-luna; fall back to JSON mode + pydantic parse if a model lacks it). Prompt content: env description + tool list, current graph JSON, K best and K worst trajectories (compact: step list + score), and rejection memory entries `{round, edits summary, val_before, val_after}`. Trajectory + memory text is tail-truncated to `Config.refiner_max_chars` (paper keeps the final L_max tokens).

### `evolve.py`
Per round k: rollout train batch (size B, rotating window through the train split) with G_{k-1} → sort by score, take top/bottom K → refiner → `G_cand` → `S_val(G_cand)` on the fixed val set → accept if `≥ S_val(G_{k-1})` (cached from the round it was accepted) else push to rejection memory. Persist `graphs/evolved/<env>/round_k.json`, `best.json`, and `evolution_log.jsonl` (per round: batch score, val score, accepted?, edit summary). Rollouts run in a `ThreadPoolExecutor(cfg.concurrency)`; each episode gets its own env reset so threads never share state.

## Scenarios

### Finance sim (`envs/finance/`)
Deterministic, seeded, no LLM in the environment.
- **State**: month, cash, monthly revenue, monthly burn, growth rate, pending financing request (at most one), notes list, capital_raised, event log.
- **Hidden per-task parameters** (in `Task.meta`, never shown to the agent): financing lag L ∈ {2,3,4} months, 2–3 scheduled crises (revenue shock −40% for 3 months, one-time cost spike, churn step-down).
- **Tools**: `audit_cash()`, `forecast_runway(months)` (naive trend extrapolation, does not know crises), `submit_financing_request(amount)` (rejected if one is pending; arrives after L months; capped), `adjust_burn(pct)`, `write_note(text)` / `read_notes()`, `advance_month()` (applies dynamics + due events, returns month summary). Episode ends at horizon H (default 24) or when cash < 0.
- **Score**: `survived` (primary, binary), `months_survived / H`, `capital_raised`, `final_cash`. `success_threshold = 1.0` on survived.
- **Task generation** (`tasks.py`): seed → sampled initial cash/revenue/burn/growth/lag/crisis schedule; writes `data/finance/{train,val,test}.jsonl` (defaults 30/20/30).
- **Calibration check without an LLM** (`tests/test_finance_policies.py`): a scripted *naive* policy (just advances months) must survive < 30% of 50 seeds; a scripted *expert* policy (raises when runway < lag + buffer, cuts burn on shock) must survive > 90%. This guarantees the scenario actually rewards the procedural knowledge the graph is supposed to capture.
- **Expert graph** (`graphs/finance_expert.json`): `Start → audit_cash → forecast_runway → submit_financing_request` (condition: runway < lag + buffer; pitfall: don't stack requests) `→ advance_month`, `advance_month → read_notes` (TRIGGERS on crisis event), `adjust_burn` CONVERGES_TO `advance_month`, etc. About 8 nodes, ~12 edges.

### HotpotQA (`envs/hotpotqa/`)
- `data.py`: `load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")`, sample N (default 300) with a seed, split 100/50/150, cache to `data/hotpotqa/*.jsonl` so the `datasets` dependency is only hit once.
- **Tools** over the example's own 10 context paragraphs: `search(query)` → top-3 paragraphs by normalized token overlap (no extra dependency), `lookup(title)` → full paragraph, `finish(answer)` → ends episode. `max_steps = 8`.
- **Score**: SQuAD-style normalized exact match and F1 (`scoring.py`). `score = F1`, `success_threshold = 0.5`.
- **Expert graph** (`graphs/hotpotqa_expert.json`): the paper's `Start → search(first hop) → analyze (REASONING) → search(second hop) → verify (REASONING) → finish` with attributes.

## CLI (`uv run pg ...`)
```
pg gen-data finance  --seed 0 --n-train 30 --n-val 20 --n-test 30
pg gen-data hotpotqa --seed 0 --n 300
pg evolve <env> --init scratch|<path> --rounds 5 --batch 10 --val-n 20
pg eval   <env> --split test [--n 10] --graphs none,graphs/x.json,graphs/evolved/<env>/best.json [--concurrency 4]
```
`eval` prints mean score, success rate, mean steps, and solver vs guidance token totals; writes trajectories to `runs/<timestamp>-eval-<env>/`. (`pg run` was dropped: it was `pg eval` with one graph.)

## Dependencies
`langgraph>=1.2,<2`, `langchain-openai>=1.6,<2`, `langchain-core`, `pydantic>=2`, `python-dotenv`, `datasets` (HotpotQA only); dev: `pytest`. No retrieval libraries, no LLM judge.

## Implementation order
1. Scaffold (`pyproject`, `.env.example`, `config.py`, `llm.py`), `graph.py`, `trajectory.py`, unit tests for neighborhood/serialize/apply.
2. `envs/base.py`, HotpotQA env + scorer + tests, `agent.py` without guidance, `pg run hotpotqa --n 3` smoke test.
3. `guidance.py`, expert HotpotQA graph, `pg run hotpotqa --graph ...`.
4. `refiner.py`, `evolve.py`, `pg evolve hotpotqa --init scratch --rounds 2` smoke test.
5. Finance sim, tools, task generator, scripted-policy calibration tests, expert graph.
6. `evaluate.py`, `pg eval` for both envs, README with results table placeholders.

## Verification
- `uv run pytest`: graph ops, edit application, EM/F1, finance sim dynamics, naive-vs-expert policy survival gap. No network.
- HotpotQA smoke: `pg eval hotpotqa --split val --n 5 --graphs none,graphs/hotpotqa_expert.json`; trajectories show guidance text in the solver system prompt and steps recorded correctly.
- Evolution smoke: `pg evolve hotpotqa --init scratch --rounds 2 --batch 5 --val-n 10` produces round graphs, an accepted or rejected decision per round, and a rejection memory entry appearing in the next round's refiner prompt.
- Finance headline: `pg eval finance --graphs none,graphs/finance_expert.json` on the test split shows the expert graph beating no-graph on survival; then `pg evolve finance --init scratch --rounds 5 --batch 10 --val-n 20` and confirm validation survival climbs across rounds, mirroring the paper's Figure 4 shape.

## Build status (2026-09-13)

All modules in the layout are written, plus `README.md`, both expert graphs, and the tests. `uv sync` and `uv run pytest` pass (no network). `data/finance/` is generated (seed 0, 30/20/30).

Offline checks done: agent loop with scripted fake chat models (`tests/test_agent.py`); evolution accept/reject gate and rejection memory reaching the next refiner prompt with rollouts and refiner stubbed (`tests/test_evolve.py`); expert graph ACTION ids equal tool names (`tests/test_expert_graphs.py`); refiner structured-output request and OpenRouter `usage.cost` parsing against a mocked HTTP transport (one-off script, not a test).

NOT yet done (needs `OPENROUTER_API_KEY` in `.env`): `pg gen-data hotpotqa` (HF download), every LLM smoke test in the Verification section, the finance headline and evolution runs, and the README results table.

Choices made during the build, beyond the design text:
- Stop on fatal API errors (2026-09-19): evolve run ea-run3 lost rounds 4-5 to OpenRouter HTTP 402 (out of credits): the hardening kept the accepted graph, but the run carried on into a round that could not succeed. `llm.is_fatal()` (HTTP 401/402) now marks a `Trajectory` as `fatal`; after a batch is written to disk, `trajectory.raise_if_fatal()` stops `evolve` / `eval` (finished episodes are kept, and `eval` writes `metrics.json` after every graph so a stopped eval is still analyzable); a fatal refiner error propagates instead of being logged as a failed round. Not done: capping `max_tokens` (OpenRouter reserves credit for the model's 65,536-token default, so runs die earlier than the balance suggests). `git_dirty` now ignores untracked files. ea-run3 result: 3 effective rounds, val survival 5/20 -> 17/20 (0.611 -> 0.946), best graph 13 nodes / 13 edges, $43.56.
- Measurement tooling (2026-09-18, on request; primary metric = `success`, i.e. survival): `trajectory.episode_rows()` flattens each episode to scalars (no steps, no guidance text); `eval` and `evolve` log them as a W&B `episodes` table (evolve: cumulative, every round, with `round`/`phase`); `--experiment` on `evolve`/`eval`/`analyze` sets the W&B group; the run config records `git_sha`/`git_dirty`, and eval rows record each graph file's `graph_sha256`. New `pg analyze <eval run dirs...>` (`src/pg/analyze.py`, stdlib only): per graph vs `--baseline`, paired by task, errored episodes dropped, mean difference + 95% task-resampling bootstrap CI, exact McNemar on `success` when there is one run per graph; several run dirs are repeats (averaged per task, McNemar omitted). It reads the local run directories (the source of truth) and logs its `paired` table to W&B only when `--experiment` is given. Decided: guidance text and trajectories are never uploaded to W&B. Checked against the 2026-09-14 hidden-horizon eval: reproduces McNemar p=0.001 (+11/-0) and +53.5 months. The protocol (test split only, skeleton `round_0.json` as control, repeats, limits) is in the README section "Measuring whether a graph helps".
- Repo split (2026-09-18, on request): code, tests, docs and expert graphs are public at github.com/paulbruffett/agentic-procedural-graphs; `third_party/`, `data/`, `runs/` and `graphs/evolved/` stay gitignored here and live in the private companion repo `agentic-procedural-graphs-private` (cloned to `private/`, symlinked by `sh private/link.sh`), so results stay reproducible if the CFO-Env review mirror or the HF dataset vanish. The README section "What is not in this repository" tells everyone else how to rebuild each path, including what to do if the mirror is gone.
- Review fixes (2026-09-18): removed dead code from the EnterpriseArena env (a duplicate `_usd`, an uncalled `_event` that referenced an unimported `re` and an undefined `_millions`, unused `ACTION_TOOLS`); `evolve` ranks only non-errored train trajectories for the refiner (same rule as `summarize()`), so harness crashes are not presented as procedural failures; a refiner failure (exception, unparseable reply, or no scored train episodes) no longer kills the run: the round is logged with rationale `refiner failed: ...`, the graph is kept, nothing is validated or added to rejection memory. Same day, on request: the val gate needs `Config.val_min_scored` (0.8) of the val episodes to have run without error, for the round-0 baseline (else abort) and for each candidate (else the round is no data: graph kept, no rejection-memory entry), so a graph can no longer be accepted on a handful of survivors; and `evolve` refuses an output directory that already holds `evolution_log.jsonl` unless `pg evolve --overwrite` (or a different `--out`) is given.
- `envs/base.py` gained `write_tasks` / `read_tasks` JSONL helpers shared by both scenarios.
- Finance `score = months_survived / H` (equals 1.0 exactly when survived), so the val gate and refiner ranking get a denser signal than binary survival; `success = survived`.
- Finance investor rules added to make timing matter: requests are declined while runway > 9 months, and capped at 6 months of net burn. Calibration over 200 seeds: never raise 0%, raise at runway < 3 / 5 / 7 months ≈ 30% / 60% / 85%, raise as soon as allowed (< 9) + cut burn on alerts ≈ 95%. The scripted expert therefore uses "runway < 9" rather than "lag + buffer".
- `evolve` skips validation for an empty edit set and does not add it to rejection memory.
- Added on request (2026-09-14): refiner prompt rule that graph text must transfer to any task in the environment (no task-specific entities/queries/answers; environment-level thresholds and tool names are fine). The paper's refiner prompt is unpublished (Appendix B.5 describes only its structure) and it does not address this; the smoke evolve run copied HotpotQA entity names into edges. Also `pg evolve --val-n` now defaults to the whole val split (hotpotqa 50, finance 20); the paper validates HotpotQA on 1,000.
- Evolution run 2 (2026-09-16/17) was stopped mid-round-4 after 32h and ~$27.50: the 1h per-episode deadline killed healthy long episodes (one at month 113/132, another at 129), so round 1's val read 0.000 with 10/10 timeouts (wrongly rejected) and round 2 was accepted on only 3 scored episodes. Round 0 val 0.436; the accepted round-2 graph (12 nodes, 7 edges) scored 1.000 on the round-3 train batch with 7/10 surviving, so the method was working. Fixes: `Config.episode_timeout_s` now 4h (0 disables); a validation pass where nothing scored counts as no data (round skipped, graph kept, nothing added to rejection memory) instead of scoring 0.0; and EnterpriseArena train/val now run to `EVOLVE_MONTHS = 66` while `test` keeps the full horizon, roughly halving the cost per evolution episode.
- Second pass of review/simplification fixes (2026-09-16): EnterpriseArena reports an `outcome` of survived/bankrupt/stopped and `summarize()` counts `stopped_early`, so harness stops are visible in the results table; `pg gen-data hotpotqa` rejects n < 6 (empty val split) and `evolve` refuses empty train/val splits; `log_table` ignores empty row lists; the unused `success_threshold` is gone from the finance env; `pg run` was dropped (it was `pg eval` with one graph); `evolve`'s W&B block and `cli.main` were split into helpers. Still open by choice: whether to keep the synthetic finance scenario now that EnterpriseArena supersedes it.
- Code review fixes (2026-09-16), after a 10h hang killed an evolve run: (1) `Episode.pop_extra_steps()` so months the episode closes itself (tool-budget exhaustion, text-only reply) are recorded as steps; (2) EnterpriseArena keeps a structured `month_log` in its metrics and `compact_trajectory` renders from it, deleting the regex parsing of observation text; (3) `summarize()` averages only over non-errored episodes and reports `n_scored`/`errors`, so crashes are no longer counted as task failures (`stopped_early` metric added per trajectory); (4) notes are capped at 20/month and hitting the cap closes the month, so a note loop cannot stall an episode; (5) `evolve` skips the validation pass when an edit set applied to nothing and marks the round `(no-op: nothing applied)`; (6) the refiner falls back to plain JSON only for ValueError/ValidationError/BadRequestError/NotFoundError, so rate-limit and auth errors propagate; (7) explicit httpx connect/read/write/pool timeouts plus a per-episode `Config.episode_timeout_s` (default 3600s) deadline, the cause of the hang (the old `timeout=180` never fired on a silently dropped connection).
- Added on request (2026-09-15): `Environment.compact_trajectory()` (default: `Trajectory.compact()`), passed to `build_prompt`, so an environment can summarize long episodes for the refiner. EnterpriseArena renders one line per month (collapsing runs of uneventful `pass` months, compressing fundraising events): a 132-month episode goes from up to 182,766 chars to 245-7,095, so the K best + K worst trajectories now fit the 60,000-char refiner budget (worst case 42,570) instead of being tail-truncated.
- Added on request (2026-09-14): EnterpriseArena adapter (`src/pg/envs/enterprisearena/`), because the synthetic finance sim was too easy (no-graph luna survived 5/5 by raising at month 0). The simulator is the authors' CFO-Env review snapshot (anonymous.4open.science/r/CFO-Env-F1B9, 2026-05-07; data identical to HF TheFinAI/CFO-Env @51d901a2), fetched by `pg gen-data enterprisearena` into gitignored `third_party/cfo-env/` and verified against `cfo_env_sha256.json`; no LICENSE (non-commercial research only, not redistributed). Core gained `Episode.pop_context_reset()` to mirror the benchmark agent's monthly conversation reset. Episodes = simulator seeds (20/20/20 + 5-seed 36-month `probe`). Scripted calibration over 50 seeds: never raise 0%, one-shot equity 0%, greedy debt 0%, designers' E-D-E schedule 14%, greedy $5M equity 56%, equity when cash < $15M 74%, greedy $20M equity 100%; `graphs/enterprisearena_expert.json` encodes the last. Probe (5 seeds x 37 months, luna, 2026-09-14): no graph 3/5 survived (score 0.951), ~5.8 steps/month, first raise only at months 26-30 as the first surge hit, $0.039/episode; expert graph 5/5 (1.000), ~1.1 steps/month, equity from month 0 and whenever free, $0.034/episode (guidance overhead 111% of solver tokens but far fewer calls). Expert still wasted ~3-5 monthly actions resubmitting while a request was pending. Full-horizon test eval (20 seeds x 132 months, horizon still disclosed, 2026-09-14): both no graph and expert survived 20/20; expert used 141 vs 760 steps/episode and cost $0.119 vs $0.161/episode ($5.60 total), but its valuation score was worse (median $156M vs $236M; paired mean -$112M, 95% CI [-163, -65], lower on 18/20 seeds): it raised $20M rounds (8.3 rounds, $139M) and wasted 12.2 actions/episode resubmitting while pending, whereas no-graph luna requested ~$131M per attempt (3.4 rounds, $257M).
- Added on request (2026-09-14): the EnterpriseArena solver no longer sees the horizon (task prompt and monthly status), matching the benchmark's own agent. The first full test eval (no graph 20/20 survived) ran with the horizon disclosed; a model check (google/gemini-3.5-flash, one of the paper's models, vs gpt-5.6-luna, both no graph, horizon hidden, test seeds 2000-2004) is meant to separate model from harness effects. Result for gpt-5.6-luna with the horizon hidden: 0/5 survived (bankrupt at months 32-60) vs 5/5 with the horizon shown on the same seeds; hidden runs raised only after cash had peaked, in small or debt requests ($10-50M), or never, i.e. the EnterpriseArena paper's failure modes. Note the two prompts also differ in wording ("through month N" vs "every month"), so horizon number and long-term framing are confounded. Gemini 3.5 Flash (same seeds, horizon hidden, $8.64): 1/5 survived the full horizon (seed 2001), 3/5 bankrupt at months 32-33, and seed 2004 is inconclusive: it crashed at month 33 while solvent ($20.1M cash, $25M equity just requested) on a `usage_of` bug when OpenRouter returned `token_usage: null`, and the harness scored the aborted episode as a failure. Bug fixed in `pg.llm.usage_of` with a regression test. Conclusion so far: without the horizon both models fail at the first growth surge in the way the paper describes, so the harness's horizon disclosure, not the model, explained most of our earlier 20/20 survival. Hidden-horizon test comparison (gpt-5.6-luna, 20 seeds x 132 months, runs/20260914-194654-eval-enterprisearena, $3.72): no graph 9/20 survived (10 bankrupt at months 32-33, one at 59; 5/20 never requested funding; first request median month 29), expert graph 20/20 (paired exact McNemar p=0.001; +53.5 months, 95% CI [+33.5, +74.7]); expert median valuation $156M vs $0 (no-graph survivors avg ~$78M). The expert graph still requested only $20M per round and wasted 14.3 actions/episode resubmitting while pending. Run-to-run variance is real: seeds 2002 and 2004 went bankrupt in the model check but survived here.
- Added on request (2026-09-14): opt-in W&B metrics logging (no tracing) in `src/pg/tracking.py`, enabled by `PG_WANDB_PROJECT`, `wandb` as the optional extra `uv sync --extra wandb`. `evolve` logs per-round metrics plus a rounds table and best-graph artifact; `run`/`eval` log per-graph summaries and a comparison table.

Two deliberate deviations from the design text above (already reflected in the written code):
1. **Environment is split into `Environment` (shared, read-only: `tasks`, `start`, `system_prompt`, `tool_descriptions`) and `Episode` (per-rollout state: `tools`, `is_done`, `score`, `on_text`).** `env.start(task)` returns a fresh `Episode`; tools close over it. This is what makes thread-pooled rollouts safe.
2. **Text-only solver turns.** When the solver replies without a tool call and the episode is not done, `agent.py` should call `episode.on_text(text)`; if it returns False, append a `HumanMessage("The task is not complete. Continue by calling tools.")` and loop back to `guide`, at most `Config.max_nudges` times. HotpotQA's `on_text` accepts the text as the final answer; finance's returns False.

## Open risks (flagged, not blocking)
- `anthropic/claude-opus-5` and `openai/gpt-5.6-luna-pro` both have live OpenRouter model pages advertising tools + response_format; pricing for the Claude one was not visible, so the first evolve run should print per-round cost from OpenRouter's `usage.cost` field. Refiner default is `anthropic/claude-opus-5`, fallback `openai/gpt-5.6-luna-pro`.
- If the naive-vs-expert survival gap is too small, sim parameters (lag, crisis depth, initial runway) get retuned before any LLM runs. This is the single biggest determinant of whether the finance results are interesting.
- LangGraph 1.x API details (imports for `StateGraph`, `START`, `END`, `MessagesState`, `add_messages`) confirmed against current docs; `ToolNode` deliberately not used.
