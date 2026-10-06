# research

An autoresearch-style loop for finding a trading strategy with a real edge. One file is
edited, one command scores it in under a second on this Mac, results go in a TSV, and
improvements advance a git branch. This mode sits beside `python -m agent run`; it does not
replace it. The human edits this file; the agent edits `research/Candidate.py`.

The one thing that makes this different from autoresearch: the score is a backtest, and a
backtest on development data is easy to fit. The handoff showed the same funding idea go from a
near-pass to losing money on a rerun, so run-to-run implementation noise is large. The loop
therefore scores the **worst** of several views rather than the pooled number, and asks for
changes to the idea's structure rather than to its constants.

## Setup

Work with the user to:

1. **Agree on a run tag**, e.g. `oct6`. The branch `research/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b research/<tag>` from `main`.
3. **Read the in-scope files**:
   - `docs/runbook.md` "Strategy file format" and "Campaigns and the held-back data".
   - `agent/prompts.py` `system_prompt()`: the data columns, the file format, the Freqtrade
     parity rules and why always-long fails. Those rules apply here unchanged.
   - `research/score.py` — the scorer. Do not modify.
   - `research/Candidate.py` — the file you modify.
4. **Verify data exists**: `data/` must hold the three `*_4h.feather` files and the three
   `*_funding_1h.feather` files. If the funding files are missing, every funding line in the
   score says `funding assumed` and funding ideas cannot be tested; tell the human to copy them
   from tela (`/data/services/backtest_api/data/`).
5. **Initialize `research/results.tsv`** with just the header row (it is gitignored).
6. **Confirm and go.**

## What you can and cannot do

**You CAN:**
- Edit `research/Candidate.py`. Everything in it is fair game within the strategy file rules:
  the signal, the filters, the exit logic, which columns it reads, whether it shorts.
- Write throwaway measurement scripts under `research/scratch/` (gitignored) that read the
  `data/` feathers directly, e.g. "after a 3-day funding z-score above 2, what is the next 24h
  return on each coin, by year?" Log those as `measure` rows. Use them before writing a
  strategy for an idea, not after a strategy fails.

**You CANNOT:**
- Call tela, the backtest API, `scripts/submit_strategy.sh`, or anything in `agent/` that
  posts. Every Phase 1 POST is logged as a campaign attempt forever and a final test is spent
  forever. This loop is local only. If something you did would have needed the network, stop.
- Modify `research/score.py`, anything under `backtest_api/`, `config/holdout.json`, or the
  data files. The scorer is the ground truth; the holdout date is the campaign.
- Read any bar on or after the holdout date (`config/holdout.json`). The scorer already cuts
  them; your scratch scripts must cut them too (`backtest_api.periods.development_part`).
- Install packages or import anything outside the allowed list (pandas, numpy, pandas_ta,
  freqtrade, typing, math).
- Add per-coin special cases, or more than 6 distinct tunable numbers. The scorer rejects the
  file if it breaks the format rules.

## The score

`uv run python -m research.score > research/run.log 2>&1` prints a block like:

```
---
score:            -1.0234   (worst-view Sharpe; view = period 2)
phase1_gate:      FAIL cagr,max_drawdown,profit_factor,sharpe,floor
full:             sharpe -0.22  cagr -11.0%  dd -50.2%  pf 0.89  trades 131  beta 0.08
benchmark_sharpe: 0.51   (buy-and-hold; Phase 1 bar is max(0.8, this))
period_1:         sharpe -0.25  ret -10.8%  dd -33.8%  pf 0.87  trades 45  [2024-01-05..2024-10-05]  FAIL ...
period_2:         ...
period_3:         ...
pair_BTC:         sharpe 0.46  ret +22.9%  dd -32.3%  pf 1.33  trades 48  funding real
pair_ETH:         ...
pair_SOL:         ...
```

- **`score` is the headline: the lowest Sharpe across the three Phase 2 periods and the three
  coins. Higher is better.** An edge that is real shows up in every period and on every coin;
  a fit to one bull run or one coin does not move the worst view.
- `phase1_gate` is exactly what tela's Phase 1 would say today. `period_N` lines show the
  Phase 2 per-period gates (pf ≥ 1.2, dd ≥ -25%, ≥ 10 trades, positive return). When
  `phase1_gate` is PASS and every period says `ok`, the candidate is worth the human's
  attention; it still may not be worth a real attempt (see below).
- `beta` near 1 means the result is mostly market exposure. The edge is what buy-and-hold does
  not already give.
- `funding real` must appear on every pair line for a funding strategy to mean anything.

Pull the numbers with `grep "^score:\|^phase1_gate:" research/run.log`. If the grep is empty
the run crashed; `tail -n 30 research/run.log` has the trace. A `score: 0.0000` with an
`error:` line is a format or look-ahead rejection, not a market result.

## Logging results

Append a row to `research/results.tsv` (tab-separated) after every run:

```
commit	score	full_sharpe	phase1	status	description
a1b2c3d	-1.0234	-0.22	FAIL	keep	baseline: FundingFade example
b2c3d4e	-0.6100	0.10	FAIL	keep	exit on z crossing +-0.5 instead of 0
c3d4e5f	-1.4000	0.30	FAIL	discard	add 50-bar trend filter (helped pooled, hurt period 1)
d4e5f6g	0.0000	0.00	-	crash	funding sum looked ahead (prefix check)
e5f6g7h	-	-	-	measure	funding z>2 -> next-24h return: BTC -0.4% ETH -0.3% SOL +0.1% (SOL does not revert)
```

`status` is `keep`, `discard`, `crash` or `measure`. The description should say what changed
and, for discards, which view got worse — that is the information the human reads in the
morning.

## Recovery

Before starting the loop, verify HEAD is at the best known state:

1. `git tag -l research-best`. If it exists and `git diff research-best --stat` shows changes,
   `git reset --hard research-best`.
2. If no tag, take the best `keep` row in `results.tsv`, `git tag -f research-best <commit>`,
   reset to it.
3. If neither exists, this is a fresh run.

## The experiment loop

LOOP FOREVER:

1. Note the current branch and commit.
2. Change `research/Candidate.py` with one idea. Keep the module docstring honest about what
   edge it is after and why it should hold on all three coins.
3. `git commit -am "<short description>"`.
4. `uv run python -m research.score > research/run.log 2>&1` (always redirect).
5. `grep "^score:\|^phase1_gate:" research/run.log`; on empty output read the trace, fix a dumb
   bug and rerun, or log `crash` and move on if the idea itself is broken.
6. Log the row in `results.tsv` (never commit the TSV; it stays untracked).
7. **Keep** only if the score improved by at least **0.10** and the file has no more tunable
   numbers than before (or it improved by any amount with fewer). Then
   `git tag -f research-best HEAD` and `git tag research-keep-<score>-<hash> HEAD`.
8. Otherwise `git reset --hard research-best`.

**The first run is always the baseline**: score `Candidate.py` exactly as it is.

**What counts as an idea.** Change the logic or structure: the signal, a filter, the exit rule,
the holding logic, which coins' columns it reads. Changing a constant (18 → 24, 2.0 → 1.5) is
not an experiment here; the file would have fit the development data a little better and
learned nothing. Use round, conventional values and leave them alone. If you genuinely do not
know whether a signal exists, run a `measure` row first.

**Simplicity criterion.** Fewer knobs and fewer lines beat a slightly higher score. Deleting a
filter and getting an equal score is a win: keep it.

**Where the time is.** Scoring takes under a second, so the loop is bounded by how fast you can
think of a new, distinct idea. That is the point. Do not fill the TSV with constant sweeps.

**SOL.** It fails nearly every strategy. A strategy that cannot be flat on SOL when its edge is
absent there is not going to pass; it may not be special-cased either. Think about what SOL does
differently (higher beta, more violent liquidation spikes) and build that into the shared logic.

**Never stop** to ask whether to continue once the loop has begun. The human may be asleep. If
you run out of ideas: reread the seeds in `agent/prompts.py` `DEFAULT_SEEDS`, combine two
near-misses, flip the holding period, go measure something.

## When a candidate looks good

A candidate with `phase1_gate: PASS` and three `ok` periods is **not** submitted by this loop.
The human decides. Leave it tagged, write its row, and keep going on a fresh idea. In the
morning the human reads `results.tsv`, inspects the file, and if it is a strong, simple
candidate, copies it under its real name to `strategies/candidates/` and runs the real
pipeline. Only the first one or two final tests of a campaign are realistic; a candidate that
scraped past the gates is not worth one.
