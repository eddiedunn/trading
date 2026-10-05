-- Trading stack Postgres schema
-- Applied via docker-entrypoint-initdb.d on first container start

-- Paper trading metrics history (hourly snapshots from monitor.py)
CREATE TABLE IF NOT EXISTS paper_snapshots (
    id            SERIAL PRIMARY KEY,
    ts            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    strategy      TEXT NOT NULL,
    profit_pct    NUMERIC,
    trade_count   INTEGER,
    win_rate      NUMERIC,
    profit_factor NUMERIC,
    max_drawdown  NUMERIC
);

CREATE INDEX IF NOT EXISTS idx_paper_snapshots_strategy ON paper_snapshots (strategy);
CREATE INDEX IF NOT EXISTS idx_paper_snapshots_ts ON paper_snapshots (ts);

-- Strategy registry: full lifecycle from candidate to live
CREATE TABLE IF NOT EXISTS strategy_registry (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    parent_strategy TEXT,
    iteration       INTEGER,
    phase1_passed   BOOLEAN,
    phase1_stats    JSONB,
    phase2_passed   BOOLEAN,
    phase2_stats    JSONB,
    paper_passed    BOOLEAN,
    paper_queued_at   TIMESTAMPTZ,
    paper_started_at  TIMESTAMPTZ,
    paper_finished_at TIMESTAMPTZ,
    promoted_live   BOOLEAN DEFAULT FALSE,
    promoted_at     TIMESTAMPTZ,
    retired_at      TIMESTAMPTZ,
    notes           TEXT
);

CREATE INDEX IF NOT EXISTS idx_strategy_registry_name ON strategy_registry (name);

-- Paper arena queue columns (added 2026-10); no-ops on a fresh database.
ALTER TABLE strategy_registry ADD COLUMN IF NOT EXISTS paper_queued_at   TIMESTAMPTZ;
ALTER TABLE strategy_registry ADD COLUMN IF NOT EXISTS paper_started_at  TIMESTAMPTZ;
ALTER TABLE strategy_registry ADD COLUMN IF NOT EXISTS paper_finished_at TIMESTAMPTZ;

-- Final test on the held-back data (added 2026-10). campaign is the holdout
-- start date the strategy was tested against.
ALTER TABLE strategy_registry ADD COLUMN IF NOT EXISTS campaign          TEXT;
ALTER TABLE strategy_registry ADD COLUMN IF NOT EXISTS final_test_passed BOOLEAN;
ALTER TABLE strategy_registry ADD COLUMN IF NOT EXISTS final_test_stats  JSONB;
