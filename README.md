# Credit Risk FYP

Explainable, fair, and drift-aware credit risk scoring system.
Student: Pujan Malla Thakuri.

## Setup

1. Place the raw Home Credit CSVs (`application_train.csv`, `bureau.csv`,
   `bureau_balance.csv`, `previous_application.csv`,
   `POS_CASH_balance.csv`, `credit_card_balance.csv`,
   `installments_payments.csv`, `HomeCredit_columns_description.csv`)
   into `data/raw/`. These are gitignored — never commit them.
2. Install [Claude Code](https://docs.claude.com/en/docs/claude-code/setup)
   if you haven't already (`npm install -g @anthropic-ai/claude-code` or
   see the docs for other install methods).
3. From this project's root directory, run:
   ```
   claude
   ```
4. Once in a session, run `/context` and confirm `CLAUDE.md` appears
   under "Memory files" — this confirms the project instructions loaded.
5. Read `docs/roadmap.md` fully before starting Phase 1 work. Consult
   `docs/reading_material.md` for concept explanations as needed.

## Repo layout

See `CLAUDE.md` for the full project-instructions and layout reference —
that file is what Claude Code reads automatically at the start of every
session, so it's kept current as the source of truth for conventions.

## Project-specific skills

Three skills are set up under `.claude/skills/`:
- `synthetic-drift-injection` — builds the synthetic period/bias overlay
- `leakage-check` — pre-flight checklist before finalizing modeling code
- `shap-stability-eval` — runs the SHAP method comparison metrics

Invoke directly with e.g. `/leakage-check`, or let Claude invoke them
automatically when the context matches (all except
`synthetic-drift-injection`, which requires manual invocation since it
has side effects on the data layer).

## Personal preferences

Copy `CLAUDE.local.md.example` to `CLAUDE.local.md` (gitignored) for any
machine-specific or personal preferences that shouldn't be shared with
anyone else who clones this repo.
