---
paths:
  - "data/**/*"
  - "sql/**/*"
---

# Data Handling Rules

- `data/raw/` is read-only. Never write, overwrite, or modify files here.
  If a transformation is needed, write the output to `data/processed/` or
  `data/synthetic/` instead.
- `data/synthetic/` must always be accompanied by an injection log
  (see the `synthetic-drift-injection` skill) — never generate synthetic
  period data without logging what was injected, where, and at what
  magnitude.
- Any SQL against the raw tables should be written assuming no absolute
  calendar time exists (see CLAUDE.md) — do not write queries that order
  by `SK_ID_CURR` and treat the result as chronological.
