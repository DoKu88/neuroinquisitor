# Red Flags

This folder tracks **known design concerns and latent issues** in the
NeuroInquisitor codebase — things that work today but warrant a
follow‑up decision before they bite a real user.

These are not bugs (file those as issues) and not roadmap items (those
live in the `SPRINT-*.md` docs). Think of this folder as a living
"things we should think harder about" list:

- Code paths that are correct for the current scope but will fail or
  underperform under realistic future workloads.
- Implicit assumptions that aren't yet documented or enforced.
- Performance / correctness trade‑offs we made deliberately and want to
  revisit later.

## Conventions

- One file per concern, named `NNN-short-slug.md` (zero‑padded, in
  rough order of discovery — not severity).
- Every file should answer:
  1. **What is the issue?** — minimal repro / pointer to the code.
  2. **Why does it matter?** — the realistic scenario where it bites.
  3. **What are the alternatives?** — including "do nothing".
  4. **Recommendation** — current best guess, explicitly marked as such.
  5. **Status** — `open`, `accepted`, `mitigated`, or `resolved`.

## Index

| ID | Title | Status |
|----|-------|--------|
| [001](./001-snapshot-tensor-conversion.md) | Snapshot tensor conversion path (`detach().cpu().numpy()`) | open |
