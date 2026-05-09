---
name: Print commands instead of running them
description: User wants all shell commands printed for them to run, not executed via Bash tool
type: feedback
---

Print commands to the user instead of running them via the Bash tool. This includes test commands, lint, typecheck, pip install, etc.

**Why:** User prefers to run commands themselves in their own shell.

**How to apply:** When you would normally run `pytest`, `ruff`, `mypy`, `pip install`, or any other shell command, print the command instead of executing it with the Bash tool.
