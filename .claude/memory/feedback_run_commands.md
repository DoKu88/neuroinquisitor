---
name: Print commands instead of running them
description: User wants commands printed for them to run, not executed via Bash
type: feedback
---

Do not run commands via the Bash tool. Instead, print the command for the user to run themselves.

**Why:** User prefers to control command execution in their own terminal.

**How to apply:** Whenever a validation step, test run, install, or any other shell command would be appropriate, output the command as a code block instead of calling Bash.
