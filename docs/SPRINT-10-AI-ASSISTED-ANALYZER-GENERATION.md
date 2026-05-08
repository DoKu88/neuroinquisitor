# Sprint 10: AI-Assisted Analyzer Generation

**Goal**: Add optional AI-assisted analyzer generation while keeping the ecosystem open.

## Tasks
- [ ] `NI-ZETA-001` Define analyzer-generation contract.
  - Require generated output to include analyzer Python code, metadata, dependencies, explanation, and tests when possible.
  - Document and validate the contract with `pydantic` v2 schemas.
- [ ] `NI-ZETA-002` Add review-first generation flow.
  - Present generated code before execution.
  - Support saving generated code as temporary analyzer, local plugin, or exportable file.
  - Prevent automatic execution without explicit user approval.
- [ ] `NI-ZETA-003` Add isolated execution path.
  - Execute generated analyzers in an isolated worker process.
  - Restrict filesystem access to selected artifacts.
  - Enforce timeouts and memory limits.
- [ ] `NI-ZETA-004` Add reproducibility logging.
  - Persist prompt, model identifier, code hash, execution result, and output artifact references.
  - Ensure generated analyses can be rerun without LLM dependency.

## Testing
- Add schema validation tests for generated analyzer contracts via `pydantic` v2 models.
- Add UX/API tests for review-first flow and explicit execution consent.
- Add sandbox tests validating path restrictions, timeout handling, and memory guardrails.
- Add failure-path tests for invalid generated analyzers with actionable error reporting.
- Add reproducibility tests to verify reruns from logged artifacts/code hash.

## Definition of Done
- Generated analyzers conform to a documented, schema-validated contract.
- No generated code can execute without explicit user review and approval.
- Isolated execution protects core runtime and surfaces clear diagnostics.
- Reproducibility logs are complete enough to rerun analyses without model regeneration.
- AI-generated analyzers remain first-class, user-owned plugins compatible with public APIs.
- Generated analyzer contracts produce deterministic `model_dump` outputs for reproducible logging.
