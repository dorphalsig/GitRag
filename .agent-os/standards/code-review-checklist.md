## Basic Health
  * [ ] All tests pass & Coverage >= Threshold.
  * [ ] No secrets/credentials in repo or diffs.
  * [ ] No SQL/OS/remote-exec injection risks.
  * [ ] No public API break without documented migration.

## Scope

  * [ ] Change matches task / feature description.
  * [ ] No unrelated files or formatting churn in diff.
  * [ ] New work beyond scope.

##  Tests

  * [ ] Unit tests for new logic exist.
  * [ ] Edge cases and error paths covered.
  * [ ] No skipped/xfailed tests without bug ticket.
  * [ ] Contract/integration tests included for cross-service changes.

## Correctness

  * [ ] Inputs validated and sanitized.
  * [ ] Boundary/null/empty cases handled.
  * [ ] Errors logged and surfaced; not silently swallowed.
  * [ ] Transactions/atomicity/locking considered where needed.

## Design & Tech-Debt

  * [ ] Single responsibility per function/class.
  * [ ] No new global state or hidden side effects.
  * [ ] DB/schema changes include migration and rollback plan.
  * [ ] Any shortcut has a tracked tech-debt ticket with owner.

## Code Quality / Smells

  * [ ] No duplicated logic; reuse existing abstractions.
  * [ ] Functions ≤ 60 LOC and low nesting.
  * [ ] Clear, explicit names; no magic literals.
  * [ ] No commented-out code left in diffs.

## Performance

  * [ ] Check for N+1 queries and large-memory loops.
  * [ ] Large payloads streamed or paginated.
  * [ ] Indexes considered for new DB predicates.

## Security & Auth

  * [ ] Auth checks present on admin/privileged endpoints.
  * [ ] Principle of least privilege applied for resources.
  * [ ] No eval/untrusted template rendering.
  * [ ] Dependencies reviewed for known critical vulns (if applicable).

## CI / Deployment / Observability

  * [ ] CI covers tests, lint, and contract checks.
  * [ ] Health/readiness endpoints present where required.
  * [ ] Logs include request/context IDs and avoid PII.
  * [ ] Migrations and deploy steps documented.

## Docs & Release

  * [ ] README or module doc updated for user-visible changes.
  * [ ] Env vars and defaults documented.
  * [ ] CHANGELOG or PR description explains user impact.

## Merge decision

  * [ ] If scope creep detected: split PR or create a ticket before merge.
  * [ ] If temporary debt accepted: require tech-debt ticket with owner and ETA.