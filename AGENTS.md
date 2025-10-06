# Agent OS wiring for Codex

You are working in a repository that uses **BuilderMethods Agent OS**.
Follow these rules and workflows exactly.

## What Agent OS is
Agent OS provides structured **standards** and **instructional workflows** for coding agents. Read and follow everything under:
- `.agent-os/standards/`  (style, testing, diffs, review)
- `.agent-os/instructions/`  (step-by-step workflows)
- `.agent-os/product/`  (where outputs/specs live)

## Slash-command → Workflow mapping
When the user types one of these, open the referenced instruction and follow it. Write outputs where specified.

- `/plan-product` → `.agent-os/instructions/core/plan-product.mdc` → write to `.agent-os/product/`
- `/analyze-product` → `.agent-os/instructions/core/analyze-product.mdc` → update `.agent-os/product/`
- `/create-spec` → `.agent-os/instructions/core/create-spec.mdc` → emit under `.agent-os/product/specs/`
- `/create-tasks` → `.agent-os/instructions/core/create-tasks.mdc` → emit under `.agent-os/product/tasks/`
- `/execute-tasks` → `.agent-os/instructions/core/execute-tasks.mdc` → implement tasks in repo
** Include any `@path` file references inside the prompt (treat them as includes) **

## Standards to enforce
- Use the project **tech stack** in `.agent-os/standards/tech-stack.md` (or `.agent-os/product/tech-stack.md` if present).
- Patches must be **unified diffs**; include a 3-line change summary (What changed / What didn’t / Why).
- Ask to confirm patch success before proceeding to follow-ups.
- Prefer tests-first and keep methods within repo limits (≤40 LOC when applicable).
- If <95% certain on a fact or external API behavior, search **only** official resources, if no info found ask for clarification with precise questions.

## Context handling:
- Before editing, scan relevant files and the instruction doc you’re following.
- Keep artifacts local to the repo; avoid external network calls unless asked.
- Save all generated specs and task lists under `.agent-os/product/…` as instructed by the workflow.
