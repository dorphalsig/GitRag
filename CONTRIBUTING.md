# Contributing to GitRag

Thanks for investing time in GitRag! The project aims to stay approachable while
maintaining deterministic indexing guarantees. Please follow the guidelines
below when proposing changes.

## Development Workflow

1. **Set up the virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the test suite locally**
   ```bash
   .venv/bin/python -m pytest -vv
   ```
   Tests stub optional dependencies (Tree-sitter, libSQL driver) where
   possible. When touching persistence code, run the live libSQL test by
   providing the `CLOUDFLARE_*` variables.

3. **Linting** is currently handled via CI formatting checks. Please keep
   imports clean and add targeted comments only when logic is non-obvious.

4. **Commit scope**: prefer focused commits per feature/fix. Each commit message
   should describe motivation and high-level approach (e.g., "Add CSV chunker
   metadata").

## Making Larger Changes

- **Chunker updates** must include fixtures + tests that demonstrate the new
  behavior. For document chunking, add inputs to `tests/fixtures` and assertions
  in `tests/test_chunker_documents.py`.
- **Persistence adapters** should register via `persistence_registry` so they
  can be swapped without touching orchestration. Add tests covering
  registration/usage.
- **CLI features** should expose flags with descriptive help text and include
  coverage in `tests/test_indexer_cli.py`.

## Filing Issues & Pull Requests

- Use GitHub issues to propose enhancements or report bugs. Include reproduction
  steps where possible.
- Draft PRs are welcome for early feedback. When ready, ensure all tests pass
  and the change log (README/mission docs) reflects user-facing updates.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/).
By participating, you agree to uphold a welcoming, harassment-free environment.

## License

By contributing, you agree that your contributions are licensed under the terms
of the [MIT License](LICENSE) with GitRag attribution.
