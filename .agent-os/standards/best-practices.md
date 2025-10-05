# Python Development Best Practices

## Core Principles

### Keep It Simple
- Implement code in the fewest lines possible  
- Avoid over-engineering solutions  
- Choose straightforward approaches over clever ones  

### Optimize for Readability
- Prioritize code clarity over micro-optimizations  
- Write self-documenting code with clear variable names  
- Add comments for "why" not "what"  

### DRY (Don't Repeat Yourself)
- Extract repeated business logic to private methods  
- Extract repeated helpers into utilities  
- Use consistent abstractions instead of copy/paste  

### File Structure
- Keep files focused on a single responsibility  
- Group related functionality together  
- Use consistent naming conventions  

---

## Dependencies
When adding third-party libraries:
- Pick actively maintained, popular projects  
- Check for recent commits (≤ 6 months), open issues, stars, and docs  
- Avoid niche/unmaintained packages  

---

## Testing Policy

### Mandatory Rules
- **Coverage:** ≥ 70% branch coverage (`--cov-branch`, `--cov-fail-under=70`). Mandatory for every file >=20 LOC
- **No Skipped Tests:** Skips and xfails are forbidden; any occurrence fails CI  
- **All Tests Must Pass:** No failing tests allowed; green build is mandatory  
### Best Practices
- Tests must be deterministic (no network/time flakiness)  
- Use fakes/mocks/fixtures for external dependencies  
- Cover both success and failure paths  

### Tooling Config

`pyproject.toml`:
`````toml
[tool.pytest.ini_options]
addopts = """
-q
--cov=.
--cov-branch
--cov-report=term-missing
--cov-report=xml
--cov-fail-under=70
--strict-config
--strict-markers
"""
xfail_strict = true
