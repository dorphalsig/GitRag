# Python Code Style Guide

## General Formatting
- Indentation: 4 spaces (never tabs)  
- Line length: ≤ 100 characters  
- Blank lines:  
  - 2 between top-level defs/classes  
  - 1 between logical sections  
- Naming:  
  - Modules/files → `snake_case.py`  
  - Packages → `lowercase`  
  - Classes/exceptions → `PascalCase`  
  - Functions/vars → `snake_case`  
  - Constants → `UPPER_SNAKE_CASE`  
  - Private/internal → `_leading_underscore`  
- Strings:  
  - Single quotes (`'foo'`) by default  
  - f-strings for interpolation (`f'{val}'`)  
  - Triple double quotes for docstrings  

## Comments & Docstrings
- Public modules, classes, and functions **must** have docstrings (Google style preferred)  
- Document **why**, not what  
- Update comments when code changes  
- No commented-out code  

## Imports
- Order: stdlib → third-party → local  
- Alphabetical within groups  
- One import per line  
- Avoid wildcard imports  

## Typing
- All public functions/methods: fully typed  
- Prefer `list[str]`, `dict[str, int]`, `Mapping`, etc.  
- Use `T | None` only if `None` is valid  
- Avoid `Any`; justify when unavoidable  

## Functions & Classes
- Functions ≤ 40 LOC (hard limit)  
- ≥ 10 LOC → docstring with Summary, Args, Returns, Raises  
- No mutable defaults (`None` sentinel instead)  
- Use `@dataclass` for data objects; `frozen=True` if possible  
- Keep class responsibilities cohesive  

## Exceptions & Logging
- Raise specific exceptions; no bare `except:`  
- Preserve context with `raise ... from e`  
- Logging: use `logging`, never `print` (except in CLIs)  
- Include context; never log secrets  

## I/O & Paths
- Use `pathlib.Path` over `os.path`  
- Always set encoding (`utf-8`)  
- Use context managers (`with`)  

## Async & Concurrency
- Prefer asyncio for IO-bound tasks  
- Never block the event loop  
- Only call `asyncio.run()` in entry points  

## Collections & Iteration
- Prefer comprehensions/generators  
- Use `enumerate`, `zip`, `itertools` for clarity  
- Avoid unnecessary intermediates  
