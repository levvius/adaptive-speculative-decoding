```markdown
# adaptive-speculative-decoding Development Patterns

> Auto-generated skill from repository analysis

## Overview
This skill teaches you how to contribute effectively to the `adaptive-speculative-decoding` Python codebase. You'll learn the project's coding conventions, commit message patterns, and structured workflows for updating documentation, implementing or refactoring core features, and maintaining tests. This guide ensures consistency and quality across contributions, making collaboration smooth and efficient.

## Coding Conventions

- **File Naming:**  
  Use `snake_case` for all Python files and modules.
  ```
  jointadaspec/utils/probs.py
  sp_samp/hf_adapter.py
  ```

- **Import Style:**  
  Prefer **relative imports** within packages.
  ```python
  from .utils import some_function
  from ..mdp import estimation
  ```

- **Export Style:**  
  Use **named exports** (explicitly define what is exported from each module).
  ```python
  __all__ = ["SpeculativeDecoder", "estimate_probs"]
  ```

- **Commit Messages:**  
  Follow the **Conventional Commits** format with these prefixes:
    - `docs:` for documentation changes
    - `feat:` for new features
    - `refactor:` for code restructuring
    - `test:` for test-related changes
    - `chore:` for maintenance tasks

  Example:
  ```
  feat: add cascade verification baseline implementation
  ```

## Workflows

### Documentation and Project Guidance Update
**Trigger:** When you need to revise documentation, mark legacy results, or update project/thesis guidance.  
**Command:** `/update-docs`

1. Edit or add documentation files in `docs/`, `papers/`, or `reports/`.
2. Update `README.MD` or `CLAUDE.md` as needed.
3. Commit changes with a `docs:` prefix in the message.
   ```
   docs: update thesis structure and add new summary
   ```
4. Submit a pull request for review.

**Files commonly involved:**
- `README.MD`
- `CLAUDE.md`
- `docs/RESULTS.md`
- `papers/build_thesis_docx.py`
- `reports/thesis_final_summary_2026-05-20.md`
- ...and other documentation/report files

---

### Core Feature or Refactor with Tests
**Trigger:** When implementing a new core feature or refactoring core logic, and updating or adding tests.  
**Command:** `/core-refactor`

1. Modify or add core implementation files in `jointadaspec/` or `sp_samp/`.
2. Update or add corresponding test files in `tests/`.
3. Commit changes with a `feat:` or `refactor:` prefix.
   ```
   feat: implement adaptive speculative decoding algorithm
   refactor: simplify probability estimation logic
   ```
4. Ensure all tests pass before submitting a pull request.

**Files commonly involved:**
- `jointadaspec/utils/probs.py`
- `sp_samp/hf_adapter.py`
- `jointadaspec/core/sd_base.py`
- `tests/test_inference.py`
- `tests/test_jointadaspec.py`
- ...and related core/test files

---

## Testing Patterns

- **Test File Naming:**  
  Test files are named using the pattern `test_*.py` and placed in the `tests/` directory.
  ```
  tests/test_verification.py
  tests/test_mdp_solver.py
  ```

- **Framework:**  
  The specific testing framework is not detected, but standard Python test structure applies (e.g., `pytest` or `unittest`).

- **Example Test:**
  ```python
  def test_cascade_verification():
      result = cascade_verification(input_data)
      assert result == expected_output
  ```

## Commands

| Command        | Purpose                                                        |
|----------------|----------------------------------------------------------------|
| /update-docs   | Start a documentation or project guidance update workflow      |
| /core-refactor | Begin a core feature addition or refactor with tests workflow  |
```
