# AgentFuse Post-Fix Audit Report & Round 2 Fix Guide

> **Date:** March 21, 2026
> **Repo:** github.com/vinaybudideti/agentfuse (commit 645ed55)
> **Previous fixes:** All 17 fixes from Round 1 verified: 20/20 checks PASS

---

## PART A: Fix Verification Results (All 17 PASS)

| Fix | File | Status |
|-----|------|--------|
| 1 | gateway.py: _rate_limiter not overwritten | PASS |
| 2 | gateway.py: configure() has all 3 globals | PASS |
| 3 | gateway.py: _validate_and_cache checks guardrails | PASS |
| 4 | pyproject.toml + __init__.py: version 0.2.1 | PASS |
| 5 | providers/openai.py: stream validates before cache | PASS |
| 6 | providers/anthropic.py: stream validates before cache | PASS |
| 7 | gateway.py: acompletion has fallback chain | PASS |
| 8 | gateway.py: uses get_running_loop | PASS |
| 9 | gateway.py: async injects stream_options | PASS |
| 10 | gateway.py: client cache exists and used | PASS |
| 11 | core/cache.py: _redis_failures in __init__ | PASS |
| 12 | core/session.py: async context manager | PASS |
| 13 | examples/*.py: all have real code | PASS |
| 14 | package-lock.json deleted, .gitignore updated | PASS |
| 15 | examples/e2e_real_test.py exists | PASS |
| 16 | CI has coverage threshold check | PASS |
| 17 | _load_balancer declared at module top | PASS |

**All 17 fixes correctly implemented. Zero errors in the applied changes.**

---

## PART B: NEW Issues Found (Post-Fix Deep Audit)

### CRITICAL 1: Published PyPI Package Would Include Sensitive Files

**Problem:** When you run `python3 -m build`, the sdist (the file uploaded to PyPI) includes files that should never be published:

- `.claude/settings.local.json` (Claude Code settings)
- `CLAUDE.md` (contains your local filesystem path `/Users/vinaykumarreddy/Documents/llm-project-2026/Project_Building`)
- `_project/BUILD_LOG.md`, `DEEP_RESEARCH_DOC.md`, `FEATURE_IDEAS.md`, `FIX_GUIDE.md`, `HELP_NEEDED.md`, `RESEARCH_QUESTIONS.md`, `TODO.md` (internal planning docs)
- `tests/` (125 test files, 1.5MB unnecessary for users)
- `agentfuse/intent_atoms/dashboard/` (entire React app, 20 files)
- `agentfuse/intent_atoms/benchmarks/` (21 benchmark result files)
- `agentfuse/intent_atoms/setup.py` (conflicting setup.py inside the package)

Total sdist: 4.0MB (should be under 500KB)

**Fix:** Create a `MANIFEST.in` file in the project root:

**Create new file:** `MANIFEST.in`

```
include LICENSE
include README.md
include pyproject.toml

recursive-include agentfuse *.py *.typed
recursive-exclude agentfuse/intent_atoms/dashboard *
recursive-exclude agentfuse/intent_atoms/benchmarks *

exclude CLAUDE.md
exclude PROGRESS.md
exclude CHANGELOG.md
prune .claude
prune _project
prune tests
prune examples
```

Also add to `pyproject.toml` after the `[tool.setuptools.packages.find]` section:

**Find this exact text:**

```
[tool.setuptools.packages.find]
include = ["agentfuse*"]
```

**Replace with this exact text:**

```
[tool.setuptools.packages.find]
include = ["agentfuse*"]

[tool.setuptools.package-data]
agentfuse = ["py.typed"]

[tool.setuptools.exclude-package-data]
"*" = ["*.json", "*.css", "*.jsx", "*.svg", "*.html"]
```

---

### CRITICAL 2: CLAUDE.md Contains Your Local Filesystem Path

**Problem:** Line 74 of CLAUDE.md contains:
```
cd /Users/vinaykumarreddy/Documents/llm-project-2026/Project_Building
```

This is your personal machine's full path. It's committed to a public repo and would be included in the PyPI package without MANIFEST.in.

**Fix:** In `CLAUDE.md`, find this exact text:

```
cd /Users/vinaykumarreddy/Documents/llm-project-2026/Project_Building
source .venv/bin/activate
```

Replace with:

```
cd /path/to/agentfuse
source .venv/bin/activate
```

---

### CRITICAL 3: setuptools-scm Is a Build Dependency But Not Used

**Problem:** `pyproject.toml` declares `setuptools-scm` as a build requirement, but the version is hardcoded (not derived from git tags). This means anyone building from source needs to install setuptools-scm for no reason, and it will fail if they do not have it.

**File:** `pyproject.toml`

**Find this exact text:**

```
requires = ["setuptools>=68", "setuptools-scm"]
```

**Replace with this exact text:**

```
requires = ["setuptools>=68"]
```

---

### HIGH 1: README Has Conflicting Test Counts

**Problem:** The README contains three different test counts:
- Badge: `tests-1022%20passing`
- Key Numbers table: `1022 unit tests, 93% core coverage`
- Changelog v0.2.0 section: `260 unit tests, 0 construction-only tests, 86% core module coverage`

Actual test function count: 1162 test functions across 104 files.
Claude Code reported: 1080 tests passed in the fix run.

**Fix:** In `README.md`, update the badge. Find:

```
![Tests](https://img.shields.io/badge/tests-1022%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/core%20coverage-88%25-green)
```

Replace with:

```
![Tests](https://img.shields.io/badge/tests-1080%20passing-brightgreen)
![Coverage](https://img.shields.io/badge/core%20coverage-86%25-green)
```

Also update the Key Numbers table. Find:

```
| Test suite | 1022 unit tests, 93% core coverage |
```

Replace with:

```
| Test suite | 1080 unit tests, 86% core coverage |
```

---

### HIGH 2: README Changelog Missing v0.2.1

**Problem:** README has changelog sections for v0.2.0 and v0.1.0 but no v0.2.1 section, even though pyproject.toml and CHANGELOG.md both reference v0.2.1.

**Fix:** In `README.md`, find this exact text:

```
### v0.2.0 — Production Rebuild (March 2026)
```

Add this ABOVE it:

```
### v0.2.1 — Production Fixes (March 2026)

- Fixed: environment variable rate limiting (`AGENTFUSE_RATE_LIMIT_RPS`) now works correctly
- Fixed: `configure(output_guardrails=...)` now properly sets module-level guardrails
- Fixed: output guardrails are checked before caching responses
- Fixed: streaming responses validated before cache storage (both OpenAI and Anthropic)
- Fixed: `acompletion()` now has automatic fallback chain matching sync `completion()`
- Fixed: async provider uses `asyncio.get_running_loop()` (deprecated `get_event_loop` removed)
- Fixed: async streaming injects `stream_options` for OpenAI usage tracking
- Added: SDK client caching across requests (reuses connection pools)
- Added: `AgentSession` async context manager (`async with`)
- Added: real-world end-to-end test script (`examples/e2e_real_test.py`)
- Added: CI coverage threshold enforcement
- Filled: all example files with working code
- 85 public exports, 1080 tests passing

### v0.2.0 — Production Rebuild (March 2026)
```

---

### HIGH 3: CLAUDE.md Claims 79 Exports, Actual Is 85

**Problem:** CLAUDE.md says "79 public exports" but `len(agentfuse.__all__)` returns 85.

**File:** `CLAUDE.md`

**Find this exact text:**

```
- **Exports:** 79 public API functions/classes
```

**Replace with this exact text:**

```
- **Exports:** 85 public API functions/classes
```

Also find:

```
- `agentfuse/__init__.py` — all 79 public exports
```

Replace with:

```
- `agentfuse/__init__.py` — all 85 public exports
```

---

### HIGH 4: PROGRESS.md Claims 79 Exports and 1022 Tests

**File:** `PROGRESS.md`

**Find this exact text:**

```
**1022 unit tests | 79 exports | 93% core coverage | 250 commits | ALL GREEN | 0 CVEs**
```

**Replace with this exact text:**

```
**1080 unit tests | 85 exports | 86% core coverage | ALL GREEN | 0 CVEs**
```

---

### MEDIUM 1: Quickstart Shows Both Old and New API

**Problem:** README quickstart shows the `completion()` gateway (new API), but further down shows `wrap_openai()` as the "Integration Example" (old API). The comparison table and feature list reference the old API more prominently. This confuses users about which API to use.

**Fix:** No code change needed. Just be aware that when you write docs, `completion()` should be the primary recommended API and `wrap_openai()`/`wrap_anthropic()` should be labeled as "Legacy API."

---

### MEDIUM 2: intent_atoms Has Conflicting setup.py

**Problem:** `agentfuse/intent_atoms/setup.py` declares a separate package called `intent-atoms` with its own dependencies (anthropic, fastapi, uvicorn). When `agentfuse-runtime` is installed, intent_atoms comes along as a subdirectory but its setup.py is never executed. It just sits there confusing people.

**Fix:** Either remove `agentfuse/intent_atoms/setup.py` or rename it to `_setup.py.bak` so it is not mistaken for an active build config. The MANIFEST.in from Critical 1 will exclude it from the sdist.

---

### MEDIUM 3: Gateway Has Stale Comment From Fix 17

**File:** `agentfuse/gateway.py`

Claude Code left a comment that references the fix process:

**Find this exact text:**

```
# NOTE: _load_balancer is initialized above with the other module-level state.
# It was previously declared here after the function that uses it.
```

**Replace with this exact text:**

```
```

(Delete the two comment lines entirely. They reference internal fix history that should not be in production code.)

---

### LOW 1: PyPI Badge Shows "not found"

**Problem:** The README has `![PyPI](https://img.shields.io/pypi/v/agentfuse-runtime)` but the package is not published yet, so this badge shows "not found" on GitHub. It looks broken.

**Fix:** Either remove the badge until you publish, or leave it. It will auto-resolve once you upload to PyPI.

---

## PART C: Prerequisites for Real-World Testing

Before publishing to PyPI, you need to:

1. **Set API keys:**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

2. **Run the real-world test:**
```bash
cd Project_Building
source .venv/bin/activate
python3 examples/e2e_real_test.py
```

3. **Verify the build is clean (after applying fixes above):**
```bash
rm -rf dist/ build/ *.egg-info/
python3 -m build
twine check dist/*
# Verify size is reasonable (should be under 500KB with MANIFEST.in)
ls -lh dist/*.tar.gz
```

4. **Test install in a fresh venv:**
```bash
python3 -m venv /tmp/test-agentfuse
source /tmp/test-agentfuse/bin/activate
pip install dist/agentfuse_runtime-0.2.1.tar.gz
python3 -c "from agentfuse import completion; print('OK')"
deactivate
rm -rf /tmp/test-agentfuse
```

5. **Upload to TestPyPI first:**
```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ agentfuse-runtime
```

6. **If TestPyPI works, upload to real PyPI:**
```bash
twine upload dist/*
```

---

## PART D: Summary

| # | File | Type | Severity |
|---|------|------|----------|
| C1 | MANIFEST.in + pyproject.toml | Sensitive files in PyPI package | CRITICAL |
| C2 | CLAUDE.md | Local path exposed publicly | CRITICAL |
| C3 | pyproject.toml | Unnecessary setuptools-scm dep | CRITICAL |
| H1 | README.md | Conflicting test counts | HIGH |
| H2 | README.md | Missing v0.2.1 changelog | HIGH |
| H3 | CLAUDE.md | Wrong export count | HIGH |
| H4 | PROGRESS.md | Wrong test/export counts | HIGH |
| M1 | README.md | Old vs new API confusion | MEDIUM |
| M2 | intent_atoms/setup.py | Conflicting build config | MEDIUM |
| M3 | gateway.py | Stale fix comment | MEDIUM |
| L1 | README.md | PyPI badge shows not found | LOW |

**Priority order:** C1, C3, C2, H1, H2, H3, H4, M3, M2
