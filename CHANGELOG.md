# Changelog

All notable changes to the Voight-Kampff System are documented in this file —
memories worth keeping, so they don't get lost in time, like tears in rain.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `vk` command-line interface scaffolded with Typer: a `vk` console entry
  point (installed via `pyproject.toml`) whose `--help` opens with the
  Voight-Kampff banner, plus a `--version`/`-V` flag reporting the incept
  date. The first step of the Nexus-2 "Deckard" milestone — the subcommands
  (interrogate, retire, search, train, serve) follow
- Real `pytest` test suite (16 tests) replacing the `test_recommender.py`
  print-script: loader shape checks, matrix/ID alignment regression tests,
  known-similarity fixtures for the model, and API tests via `TestClient`
  ([#4](https://github.com/marcostx/voight-kampff/issues/4),
  [PR #36](https://github.com/marcostx/voight-kampff/pull/36))
- `.pylintrc` configuration; the codebase now lints clean at 10/10
  ([#5](https://github.com/marcostx/voight-kampff/issues/5),
  [PR #37](https://github.com/marcostx/voight-kampff/pull/37))
- `pyproject.toml` packaging with metadata, dependencies, a `dev` extra, and
  the `voight-kampff-server` console entry point
  ([#6](https://github.com/marcostx/voight-kampff/issues/6))
- Project roadmap (`ROADMAP.md`) with Nexus-themed milestones, mirrored as
  [GitHub issues and milestones](https://github.com/marcostx/voight-kampff/milestones)

### Changed

- Package moved to a src layout under a real import name: `src/voight_kampff/`
  instead of a top-level package literally named `src`; the `sys.path.insert`
  hack in `server.py` is gone
  ([#6](https://github.com/marcostx/voight-kampff/issues/6))
- CI rebuilt as a single workflow: pylint job plus a pytest matrix on
  Python 3.11–3.13, replacing the Pylint workflow that tested 3.8–3.10
  ([#5](https://github.com/marcostx/voight-kampff/issues/5),
  [PR #37](https://github.com/marcostx/voight-kampff/pull/37))
- Supported Python is now declared as 3.11+ (`requires-python`), matching
  what the test suite actually verifies
  ([#6](https://github.com/marcostx/voight-kampff/issues/6))
- README install/run/test instructions updated for packaged usage
  (`pip install -e .`, `voight-kampff-server`)
- Repo hygiene: `.gitignore` is now authoritative — ignores macOS
  `.DS_Store`, the `.idea/` IDE folder, and the local `data/` directory
  explicitly, replacing a blanket `*.csv` pattern that would also have
  hidden future test fixtures; stray Finder metadata deleted and
  `ROADMAP.md` checked in
  ([#8](https://github.com/marcostx/voight-kampff/issues/8))

### Fixed

- **Movie ID misalignment (critical):** the user-item matrix only contains
  movies that have ratings (9,724 columns), but the model was fit with all
  9,742 IDs from `movies.csv`, attributing similarity scores to the wrong
  movies from index 816 onward. The loader now returns the matrix together
  with its aligned movie IDs — every recommendation now passes its empathy
  test ([#3](https://github.com/marcostx/voight-kampff/issues/3),
  [PR #35](https://github.com/marcostx/voight-kampff/pull/35))
- Self-recommendation bug: the query movie was excluded from results by
  assuming it sorted first, which breaks when other movies tie at
  similarity 1.0; it is now excluded by index
  ([PR #36](https://github.com/marcostx/voight-kampff/pull/36))
- Proper exception chaining (`raise ... from e`) in the recommendations
  endpoint ([PR #37](https://github.com/marcostx/voight-kampff/pull/37))

### Removed

- `setup.py`, `requirements.txt`, and `requirements-dev.txt` — retired in
  favor of `pyproject.toml`
  ([#6](https://github.com/marcostx/voight-kampff/issues/6))
- Unused `matplotlib` and `seaborn` dependencies — nothing imports them
  ([#6](https://github.com/marcostx/voight-kampff/issues/6))
- Orphaned CI workflows: mdBook and Jekyll (no book or site to build) and
  EthicalCheck (it scanned apisec's demo API, not this project)
  ([#5](https://github.com/marcostx/voight-kampff/issues/5),
  [PR #37](https://github.com/marcostx/voight-kampff/pull/37))
