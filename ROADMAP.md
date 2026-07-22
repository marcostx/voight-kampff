# Voight-Kampff Roadmap

> "I've seen things you people wouldn't believe..." — and this roadmap is how we make sure
> this recommender isn't one of them, lost in time, like tears in rain.

This document tracks the milestones for evolving Voight-Kampff from a prototype recommender
into a polished CLI-first movie recommendation tool. Milestones are named after replicant
generations — each Nexus is more capable (and more human) than the last.

---

## Nexus-1 — "The Baseline Test" (stabilize & correctness)

*Before an agent goes into the field, it must pass its baseline. Fix what's broken first.*

- [x] **Fix the movie ID misalignment bug (critical).** `create_user_item_matrix()` pivots
      ratings into 9,724 columns (movies that have ratings, ordered by `movieId`), but the
      model is fit with all 9,742 IDs from `movies.csv`. The orderings diverge from index
      816 onward, so similarity scores get attributed to the wrong movies. Return the
      pivot's column index from the loader and fit with that. *A recommender that confuses
      one movie for another has failed its empathy test.*
- [x] **Write a real test suite.** `tests/` is currently empty and `test_recommender.py` is a
      print-script. Port it to `pytest` with assertions: loader shape checks, matrix/ID
      alignment (regression test for the bug above), known-similarity fixtures, API tests
      via `TestClient`. *"Is this to be an empathy test?" — Yes. Administer it on every commit.*
- [x] **Fix CI.** The Pylint workflow tests Python 3.8–3.10 while the README requires 3.12+;
      align the matrix, add a `pytest` job, and add a `pylintrc`/`ruff` config so linting is
      deterministic. Remove or properly configure the mdBook and Jekyll workflows (there is
      currently no book or site for them to build — they're chasing phantom replicants).
- [x] **Proper packaging.** Replace the bare `setup.py` with a `pyproject.toml` carrying
      dependencies, metadata, and console entry points; kill the `sys.path.insert` hack in
      `server.py`. Configure the appropriate package registry/index for the environment the
      project is developed in.
- [x] **Repo hygiene.** Ignore/remove `.DS_Store` files, `__pycache__`, and `.idea`; keep
      `.gitignore` authoritative.

## Nexus-2 — "Deckard" (a real CLI)

*This is billed as a CLI solution, but right now there's no CLI — only an API server and a
script. Time to put a blade runner on the street.*

- [x] **Scaffold the `vk` command** with Typer (or Click): entry point installed via
      `pyproject.toml`, `--help` that opens with the Voight-Kampff banner.
- [x] **`vk interrogate <movie>`** — the core command: given a movie (ID or title), return
      the top-N similar movies. *You sit the movie down, you ask it questions, you watch
      which memories respond.*
- [ ] **`vk retire <movie>`** — exclude a movie (or list) from future recommendations.
- [ ] **`vk search <title>`** — fuzzy title lookup so users never need to know raw
      `movieId`s. *An Esper machine for the catalog: "Enhance 224 to 176."*
- [x] **`vk train`** — build and persist the similarity matrix as an artifact, stamped with
      its **incept date** (dataset hash + version metadata).
- [x] **`vk serve`** — launch the FastAPI server from the CLI instead of `python src/api/server.py`.
- [x] **CLI ergonomics:** `--json` output for piping, meaningful exit codes, Rich-powered
      tables/progress, shell completion, and a config file (`~/.config/voight-kampff/`).

## Nexus-3 — "The Esper Machine" (model & data improvements)

*Enhance. Stop. Move in. Enhance. Better inputs, better memories, better recommendations.*

- [ ] **Model persistence.** Stop retraining on every server startup: load the artifact
      produced by `vk train`; retrain only when the dataset's incept date changes.
- [ ] **Sparse matrices.** Replace the dense `fillna(0)` pivot with `scipy.sparse` so the
      full MovieLens dataset (not just `ml-latest-small`) fits in memory.
- [ ] **Adjusted cosine / mean-centering.** Raw cosine on zero-filled ratings has heavy
      popularity bias; center by user mean so we measure taste, not fame. *More human than
      human is our motto — the model should understand people, not just count them.*
- [ ] **Evaluation harness.** Train/test split with precision@k, recall@k, and coverage —
      the model's own baseline test, run in CI so regressions are caught before deployment.
      *"Cells. Interlinked. Within cells interlinked."*
- [ ] **User-based recommendations.** `vk interrogate --user <id>`: recommend from a user's
      rating history, not just item-item similarity.
- [ ] **Hybrid signals.** Blend genre/tag content features to handle cold-start movies —
      *new replicants with implanted memories still deserve recommendations.*

## Nexus-4 — "Tyrell Corporation" (API & operations hardening)

*Commerce is our goal here. Make the service something you could actually run.*

- [ ] **Lifespan-managed startup:** move data/model loading into FastAPI lifespan events
      with a `/health` endpoint that reports model incept date and dataset version.
- [ ] **API polish:** typed error responses, pagination, `GET /movies?search=`,
      OpenAPI examples, and API versioning (`/v1/`).
- [ ] **Containerization:** Dockerfile + compose file. *Build your own replicant, ship it
      anywhere — even the off-world colonies.*
- [ ] **Observability:** structured logging and basic request/latency metrics. *Gaff always
      leaves an origami trail; so should every request.*
- [ ] **Input hardening:** validate `n_recommendations` bounds, rate-limit public endpoints,
      and review dependency pins for known CVEs.

## Nexus-5 — "More Human Than Human" (UX, docs & release)

- [ ] **Rewrite the README** around the CLI: quickstart, demo GIF of `vk interrogate`,
      corrected project name (it still says `movie_recommender/`), and accurate claims.
- [ ] **Real documentation site** (mdBook or MkDocs) — finally giving the existing docs
      workflows a purpose: architecture, CLI reference, model methodology.
- [ ] **Themed terminal experience:** optional `--noir` output theme, VK-test-styled
      interactive mode ("Describe in single words only the good things that come into your
      mind about... your favorite movie.").
- [ ] **Versioned releases** with changelogs (`Nexus-6.0.0` has a nice ring to it) and a
      published package.

## Beyond the Tannhäuser Gate (stretch goals)

*Things this project could see that most recommenders wouldn't believe.*

- [ ] Matrix factorization / ALS with implicit feedback.
- [ ] Embedding-based similarity with an approximate nearest-neighbor index.
- [ ] A "memory maker" module: per-user profiles that persist and evolve between sessions.
- [ ] A **Joi** companion mode: conversational recommendations on top of the engine.

---

> "The light that burns twice as bright burns half as long."
> Ship small, ship steadily — this project should burn very, very long.
