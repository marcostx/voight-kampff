# Voight-Kampff

> "I've seen things you people wouldn't believe..."

**Voight-Kampff** is a CLI-first movie recommendation system built on
item-based collaborative filtering over the
[MovieLens](https://grouplens.org/datasets/movielens/) dataset.

## Quickstart

```bash
pip install -e .

# Meet the blade runner
vk --help

# Interrogate a movie — by id or title — for its nearest neighbours
vk interrogate "Toy Story (1995)"
vk interrogate 1 --number 10
```

## Where to next

- **[Architecture](architecture.md)** — how the loader, model, and API fit together.

This documentation is intentionally small for now; a fuller CLI reference and
model methodology arrive with the Nexus-5 milestone on the project roadmap.
