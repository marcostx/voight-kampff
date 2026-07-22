# Voight Kampff System
<img width="733" height="412" alt="CNg0GG8x-13366_4643dd49216b67d9c617ceb260e45684" src="https://github.com/user-attachments/assets/6e369020-62c1-4b2e-aae7-0c7d7f4ae074" />

A simple yet organized movie recommendation system built with Python

## Features

- Item-based collaborative filtering recommendation engine
- FastAPI REST API for serving recommendations
- Comprehensive documentation and data pipeline visualization
- Type-hinted Python code for better maintainability
- Built using the MovieLens dataset

## Project Structure

```
voight-kampff/
├── src/
│   └── voight_kampff/
│       ├── data/      # Data processing and loading modules
│       ├── models/    # Recommendation models
│       ├── utils/     # Utility functions
│       └── api/       # FastAPI endpoints
├── docs/              # Documentation
│   ├── architecture.md       # System architecture details
│   ├── data_pipeline.py     # Pipeline visualization generator
│   └── data_pipeline.png    # Visual representation of data flow
├── tests/             # Unit tests
├── pyproject.toml     # Packaging, dependencies, and entry points
└── data/
    ├── raw/          # Raw dataset files
    └── processed/    # Processed dataset files
```

## Requirements

- Python 3.11+
- Virtual environment (recommended)
- MovieLens dataset in `data/raw/ml-latest-small/`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/marcostx/voight-kampff.git
cd voight-kampff
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Running the API Server

Start the FastAPI server from the repository root (it reads the dataset
from `data/raw/`):
```bash
vk serve
```

The API will be available at `http://localhost:8000`

The existing `voight-kampff-server` executable remains available for
backward compatibility.

### Getting Recommendations

Use the `/recommendations/{movie_id}` endpoint to get movie recommendations:
```bash
curl http://localhost:8000/recommendations/1
```

### Testing

Install the dev dependencies and run the test suite with pytest:
```bash
pip install -e ".[dev]"
pytest
```

## Documentation

- See `docs/architecture.md` for detailed system architecture
- View `docs/data_pipeline.png` for visual data flow
- All code includes comprehensive docstrings and type hints

## Data Source

This project uses the [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/) which includes:
- 100,000+ ratings
- 9,000+ movies
- 600+ users

## License

MIT License - Feel free to use this code for your own projects.
