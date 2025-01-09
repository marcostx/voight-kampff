# Voight Kampff System

A simple yet organized movie recommendation system built with Python

## Features

- Item-based collaborative filtering recommendation engine
- FastAPI REST API for serving recommendations
- Comprehensive documentation and data pipeline visualization
- Type-hinted Python code for better maintainability
- Built using the MovieLens dataset

## Project Structure

```
movie_recommender/
├── src/
│   ├── data/          # Data processing and loading modules
│   ├── models/        # Recommendation models
│   ├── utils/         # Utility functions
│   └── api/           # FastAPI endpoints
├── docs/              # Documentation
│   ├── architecture.md       # System architecture details
│   ├── data_pipeline.py     # Pipeline visualization generator
│   └── data_pipeline.png    # Visual representation of data flow
├── tests/             # Unit tests
└── data/
    ├── raw/          # Raw dataset files
    └── processed/    # Processed dataset files
```

## Requirements

- Python 3.12+
- Virtual environment (recommended)
- MovieLens dataset (automatically downloaded)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd movie_recommender
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the API Server

Start the FastAPI server:
```bash
python src/api/server.py
```

The API will be available at `http://localhost:8000`

### Getting Recommendations

Use the `/recommendations/{movie_id}` endpoint to get movie recommendations:
```bash
curl http://localhost:8000/recommendations/1
```

### Testing

Run the test script to verify the system:
```bash
python test_recommender.py
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
