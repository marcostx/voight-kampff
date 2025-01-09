# Movie Recommender Architecture

## Code Structure

```
movie_recommender/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # Handles data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── collaborative_filtering.py  # Implements recommendation algorithm
│   ├── utils/
│   │   └── __init__.py        # Utility functions (future use)
│   └── api/
│       ├── __init__.py
│       └── server.py          # FastAPI endpoints
└── data/
    ├── raw/                   # Original MovieLens dataset
    └── processed/             # Processed data files
```

## Data Pipeline

1. Data Loading
   ```
   MovieLens Dataset (CSV)
   └── MovieDataLoader
       ├── movies.csv → DataFrame
       └── ratings.csv → DataFrame
           └── create_user_item_matrix()
               └── User-Item Matrix (numpy array)
   ```

2. Model Pipeline
   ```
   User-Item Matrix
   └── CollaborativeFilter
       ├── fit() → Item-Item Similarity Matrix
       └── get_recommendations()
           └── Top-N Similar Movies
   ```

3. API Flow
   ```
   HTTP Request
   └── FastAPI Server
       ├── Load Data
       ├── Train Model
       └── /recommendations/{movie_id}
           └── JSON Response with Similar Movies
   ```

## Components Description

### Data Loading (loader.py)
- Handles reading and processing of MovieLens dataset
- Creates user-item matrix for collaborative filtering
- Uses type hints for all function parameters and returns

### Recommendation Model (collaborative_filtering.py)
- Implements item-based collaborative filtering
- Uses cosine similarity for movie-movie similarity computation
- Provides typed interfaces for model training and prediction

### API Server (server.py)
- RESTful endpoints for movie recommendations
- Pydantic models for request/response validation
- Proper error handling and type safety
