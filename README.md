# TOTO Predictor SG

A production-grade Singapore TOTO prediction system with multiple ML models, comprehensive statistical analysis, backtesting, and a deployed Streamlit web application.

## Disclaimer

**TOTO is a truly random lottery. No prediction model can guarantee wins.** Actual odds of Group 1 jackpot: 1 in 13,983,816. This system identifies historically common patterns and avoids statistically unlikely combinations, but cannot predict random outcomes with certainty. Play responsibly.

## Features

- **13+ Statistical Analyses**: Frequency, hot/cold, pairs/triplets, odd/even, high/low, sum range, gaps, group spread, day-of-week, temporal drift, burst/dormancy detection, and more
- **7 Prediction Models**: Weighted Scoring, Random Forest, LSTM/MLP Neural Net, Monte Carlo (1M simulations), Markov Chain, Cluster Analysis, Ensemble
- **3 Board Strategies**: Ensemble High Probability, Diversified/Contrarian, Maximum Prize Value (Anti-Sharing)
- **7 Hard Filters**: Sum range, odd/even balance, high/low balance, group spread, consecutive limit, no duplicates, cluster fit
- **Walk-Forward Backtesting**: Never trains on future data, statistical significance testing
- **Multi-Draw Strategy**: 4-draw rolling strategy maximizing number coverage
- **Interactive Dashboard**: 6-page Streamlit app with Plotly charts

## Tech Stack

- Python 3.12
- Streamlit (web framework)
- scikit-learn (ML models)
- Plotly (interactive charts)
- pandas, numpy, scipy (data processing)

## Quick Start

### 1. Clone and Install

```bash
git clone <repo-url>
cd toto-predictor
pip install -r requirements.txt
```

### 2. Collect Data

```bash
python -c "from src.scraper import collect_all_data; collect_all_data()"
```

### 3. Run Predictions

```bash
python scripts/run_prediction.py
```

### 4. Launch Web App

```bash
streamlit run app.py
```

### 5. Run Backtester

```bash
python -c "from src.backtester import run_backtest; from src.scraper import load_data; run_backtest(load_data())"
```

## Project Structure

```
toto-predictor/
├── data/                    # Generated data files
├── src/
│   ├── scraper.py          # Data collection
│   ├── analysis.py         # 13+ statistical analysis functions
│   ├── predictor.py        # Board generation & multi-draw strategy
│   ├── filters.py          # 7 hard elimination filters
│   ├── backtester.py       # Walk-forward backtesting engine
│   └── models/
│       ├── weighted_scoring.py   # 9-factor weighted scoring
│       ├── random_forest.py      # RF classifier
│       ├── lstm_model.py         # LSTM/MLP neural network
│       ├── monte_carlo.py        # 1M simulation Monte Carlo
│       ├── markov_chain.py       # 49x49 transition model
│       ├── cluster_analysis.py   # K-means clustering
│       └── ensemble.py           # Weighted consensus ensemble
├── scripts/
│   ├── update_data.py      # Data update pipeline
│   └── run_prediction.py   # Standalone prediction script
├── app.py                  # Streamlit web application
├── requirements.txt
├── Dockerfile
└── .github/workflows/update.yml
```

## Model Methodology

### Board 1 - Ensemble High Probability
Top 6 numbers from ensemble ranking, validated against all 7 hard filters.

### Board 2 - Diversified/Contrarian
3 hot numbers (recent 3-month frequency) + 3 overdue cold numbers. Zero overlap with Board 1.

### Board 3 - Maximum Prize Value
Avoids commonly picked numbers (birthday range 1-31, SG lucky numbers) to maximize prize if won.

### Ensemble Scoring
Each model contributes its top 15 ranked numbers. Consensus voting weighted by model performance determines final rankings.

## Data Update

```bash
python scripts/update_data.py
```

This fetches the latest draw, checks previous prediction accuracy, re-trains models, and generates new predictions.

## Deployment

### Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io
3. Connect repo, set main file to `app.py`
4. Deploy

### Docker
```bash
docker build -t toto-predictor .
docker run -p 8501:8501 toto-predictor
```

## License

MIT
