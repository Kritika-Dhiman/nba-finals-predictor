# 🏀 NBA Finals Winner Predictor

A machine learning project that predicts NBA game winners using an ensemble of three models trained on 5,242 real NBA games from 2022–2026. Built entirely in Python with real NBA data via `nba_api`.

---

## 📊 Model Performance

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 74.7% | 0.796 |
| Random Forest | 74.2% | 0.804 |
| XGBoost | 72.5% | 0.795 |
| **Ensemble (all 3)** | **~74%** | **~0.80** |

**Cross-validation (5-fold time series):**

| Model | Mean CV Accuracy | Std Dev |
|---|---|---|
| Logistic Regression | 65.1% | ±2.4% |
| Random Forest | 65.1% | ±2.4% |
| XGBoost | 63.7% | ±3.7% |

**Playoff-only accuracy (82 games):**

| Model | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 69.5% | 0.751 |
| Random Forest | 69.5% | 0.754 |
| XGBoost | 68.3% | 0.768 |

> NBA game prediction above 65% is considered strong — the sport has inherent randomness from injuries, foul trouble, and hot shooting nights that no model can fully capture.

---

## 🏆 2026 NBA Finals Prediction

**New York Knicks vs San Antonio Spurs**

| Game | Location | Model Prediction | Actual |
|---|---|---|---|
| Game 1 | San Antonio | NYK 52.9% | ✅ NYK won |
| Game 2 | San Antonio | NYK 52.9% | ✅ NYK won |
| Game 3 | New York | NYK 80.0% | TBD |
| Game 4 | New York | NYK 78.8% | TBD |
| Game 5 (if needed) | San Antonio | NYK 52.8% | TBD |

**→ Model predicts NYK Knicks to win the 2026 NBA Finals 5-0 in simulated games 🗽**

The model correctly predicted both Games 1 and 2 as NYK wins. NYK leads the series 2-0.

---

## 🔧 Features Engineered

**Dataset:** 5,242 rows × 79 columns → ~75 model features after engineering

### Original features
- Rolling averages over last 10 games (PTS, FG%, 3P%, REB, AST, TOV, +/-)
- Home court advantage flag
- Rest days between games
- Head-to-head win rate (last 3 meetings)
- Season progress indicator
- Playoff flag

### Improvement 1 — Series context
The biggest improvement. For every playoff game, the model now knows the series state *before* that game is played:
- `SERIES_GAME_NUM` — which game in the series (1–7)
- `HOME/AWAY_SERIES_WINS` — wins accumulated so far in the series
- `DIFF_SERIES_WINS` — series lead or deficit
- `HOME/AWAY_ELIMINATION` — facing elimination (0/1)
- `SERIES_TIED` — series is even (0/1)

Regular season games have these set to 0.

### Improvement 2 — Momentum
Short-window rolling stats that capture whether a team is hot *right now*, not just over the last 10 games:
- `ROLL_WIN_RATE_L3` — last 3 games win rate
- `ROLL_WIN_RATE_L5` — last 5 games win rate
- `MOMENTUM` — difference between L3 and L10 win rate (trending up or fading)
- `STREAK` — current consecutive win/loss streak
- `ROLL_PTS_L3` — last 3 games scoring average

### Improvement 3 — Margin quality
Playoff series are won in close games. Teams that consistently win close games all season are better Finals picks:
- `ROLL_CLOSE_WIN_RATE` — win rate in games decided by ≤5 points
- `ROLL_WIN_MARGIN` — average winning margin in victories
- `ROLL_LOSS_MARGIN` — average losing margin in defeats
- `ROLL_POINT_DIFF` — rolling net point differential (net rating proxy)

All features are also computed as `DIFF_` (HOME minus AWAY) columns — giving the model direct team-vs-team comparisons. 22 difference features total.

---



---

## 🚀 How to Run

### Option 1 — Google Colab (recommended, zero setup)
1. Open `nba_step1_data_collection.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run all cells → saves `nba_games_raw.csv`
3. Run `nba_step2_improved.py` → saves `nba_games_features.csv` (5,242 rows, 79 cols)
4. Run `nba_step4_single_cell.py` → trains 3 models, saves `.pkl` files
5. Run `nba_step5_6_eval_and_predict.py` → cross-validation + live predictions

### Option 2 — Local (Python 3.9+)
```bash
git clone https://github.com/YOUR_USERNAME/nba-finals-predictor
cd nba-finals-predictor
pip install nba_api scikit-learn xgboost pandas numpy matplotlib seaborn joblib
```
Run files in order: Step 1 → 2 → 4 → 5+6.

---

## 🎯 Making Predictions

After running all steps, use `predict_game()`:

```python
# Predict Game 3 — NYK at home, leads series 2-0
predict_game('NYK', 'SAS', home_rest=2, away_rest=2, is_playoff=1)
```

```
🏆 PLAYOFF PREDICTION
  NYK (home, 2d rest)  vs  SAS (away, 2d rest)
  ────────────────────────────────────────────────
  XGBoost    : NYK 82.8%  |  SAS 17.2%
  Rnd Forest : NYK 75.9%  |  SAS 24.1%
  Log Reg    : NYK 81.4%  |  SAS 18.6%
  ────────────────────────────────────────────────
  🎯 ENSEMBLE : NYK 80.0%  |  SAS 20.0%
  → Predicted winner: NYK  (confidence: 80.0%)
```

**All 30 NBA team codes:**
```
ATL  BOS  BKN  CHA  CHI  CLE  DAL  DEN  DET  GSW
HOU  IND  LAC  LAL  MEM  MIA  MIL  MIN  NOP  NYK
OKC  ORL  PHI  PHX  POR  SAC  SAS  TOR  UTA  WAS
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `nba_api` | Official NBA stats data collection |
| `pandas` / `numpy` | Data manipulation and feature engineering |
| `scikit-learn` | Logistic Regression, Random Forest, preprocessing, evaluation |
| `xgboost` | Gradient boosted tree model |
| `matplotlib` / `seaborn` | Visualisations |
| `joblib` | Model serialisation |
| Google Colab | Development environment |

---

## 📈 How It Works

```
NBA API → Raw game logs → Feature engineering → Time-based split → 3 models → Ensemble
```

1. **Data** — 5,242 games from official NBA stats API (Oct 2022 → Jun 2026), regular season + playoffs
2. **Features** — Rolling stats, momentum windows, series context, margin quality — 75 features per game
3. **Split** — Time-based cutoff at April 1 2026 (train on past, test on 2026 playoffs). No random shuffling to prevent data leakage
4. **Models** — Three independent classifiers whose win probabilities are averaged for the final ensemble prediction
5. **Evaluation** — Time series cross-validation (5 folds) ensures accuracy estimates reflect real-world performance

---

## ⚠️ Limitations

- Does not ingest real-time injury reports or player availability
- Predictions assume average lineup for both teams
- Small playoff test set (82 games) means accuracy estimates have some variance
- Series momentum is captured via `DIFF_SERIES_WINS` but in-game fatigue within a series is not modelled
- Model is retrained on historical data — live retraining after each game would improve accuracy further

---

## 🔮 Future Improvements

- [ ] Streamlit web app so anyone can get predictions without running code
- [ ] Live data pipeline that auto-updates rolling stats after each game
- [ ] Player-level features (star player availability, usage rate, injuries)
- [ ] Betting line comparison — model probability vs Vegas implied probability
- [ ] Full series outcome prediction (probability of winning in 4, 5, 6, or 7 games)
- [ ] Automatic retraining after each playoff game

---



*Data sourced from the NBA Stats API via `nba_api`. For educational purposes only.*
