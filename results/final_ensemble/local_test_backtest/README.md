# Local historical test backtest

Scored after the ensemble and original test forecast files were frozen. No
models, weights or forecasts were modified. Forecast hashes are in `audit.json`.

The organiser's `actuals_test.csv` is not present. This evaluation reconstructs
realised prices, direction and spot from the local package's input series.
The same reconstruction matches every supplied validation actual (including
missing values), and covers all 26,920 test forecast rows from 673 origins.
The official scorer 0.1.1 was used. These are local scores, not organiser-certified
scores; the organiser may use a different locked actuals snapshot.

| Metric | Final ensemble | Previous LightGBM |
|---|---:|---:|
| MAE EUR/MWh | 26.702820 | 26.690916 |
| Mean P10/P90 pinball | 6.830206 | 7.086564 |
| Coverage | 80.7207% | 89.4019% |
| Spike Brier | 0.056664 | 0.056534 |
| Direction Brier | 0.393744 | 0.394655 |

The ensemble improves quantile loss and direction Brier; median-price MAE and
spike Brier are slightly worse. Its coverage is closer to the nominal 80% target.
Validation and test have different price regimes: do not interpret lower test
MAE than validation MAE as a model improvement.

Earlier reports and video narration saying test accuracy is unknown predate
this local evaluation. The accurate current statement is: the historical test
backtest has been scored locally; the official organiser score is pending.

Reproduce: `.venv/bin/python src/pkg/score_local_test.py`.
Reconstructed label CSV stays in ignored `work_private/final_ensemble/`.
