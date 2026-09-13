# EV Purchase Prediction

Predicts the probability that a person will buy an electric vehicle, using a LightGBM model trained on a heavily imbalanced dataset (~83% "No" / ~17% "Yes").

## Dataset

- **Rows:** 668,665 (train), no missing values, no duplicates
- **Target:** `Will_Buy_EV` (Yes/No) — mapped to 1/0
- **Features:**
  - Numeric: `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`
  - Categorical: `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`

## Pipeline

1. **EDA** — checked class balance, nulls/duplicates, feature distributions (density plots for numeric, count plots for categorical).
2. **Split** — stratified 3-way split on the raw data:
   - 80% train
   - 10% val (hyperparameter tuning + early stopping)
   - 10% calibration (fitting the probability calibrator)
3. **Preprocessing** — `ColumnTransformer` with `StandardScaler` on numeric features and `OneHotEncoder` (`handle_unknown='ignore'`) on categorical features. Fit once on train only; val/calib/test are transformed with the same fitted preprocessor (never re-fit).
4. **Imbalance handling** — `scale_pos_weight` (ratio of negative to positive class counts in train), tuned within Optuna's search rather than fixed.
5. **Model** — LightGBM (`objective="binary"`), chosen for training speed at this data size and native imbalance support.
6. **Hyperparameter search** — Optuna, 100 trials, **optimizing ROC-AUC** on the validation set. AUC (a ranking metric) was chosen over log loss for the search objective because log loss is distorted by `scale_pos_weight` reweighting — optimizing raw log loss during search was found to push `scale_pos_weight` toward 1 (i.e. away from correcting for the imbalance at all), so AUC keeps the search honest about ranking quality independent of reweighting.
7. **Calibration** — the final tuned model is refit as an `LGBMClassifier`, wrapped in `FrozenEstimator`, and calibrated with `CalibratedClassifierCV(method="isotonic")` on the held-out calibration split. This corrects the probability outputs, since `scale_pos_weight` skews raw probabilities away from true likelihoods.
8. **Evaluation** — log loss and ROC-AUC on the validation set, using calibrated probabilities.
9. **Submission** — calibrated probabilities generated for the test set and written to `submission_2.csv`.

## Results

| Metric | Value |
|---|---|
| Best Optuna trial (val ROC-AUC) | 0.9420 |
| Final model — val log loss (calibrated) | 0.2267 |
| Final model — val ROC-AUC (calibrated) | 0.9418 |

Best hyperparameters found:
```
learning_rate: 0.0745
num_leaves: 15
min_child_samples: 168
feature_fraction: 0.525
bagging_fraction: 0.947
bagging_freq: 9
lambda_l1: 1.23e-06
lambda_l2: 1.03e-08
scale_pos_weight: 1.37
```

## Notes / Limitations

- `scale_pos_weight` converged to ~1.37 rather than the raw class ratio (~5), since AUC only rewards correct *ranking* and doesn't penalize under-reweighting the way recall would. If recall on the positive class is a priority, it's worth checking that metric specifically and potentially fixing `scale_pos_weight` closer to the true ratio rather than leaving it to the search.
- The validation set is used for both hyperparameter selection and final reported metrics; there's no fully independent labeled test set held out beyond that, since the actual `test.csv` is unlabeled (used only for submission).
- Output is a predicted probability per row, not a thresholded class label — no decision threshold has been tuned or applied.

## Files

- `electric_vehicle_purchasing.ipynb` — full pipeline (EDA → preprocessing → training → tuning → calibration → submission)
- `submission_2.csv` — final predicted probabilities for the test set
