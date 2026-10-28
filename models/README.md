---
license: mit
library_name: sklearn
pipeline_tag: tabular-regression        # or: tabular-classification
tags:
  - sklearn
  - tabular-regression                  # or: tabular-classification
  - built-environment
  - iaac-maai
  - ai-methods-1
metrics:
  - r2                                  # classification: accuracy, f1 ...
  - mae
---

# <king_county_house_price>: predicting <price> from <Housing Characteristics>

This model predicts residential property sale prices based on physical characteristics, construction quality, and geographic information. It is a scikit-learn pipeline combining preprocessing with a Histogram Gradient Boosting Regressor, trained on historical housing sales in King County, Washington, USA. The model is designed to support real estate professionals, buyers, and investors in estimating property values. It was built as a course assignment for AI Methods I (IAAC MaAI 2026/27).

- **GitHub repository (full pipeline):** https://github.com/iaac-maai/<2-day2-assignment>
**Team 2:** 
- Karthikeyan Saravanan
- Emmanuel Marquez
- Bars babila
- Samiha Abeda Nur

## Intended use

- **Intended users and use:** Real estate professionals, property buyers, and investors seeking preliminary residential property price estimates based on building characteristics and location.
- **Out-of-scope uses:** Properties outside King County, other building types such as commercial or industrial properties, and current market valuations without updated data. Predictions should not replace professional property appraisals or financial assessments.

## Model details

| | |
|---|---|
| **Problem type** | Regression |
| **Algorithm** | HistGradientBoostingRegressor |
| **Key hyperparameters** | Default configuration, `random_state=42` |
| **Preprocessing** | Median imputation (numeric features); most-frequent imputation and One-Hot Encoding (zipcode). No scaling. |
| **File** | `king_county_house_price.joblib`: fitted scikit-learn pipeline with model metadata |
| **Library versions** | scikit-learn 1.9.1, pandas 3.0.6, numpy 2.5.3, Python 3.12.15 |

## Input features

The model expects a `pandas.DataFrame` with exactly these columns:

| Column | Type | Unit | Description | Preprocessing |
|---|---|---|---|---|
| `bedrooms` | Integer | Count | Number of bedrooms | Median imputation |
| `bathrooms` | Float | Count | Number of bathrooms | Median imputation |
| `sqft_living` | Integer | ft² | Interior living area | Median imputation |
| `sqft_lot` | Integer | ft² | Total lot area | Median imputation |
| `floors` | Float | Count | Number of floors | Median imputation |
| `condition` | Integer | Rating 1–5 | Overall property condition | Median imputation |
| `grade` | Integer | Rating | Construction and design quality | Median imputation |
| `sqft_basement` | Integer | ft² | Basement area | Median imputation |
| `yr_built` | Integer | Year | Year of construction | Median imputation |
| `yr_renovated` | Integer | Year | Year of renovation (0 if none) | Median imputation |
| `zipcode` | Integer | ZIP code | Property postal code | Most-frequent imputation + One-Hot Encoding |
| `lat` | Float | Degrees | Latitude | Median imputation |
| `long` | Float | Degrees | Longitude | Median imputation |

## Output

The model predicts residential property prices.

The training workflow uses a base-10 logarithmic transformation of the target (log10(price)). If the saved model outputs predictions on this transformed scale, predictions must be converted back to USD:
...

## Training data

| | |
|---|---|
| **Dataset** | House Sales in King County, USA|
| **Source** | [https://www.kaggle.com/datasets/harlfoxem/housesalesprediction](Kaggle — House Sales Prediction) |
| **Licence** | CCO: Public Domain |
| **One row is** | One residential property sale, including its sale price and housing characteristics |
| **Size used** | 21,613 original observations; 16,556 training, 4,147 test. |
| **Coverage** | King County, Washington, USA, including Seattle; May 2014–May 2015; residential properties |

**Cleaning:** The original dataset was inspected and cleaned before model training. The sale date was converted from a string to a datetime format, and selected features were removed based on the team's data exploration and modeling decisions.

## Training procedure

- **Split:** 80 / 20 train/test, `random_state=42` 
- **Validation:** 5-fold cross-validation using GroupKFold, grouping observations by zip code to avoid geographic overlap between training and validation folds.
- **Models compared:**

| Model | CV R² (mean ± std) | CV MAE log10 (mean ± std) |
|---|---|---|
| Baseline (mean) | -0.0004 ± 0.0004 | 0.1675 ± 0.0017 |
| Linear Regression | 0.8512 ± 0.0032 | 0.0586 ± 0.0009 |
| Ridge Regression | 0.8511 ± 0.0031 | 0.0587 ± 0.0009 |
| KNN (k=10) | 0.8229 ± 0.0015 | 0.0633 ± 0.0007 |
| Decision Tree | 0.8119 ± 0.0051 | 0.0648 ± 0.0005 |
| Random Forest | 0.8769 ± 0.0050 | 0.0517 ± 0.0009 |
| **Gradient Boosting** | **0.8858 ± 0.0053** | **0.0501 ± 0.0012** |

- **Tuning:** A grid search evaluated 9 hyperparameter combinations using 5-fold cross-validation. The best tested configuration (learning_rate=0.05, max_leaf_nodes=31, max_iter=300) achieved an R² of 0.8675, compared with 0.8858 for the default Gradient Boosting model.


- **Why this model:** The default Histogram Gradient Boosting model achieved the highest cross-validation R² (0.8858 ± 0.0053) and the lowest MAE (0.0501 ± 0.0012 in log10 units) among the evaluated models. Since hyperparameter tuning did not improve performance, the default configuration was selected.

## Evaluation

Results on the held-out **test set** (used once):

## Evaluation

Results on the held-out **test set** (used once):

| Metric | Test | Cross-validation |
|---|---|---|
| R² | 0.8751 (USD) | 0.8858 ± 0.0053 (log10) |
| MAE (USD) | $60,200.51 | — |
| RMSE (USD) | $98,383.87 | — |
| MAE (log10) | — | 0.0501 ± 0.0012 |
| MAPE (%) | 11.59% | — |
| Approximate typical error (%) | — | 12.22% |
| Predictions within ±20% | 83.70% | — |

**Where it performs worst:** The model shows its highest mean absolute percentage error in the lowest-priced properties (13.24%), followed by the highest-priced properties (11.95%). In comparison, medium-high-priced properties achieve the lowest error (9.81%). This suggests that the model performs less consistently at the lower and upper ends of the housing market, where predictions should be interpreted with additional caution.

**Where it performs worst:** ...

## Limitations and biases

- Historical data: The dataset covers 2014–2015 transactions and may not represent current property prices or market conditions.

Geographic limitations: The model was trained exclusively on King County properties and may not generalize to other locations.

Missing information: Important factors such as renovation quality, architectural design, neighborhood amenities, and current economic conditions may not be fully captured.

Geographic bias: Property values are influenced by location and historical market patterns, which may be reflected in the model's predictions.

Prediction uncertainty: Unusual properties or homes with characteristics outside the training distribution may produce unreliable estimates.

Model interpretation: Associations between housing features and predicted prices should not be interpreted as causal relationships.

## How to use

```python
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="iaac-maai/<model-repo>", filename="model.joblib")
model = joblib.load(path)

example = pd.DataFrame([{
    "column_1": ...,
    "column_2": ...,
}])
print(model.predict(example))
```

⚠️ `.joblib` files are Python pickles and can execute code when loaded: only load models from sources you trust. Use the library versions listed above; the model may not load with other versions.

## Reproducibility

The full pipeline (data loading, cleaning, training, evaluation) is in the GitHub repository: https://github.com/iaac-maai/<2-day2-assignment>. Run `notebook/supervised_learning.ipynb` with the packages in `requirements.txt`.

## Authors and licence

Karthikeyan Saravanan, Emmanuel Marquez, Bars babila, Samiha Abeda Nur. Course assignment for **AI Methods I: Machine Learning for the Built Environment**, IAAC MaAI 2026/27.

Model released under the MIT License. The training data is subject to its own licence: ...