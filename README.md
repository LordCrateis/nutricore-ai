<div align="center">

# NutriCore AI

**A Flask-based nutrition engine powered by scikit-learn regressors**

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-server-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

NutriCore AI is a Flask-based nutrition engine that converts a person's profile, training activity, goals, dietary pattern, health focus, and reported deficiencies into a structured nutrition plan. It returns calorie and macronutrient targets, hydration guidance, meal timing, and a ranked supplement-priority catalogue.

The project uses scikit-learn regression models trained on physiologically motivated synthetic data.

> **Not a medical tool.** This is an educational machine-learning application and planning prototype — not a medical diagnostic system or a substitute for advice from a qualified healthcare professional.

## Table of Contents

- [Features](#features)
- [Technology Stack](#technology-stack)
- [How the System Works](#how-the-system-works)
- [Model Design](#model-design)
- [Prediction Outputs](#prediction-outputs)
- [Local Setup](#local-setup)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Model Caching](#model-caching)
- [Docker](#docker)
- [Security and Safety Considerations](#security-and-safety-considerations)
- [Limitations](#limitations)
- [Recommended Improvements](#recommended-improvements)
- [License](#license)
- [References](#references)

---

## Features

| Capability | Description |
| --- | --- |
| **Nutrition analysis form** | Collects age, weight, height, gender, activity level, goal, diet, health focus, and optional deficiencies. |
| **ML-generated targets** | Predicts TDEE, calorie targets, macro ranges, water intake, and supplement-priority scores. |
| **Personalized meal distribution** | Produces meal timing and estimated calories, protein, carbohydrates, and fat for each meal. |
| **Supplement ranking** | Ranks a built-in supplement catalogue by context-sensitive priority and includes dose and rationale metadata. |
| **Physiology-inspired features** | Derives BMI and BMR using the Mifflin–St Jeor equation and estimates TDEE from activity multipliers. |
| **Model caching** | Saves trained regressors and the scaler to `model_cache.joblib` so subsequent restarts can load the models instead of retraining. |
| **REST API** | Exposes `POST /api/analyze` for frontend or programmatic use. |
| **Docker support** | Includes a Python 3.11 slim image that serves the Flask app on port `8080`. |

## Technology Stack

Python, Flask, scikit-learn, GradientBoostingRegressor, pandas, NumPy, joblib, HTML, JavaScript, Bootstrap, Chart.js, Docker

The dependency versions are declared in [`requirements.txt`](https://github.com/LordCrateis/nutricore-ai/blob/main/requirements.txt). [2]

## How the System Works

The application is composed of a Flask server, a model module, and a server-rendered single-page frontend:

```text
Browser form
    │
    │ POST /api/analyze
    ▼
Flask application (app.py)
    │
    ├── Validate required fields
    ├── Normalize optional deficiencies
    └── Call predict(...)
            │
            ▼
      Nutrition model layer (ml_model.py)
            │
            ├── Encode categorical inputs
            ├── Calculate BMI and BMR
            ├── Scale model features
            ├── Run target-specific regressors
            ├── Apply safety floors and output ranges
            ├── Rank supplements
            └── Build meal distribution
                    │
                    ▼
             JSON response rendered by the UI
```

The main application is implemented in [`app.py`](https://github.com/LordCrateis/nutricore-ai/blob/main/app.py), while data generation, training, caching, inference, supplement metadata, and meal distribution are implemented in [`ml_model.py`](https://github.com/LordCrateis/nutricore-ai/blob/main/ml_model.py). [1] [3]

## Model Design

### Synthetic training data

NutriCore generates 5,000 synthetic profiles using a fixed random seed of `42`. The generated feature ranges are:

| Feature | Generated range or values |
| --- | --- |
| Age | Integer values from 18 through 64. |
| Weight | Uniformly sampled from 45–130 kg. |
| Height | Uniformly sampled from 150–200 cm. |
| Gender | Encoded as female or male. |
| Activity | One of `1.2`, `1.375`, `1.55`, `1.725`, or `1.9`. |
| Goal | Cut, recomp, bulk, or maintain. |
| Diet | Omnivore, vegetarian, vegan, or keto. |
| Health focus | General, skin, performance, or hormones. |

The target values are generated from domain-inspired formulas with added random noise. Calories use TDEE and goal-specific adjustments, protein responds to goal, activity, performance focus, and age, fats respond to diet and skin focus, carbohydrates fill the remaining calorie budget, and water responds to body weight and activity.

The supplement priority scores are also generated from contextual rules. Examples include higher B12 priority for vegan diets, higher electrolyte priority for keto diets, and higher creatine priority for performance-oriented recomp or bulk goals.

Because the training data is generated from formulas rather than learned from measured nutrition outcomes, the model is best understood as a structured prediction and personalization demonstration. It does not establish clinical efficacy or individualized medical requirements.

### Derived features

The model trains on ten input features:

```text
age, weight, height, gender, activity,
goal, diet, health, bmi, bmr
```

Categorical values are converted to numeric codes using the mappings defined in `ml_model.py`:

| Input | Encoded values |
| --- | --- |
| `gender` | `female = 0`, `male = 1` |
| `goal` | `cut = 0`, `recomp = 1`, `bulk = 2`, `maintain = 3` |
| `diet` | `omnivore = 0`, `vegetarian = 1`, `vegan = 2`, `keto = 3` |
| `health` | `general = 0`, `skin = 1`, `performance = 2`, `hormones = 3` |

BMI is calculated as `weight / height²` after converting height from centimeters to meters. BMR uses the gender-specific Mifflin–St Jeor formula.

### Regressors and validation

The current implementation trains one `GradientBoostingRegressor` per target. There are 26 target columns covering TDEE, calories, macro and water ranges, and ten supplement scores.

Each regressor uses the following configuration:

```python
GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.85,
    random_state=42,
)
```

The generated data is split into 85% training and 15% hold-out data with `random_state=42`. A `StandardScaler` is fitted on the training features. The code reports mean absolute error for each target on the hold-out partition, then fits a final estimator on all generated data before saving the model cache.

> **No committed evaluation report.** The repository does not currently include a committed evaluation report, benchmark table, automated tests, or experiment-tracking metadata. The printed MAE values should be treated as diagnostics for the synthetic data-generating process rather than evidence of real-world nutrition accuracy.

## Prediction Outputs

The `predict()` function returns the following structure:

| Output | Description |
| --- | --- |
| `bmi` | Body mass index rounded to one decimal place. |
| `bmr` | Basal metabolic rate rounded to the nearest calorie. |
| `tdee` | Total daily energy expenditure estimate. |
| `deficit` | Difference between target calories and TDEE. |
| `calories` | `{min, target, max}` calorie range with a 1,200 kcal minimum floor. |
| `protein` | `{min, target, max}` grams with a minimum floor and a 2.5 g/kg upper sanity limit. |
| `fat` | `{min, target, max}` grams. |
| `carb` | `{min, target, max}` grams, with a keto cap applied. |
| `water` | `{min, target, max}` liters with a minimum floor of 1.5 liters. |
| `supplements` | Ranked list with name, dose, description, and priority score from 0–10. |
| `meals` | Goal- and activity-dependent meal schedule with macro and calorie estimates. |

Legacy flat keys are also returned for compatibility with existing frontend sections: `target_cals`, `protein_g`, `fat_g`, `carb_g`, and `water_l`.

## Local Setup

### Prerequisites

Install **Python 3.11 or newer**. The included Dockerfile uses `python:3.11-slim`, and the application does not require a database, API key, or external service for local operation. [5]

### Installation

```bash
git clone https://github.com/LordCrateis/nutricore-ai.git
cd nutricore-ai

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the application

```bash
python app.py
```

Open [http://localhost:8080](http://localhost:8080) in a browser.

The Flask application binds to `0.0.0.0` on port `8080`. On the first startup, the application begins training in a background thread. While training is in progress, analysis requests return HTTP `503` with a retry message. Once training finishes, the models are saved to `model_cache.joblib` and subsequent starts can load the cache.

## API Reference

### `GET /`

Returns the HTML user interface.

### `POST /api/analyze`

Accepts a JSON nutrition profile and returns either a successful analysis or a structured error.

#### Required request fields

| Field | Type | Accepted values or meaning |
| --- | --- | --- |
| `age` | integer-compatible value | Age in years. |
| `weight` | number | Body weight in kilograms. |
| `height` | number | Height in centimeters. |
| `gender` | string | `male` or `female`. |
| `activity` | number | Activity multiplier such as `1.2`, `1.375`, `1.55`, `1.725`, or `1.9`. |
| `goal` | string | `cut`, `recomp`, `bulk`, or `maintain`. |
| `diet` | string | `omnivore`, `vegetarian`, `vegan`, or `keto`. |
| `health` | string | `general`, `skin`, `performance`, or `hormones`. |
| `deficiencies` | optional list or string | Optional deficiency labels such as `iron`, `b12`, `magnesium`, `zinc`, `vitamin d`, or `omega-3`. A comma-separated string is also accepted. |

Example request:

```bash
curl -X POST http://localhost:8080/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "weight": 72,
    "height": 178,
    "gender": "male",
    "activity": 1.55,
    "goal": "recomp",
    "diet": "omnivore",
    "health": "performance",
    "deficiencies": ["magnesium"]
  }'
```

Example response shape:

```json
{
  "success": true,
  "data": {
    "bmi": 22.7,
    "bmr": 1718,
    "tdee": 2663,
    "deficit": -133,
    "calories": {
      "min": 2277,
      "target": 2530,
      "max": 2783
    },
    "protein": {
      "min": 115,
      "target": 137,
      "max": 144
    },
    "fat": {
      "min": 70,
      "target": 76,
      "max": 84
    },
    "carb": {
      "min": 240,
      "target": 275,
      "max": 309
    },
    "water": {
      "min": 2.1,
      "target": 2.9,
      "max": 3.4
    },
    "supplements": [],
    "meals": []
  }
}
```

The numeric values in the example are illustrative response-shape values. Actual values depend on the trained cache and submitted profile.

### Error responses

| Status | Meaning |
| --- | --- |
| `400` | Required field is missing, a value cannot be converted, or a categorical value is invalid. |
| `503` | The model is still initializing in the background. Retry after training completes. |
| `500` | An unexpected exception occurred while computing the analysis. |

## Frontend

The server-rendered interface is defined in [`templates/index.html`](https://github.com/LordCrateis/nutricore-ai/blob/main/templates/index.html). It provides a form for the core profile fields and optional deficiencies, then renders:

- BMI, BMR, TDEE, and calorie deficit information.
- Calorie, protein, fat, carbohydrate, and water ranges.
- Food-source and micronutrient suggestions from frontend lookup data.
- A ranked supplement protocol.
- Meal-timing recommendations.
- Visual summaries rendered with Chart.js.

The page submits JSON to `/api/analyze` on the same Flask origin. Some presentation-layer food and micronutrient content is maintained in static frontend lookup tables, while the core calorie, macro, water, supplement-priority, and meal outputs are returned by the Python backend.

## Model Caching

The model module stores the trained state in:

```text
model_cache.joblib
```

The cache contains the dictionary of target-specific regressors and the fitted `StandardScaler`. If the cache exists, NutriCore loads it at startup; otherwise it generates synthetic data, trains all target models, prints hold-out MAE diagnostics, and writes a new cache.

To force retraining, stop the application and remove the cache:

```bash
rm -f model_cache.joblib
python app.py
```

> **Cache trust note.** Do not load a cache file from an untrusted source. Joblib uses Python object serialization and should be treated with the same care as other pickle-compatible model artifacts.

## Docker

Build and run the included image:

```bash
docker build -t nutricore-ai .
docker run --rm -p 8080:8080 nutricore-ai
```

Open [http://localhost:8080](http://localhost:8080).

The [`Dockerfile`](https://github.com/LordCrateis/nutricore-ai/blob/main/Dockerfile) uses Python 3.11 slim, installs `requirements.txt`, copies the project into `/app`, exposes port `8080`, and starts the application with `python app.py`. [5]

### Platform deployment note

The application currently hardcodes port `8080` in `app.py`. If your hosting platform supplies a dynamic `PORT` environment variable, update the entrypoint to use it, for example:

```python
port = int(os.environ.get("PORT", 8080))
app.run(host="0.0.0.0", port=port)
```

The current Dockerfile is suitable for platforms that route traffic to port `8080` or allow the container port to be configured explicitly.

## Security and Safety Considerations

NutriCore does not currently include authentication, rate limiting, CSRF protection, or server-side request-size limits. Add these controls before exposing the API publicly.

The model returns nutrition and supplement suggestions based on synthetic formulas and user-entered profile data. Outputs can be inaccurate, unsuitable for individual medical conditions, or inappropriate where medications, allergies, pregnancy, eating disorders, kidney disease, liver disease, or other clinical factors are involved. The application should clearly display a health disclaimer and direct users to a qualified clinician or registered dietitian for personalized medical guidance.

Supplement doses and recommendations should not be treated as prescriptions. Deficiency-related supplementation should generally be guided by appropriate laboratory testing and professional advice.

## Limitations

The current project has several important limitations:

1. Training data is synthetic and generated from hand-authored formulas rather than measured outcomes.
2. The model predicts nutrition targets and priority scores, not health outcomes.
3. The synthetic training distribution is limited to adults aged 18–64, weights of 45–130 kg, and heights of 150–200 cm.
4. Model quality is reported through MAE on synthetic hold-out data, so the reported errors do not measure performance on real people.
5. The application trains during startup when the cache is absent, which can increase cold-start latency.
6. The model cache is not versioned with a schema or training metadata manifest.
7. The API performs only lightweight input validation and does not enforce comprehensive physiological bounds.
8. There is no automated test suite or CI workflow in the repository.

## Recommended Improvements

For a more robust release, separate training from serving, version the model artifact, store training configuration alongside the cache, add unit tests for feature engineering and output bounds, validate all numeric and categorical inputs with a schema, add real-world evaluation data with appropriate consent and privacy controls, and make the port configurable through the environment.

The frontend and backend would also benefit from a dedicated API schema, consistent error objects, server-side request logging without sensitive data, health and readiness endpoints, and a clear distinction between educational estimates and clinical nutrition advice.

## License

The repository includes a license file. Review [`LICENSE`](https://github.com/LordCrateis/nutricore-ai/blob/main/LICENSE) for the exact terms before redistributing or deploying the project. [6]

## References

1. [NutriCore Flask application](https://github.com/LordCrateis/nutricore-ai/blob/main/app.py)
2. [NutriCore Python dependencies](https://github.com/LordCrateis/nutricore-ai/blob/main/requirements.txt)
3. [NutriCore model training and inference module](https://github.com/LordCrateis/nutricore-ai/blob/main/ml_model.py)
4. [NutriCore frontend template](https://github.com/LordCrateis/nutricore-ai/blob/main/templates/index.html)
5. [NutriCore Dockerfile](https://github.com/LordCrateis/nutricore-ai/blob/main/Dockerfile)
6. [NutriCore license](https://github.com/LordCrateis/nutricore-ai/blob/main/LICENSE)
