---
title: NutriCore
emoji: 🥗
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8080
---

# NutriCore AI — Python ML Nutrition Engine

## Setup & Run

```bash
pip install -r requirements.txt
python app.py
```

Then open: http://localhost:8080

## How It Works

1. `ml_model.py` generates 4,000 training samples from physiologically accurate
   nutrition formulas (Mifflin-St Jeor BMR, ACSM activity multipliers, ISSN protein
   guidelines) with realistic noise added.

2. 9 separate GradientBoostingRegressor models are trained — one per output target:
   - target_cals, protein_g, fat_g, carb_g, water_l, tdee
   - omega3_priority, vitd_priority, creatine_priority

3. On form submit, the frontend POSTs to `/api/analyze` → Flask calls `predict()` →
   returns JSON → full UI renders with ML predictions.

## ML Details
- Model: GradientBoostingRegressor (n_estimators=200, lr=0.08, max_depth=4)
- Features: age, weight, height, gender, activity, goal, diet, health focus, BMI, BMR
- MAE on hold-out: ~30 kcal for calories, ~3g for protein, ~0.25 for priority scores