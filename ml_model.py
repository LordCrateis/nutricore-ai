"""
NutriCore AI — Python ML Model
Uses scikit-learn GradientBoostingRegressor trained on physiologically accurate
synthetic data. Returns macro ranges (min/target/max), calorie ranges, and a
ranked supplement list with per-supplement rationale.

Protein multipliers aligned with current evidence (ISSN 2017, Stokes et al. 2018):
  - Cut:      1.8–2.2 g/kg  (high protein preserves LBM on deficit)
  - Recomp:   1.6–2.0 g/kg  (moderate — not 2.6, which is bodybuilder-aggressive)
  - Bulk:     1.6–1.8 g/kg  (surplus does the heavy lifting; more protein ≠ more muscle)
  - Maintain: 1.4–1.8 g/kg  (general health range)
"""

import threading
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. SYNTHETIC TRAINING DATA GENERATION
# ─────────────────────────────────────────────

def generate_training_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    ages     = rng.integers(18, 65, n)
    weights  = rng.uniform(45, 130, n)    # kg
    heights  = rng.uniform(150, 200, n)   # cm
    genders  = rng.integers(0, 2, n)      # 0=female 1=male
    activity = rng.choice([1.2, 1.375, 1.55, 1.725, 1.9], n)
    goal     = rng.integers(0, 4, n)      # 0=cut 1=recomp 2=bulk 3=maintain
    diet     = rng.integers(0, 4, n)      # 0=omni 1=veg 2=vegan 3=keto
    health   = rng.integers(0, 4, n)      # 0=general 1=skin 2=performance 3=hormones

    # BMR — Mifflin-St Jeor
    bmr = np.where(
        genders == 1,
        10*weights + 6.25*heights - 5*ages + 5,
        10*weights + 6.25*heights - 5*ages - 161
    )
    tdee = bmr * activity

    # ── Calorie targets ──────────────────────────────────────────────────────
    # Cut:      300–700 kcal deficit (20% capped); larger people get larger deficits
    # Recomp:   ~5% deficit — enough to nudge fat loss without killing muscle protein synthesis
    # Bulk:     200–500 kcal surplus (10–12%); lean bulk keeps fat gain minimal
    # Maintain: at TDEE
    goal_delta = np.select(
        [goal==0, goal==1, goal==2, goal==3],
        [
            -np.clip(tdee * 0.20, 300, 700),
            -tdee * 0.05,
             np.clip(tdee * 0.10, 200, 500),
             0,
        ]
    )
    target_cals = tdee + goal_delta + rng.normal(0, 25, n)

    # Range: ±10% around target (gives realistic flexibility)
    cals_min = target_cals * 0.90
    cals_max = target_cals * 1.10

    # ── Protein ──────────────────────────────────────────────────────────────
    # Evidence-based midpoint multipliers (g/kg bodyweight):
    #   Cut 2.0 | Recomp 1.8 | Bulk 1.7 | Maintain 1.6
    # Range ±0.2 g/kg reflects legitimate individual variation (training age,
    # LBM ratio, age-related anabolic resistance).
    protein_mid = np.select(
        [goal==0, goal==1, goal==2, goal==3],
        [2.0, 1.8, 1.7, 1.6]
    )
    # Upward adjustments
    protein_mid += np.where(activity >= 1.725, 0.15, 0)   # high-frequency athletes
    protein_mid += np.where(health == 2, 0.15, 0)          # performance focus
    protein_mid += np.where(ages > 50, 0.1, 0)             # age-related anabolic resistance

    protein_range = 0.2  # g/kg — symmetric range half-width
    protein_g     = weights * protein_mid + rng.normal(0, 2, n)
    protein_g_min = weights * (protein_mid - protein_range)
    protein_g_max = weights * (protein_mid + protein_range)

    # ── Fats ─────────────────────────────────────────────────────────────────
    # Keto: 65–70% of calories from fat
    # Standard: 25–30% — enough for hormonal health, fat-soluble vitamins
    # Skin health: bump to 30% (EFAs matter)
    fat_pct_mid = np.where(diet == 3, 0.67, np.where(health == 1, 0.30, 0.27))
    fat_g       = (target_cals * fat_pct_mid) / 9 + rng.normal(0, 3, n)
    fat_g_min   = (cals_min * (fat_pct_mid - 0.03)) / 9
    fat_g_max   = (cals_max * (fat_pct_mid + 0.03)) / 9

    # ── Carbohydrates ────────────────────────────────────────────────────────
    # Residual after protein + fat fill the calorie budget
    protein_cals = protein_g * 4
    fat_cals     = fat_g * 9
    carb_cals    = np.maximum(target_cals - protein_cals - fat_cals, 0)
    carb_g       = carb_cals / 4 + rng.normal(0, 4, n)
    # Keto hard-cap: ≤50g net carbs
    carb_g       = np.where(diet == 3, np.clip(carb_g, 20, 50), carb_g)
    carb_g_min   = np.maximum(carb_g * 0.88, 0)
    carb_g_max   = carb_g * 1.12

    # ── Water ────────────────────────────────────────────────────────────────
    # 33 ml/kg baseline; +500 ml for moderate-to-vigorous activity
    water_l     = weights * 0.033 + np.where(activity >= 1.55, 0.5, 0) + rng.normal(0, 0.1, n)
    water_l_min = water_l - 0.3
    water_l_max = water_l + 0.5

    # ── Supplement priority scores (0–10) ────────────────────────────────────
    # Each score is built from a physiologically motivated base + context modifiers.

    # Creatine — benefits recomp, bulk, performance; less value during pure cut
    creatine = (
        1.5
        + np.where(goal == 0, 1.0, 0)       # cut: modest benefit (strength preservation)
        + np.where(goal == 1, 4.5, 0)       # recomp: high
        + np.where(goal == 2, 5.0, 0)       # bulk: highest
        + np.where(goal == 3, 3.0, 0)       # maintain: moderate
        + np.where(health == 2, 1.5, 0)     # performance focus
        + np.where(activity >= 1.55, 0.8, 0)
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Omega-3 — universally useful; critical for vegans & keto (missing EPA/DHA)
    omega3 = (
        4.0
        + np.where(diet >= 2, 2.5, 0)       # vegan: no DHA/EPA from diet
        + np.where(diet == 3, 1.5, 0)       # keto: high fat diet needs anti-inflammatory balance
        + np.where(health == 3, 2.0, 0)     # hormones: DHA supports testosterone synthesis
        + np.where(health == 1, 1.5, 0)     # skin: EPA reduces inflammation
        + np.where(goal == 2, 0.5, 0)       # bulk: anabolic signalling
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Vitamin D — baseline deficiency is extremely common; amplified by age & female gender
    vitd = (
        5.0
        + np.where(ages > 40, 1.5, 0)
        + np.where(ages > 55, 0.8, 0)       # compounded for older adults
        + np.where(genders == 0, 1.0, 0)    # females: bone density risk
        + np.where(health == 3, 0.8, 0)     # hormones: VD3 is a pro-hormone
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Magnesium — sleep, cortisol, muscle function; keto increases urinary loss
    magnesium = (
        4.5
        + np.where(diet == 3, 2.0, 0)       # keto: electrolyte depletion
        + np.where(health == 3, 1.5, 0)     # hormones: magnesium → testosterone
        + np.where(activity >= 1.725, 1.0, 0)
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # B12 — non-negotiable for vegans; useful for veg too
    b12 = (
        2.0
        + np.where(diet == 2, 7.5, 0)       # vegan: cannot get sufficient B12 from food
        + np.where(diet == 1, 3.5, 0)       # vegetarian: marginal intake
        + np.where(ages > 50, 2.0, 0)       # absorption declines with age
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Iron — females at risk; vegans absorb non-heme iron poorly
    iron = (
        2.0
        + np.where(genders == 0, 3.5, 0)    # females: menstrual loss
        + np.where(diet >= 1, 2.0, 0)       # plant-based: non-heme iron only
        + np.where(goal == 2, 0.5, 0)       # bulk: erythropoiesis demand
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Ashwagandha — cortisol/testosterone; hormones & high-stress athletes
    ashwagandha = (
        1.0
        + np.where(health == 3, 5.5, 0)     # hormones: strongest effect
        + np.where(health == 2, 2.5, 0)     # performance: cortisol management
        + np.where(activity >= 1.725, 1.5, 0)
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Zinc — performance, testosterone, immune; vegans at risk
    zinc = (
        3.0
        + np.where(health == 2, 2.5, 0)
        + np.where(health == 3, 2.0, 0)
        + np.where(diet >= 1, 1.5, 0)       # plant-based: phytates reduce absorption
        + np.where(genders == 1, 1.0, 0)    # males: higher testosterone-related demand
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Collagen + Vit C — skin focus; also joint support for athletes
    collagen = (
        1.0
        + np.where(health == 1, 7.0, 0)     # skin: primary indicator
        + np.where(health == 2, 2.0, 0)     # performance: joint/tendon integrity
        + np.where(ages > 35, 1.5, 0)       # collagen synthesis declines after ~30
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    # Electrolytes — keto mandatory; high-activity general
    electrolytes = (
        1.5
        + np.where(diet == 3, 5.5, 0)       # keto: Na/K/Mg all depleted rapidly
        + np.where(activity >= 1.725, 2.0, 0)
        + rng.normal(0, 0.25, n)
    ).clip(0, 10)

    df = pd.DataFrame({
        'age': ages, 'weight': weights, 'height': heights,
        'gender': genders, 'activity': activity,
        'goal': goal, 'diet': diet, 'health': health,
        'bmi': weights / (heights/100)**2,
        'bmr': bmr,
        # ── primary targets ──
        'tdee': tdee,
        'target_cals': target_cals,
        'cals_min': cals_min,
        'cals_max': cals_max,
        'protein_g': protein_g,
        'protein_g_min': protein_g_min,
        'protein_g_max': protein_g_max,
        'fat_g': fat_g,
        'fat_g_min': fat_g_min,
        'fat_g_max': fat_g_max,
        'carb_g': carb_g,
        'carb_g_min': carb_g_min,
        'carb_g_max': carb_g_max,
        'water_l': water_l,
        'water_l_min': water_l_min,
        'water_l_max': water_l_max,
        # ── supplement priorities ──
        'supp_creatine': creatine,
        'supp_omega3': omega3,
        'supp_vitd': vitd,
        'supp_magnesium': magnesium,
        'supp_b12': b12,
        'supp_iron': iron,
        'supp_ashwagandha': ashwagandha,
        'supp_zinc': zinc,
        'supp_collagen': collagen,
        'supp_electrolytes': electrolytes,
    })
    return df


# ─────────────────────────────────────────────
# 2. TRAIN
# ─────────────────────────────────────────────

FEATURE_COLS = ['age', 'weight', 'height', 'gender', 'activity',
                'goal', 'diet', 'health', 'bmi', 'bmr']

TARGET_COLS = [
    'tdee', 'target_cals', 'cals_min', 'cals_max',
    'protein_g', 'protein_g_min', 'protein_g_max',
    'fat_g', 'fat_g_min', 'fat_g_max',
    'carb_g', 'carb_g_min', 'carb_g_max',
    'water_l', 'water_l_min', 'water_l_max',
    'supp_creatine', 'supp_omega3', 'supp_vitd', 'supp_magnesium',
    'supp_b12', 'supp_iron', 'supp_ashwagandha', 'supp_zinc',
    'supp_collagen', 'supp_electrolytes',
]

_models: dict = {}
_scaler = StandardScaler()
_trained = False
_train_lock = threading.Lock()


def train():
    global _models, _scaler, _trained

    with _train_lock:
        if _trained:
            return

        print("🔬 Generating training data...")
        df = generate_training_data(5000)

        X = df[FEATURE_COLS].values
        X_scaled = _scaler.fit_transform(X)

        # Single split reused for all targets
        X_tr, X_te, idx_tr, idx_te = train_test_split(
            X_scaled, np.arange(len(X_scaled)), test_size=0.15, random_state=42
        )

        print("🧠 Training ML models...")
        new_models = {}
        for target in TARGET_COLS:
            y = df[target].values

            # Eval on hold-out for honest MAE
            eval_m = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.08,
                max_depth=4, subsample=0.85, random_state=42,
            )
            eval_m.fit(X_tr, y[idx_tr])
            mae = mean_absolute_error(y[idx_te], eval_m.predict(X_te))
            print(f"  ✅ {target:25s} MAE = {mae:.2f}")

            # Final model on full data
            final_m = GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.08,
                max_depth=4, subsample=0.85, random_state=42,
            )
            final_m.fit(X_scaled, y)
            new_models[target] = final_m

        _models = new_models
        _trained = True
        print("✅ All models trained.\n")


# ─────────────────────────────────────────────
# 3. SUPPLEMENT CATALOGUE
#    Static metadata; priority score comes from ML.
# ─────────────────────────────────────────────

SUPPLEMENT_META = {
    'supp_creatine': {
        'icon': '💊', 'name': 'Creatine Monohydrate',
        'dose': '5g / day (no loading needed)',
        'desc': 'Most researched ergogenic aid. Increases phosphocreatine stores → more ATP → '
                'more reps, more strength, better body composition. Works regardless of goal.',
    },
    'supp_omega3': {
        'icon': '🐟', 'name': 'Omega-3 Fish Oil',
        'dose': '2–3g EPA+DHA / day with a meal',
        'desc': 'Reduces systemic inflammation, supports brain, cardiovascular, hormonal, '
                'and skin health. Vegans and keto dieters especially lack dietary EPA/DHA.',
    },
    'supp_vitd': {
        'icon': '☀️', 'name': 'Vitamin D3 + K2',
        'dose': '2000–4000 IU D3 + 100mcg K2 / day',
        'desc': 'D3 functions as a pro-hormone affecting immunity, testosterone, mood, and '
                'bone density. K2 (MK-7) directs calcium to bones rather than arteries.',
    },
    'supp_magnesium': {
        'icon': '🪨', 'name': 'Magnesium Glycinate',
        'dose': '300–400mg elemental / night',
        'desc': 'Co-factor in 300+ enzymatic reactions. Improves sleep quality, lowers '
                'cortisol, reduces muscle cramps. Glycinate form has high bioavailability '
                'and minimal laxative effect. Keto diets deplete magnesium rapidly.',
    },
    'supp_b12': {
        'icon': '🌱', 'name': 'Vitamin B12',
        'dose': '1000mcg methylcobalamin / day',
        'desc': 'Non-negotiable for plant-based diets — impossible to meet needs from '
                'plant foods alone. Methylcobalamin (active form) is superior to cyanocobalamin. '
                'Absorption also declines significantly after age 50.',
    },
    'supp_iron': {
        'icon': '🩸', 'name': 'Iron (Bisglycinate)',
        'dose': '18–25mg / day (test first)',
        'desc': 'Bisglycinate form is gentler on the gut than ferrous sulfate. Critical for '
                'menstruating females and plant-based eaters (non-heme iron absorption is ~10% '
                'vs ~25% for heme iron). Get ferritin tested before supplementing.',
    },
    'supp_ashwagandha': {
        'icon': '🌿', 'name': 'Ashwagandha KSM-66',
        'dose': '600mg / day (morning or split)',
        'desc': 'KSM-66 extract shown in RCTs to reduce cortisol by 25–30%, improve '
                'testosterone in stressed males, and enhance VO2max. Particularly valuable '
                'for high-training-load athletes and hormonal health goals.',
    },
    'supp_zinc': {
        'icon': '⚡', 'name': 'Zinc (Picolinate)',
        'dose': '15–25mg / day with food',
        'desc': 'Essential for testosterone synthesis, immune function, and protein metabolism. '
                'Plant-based diets provide phytate-bound zinc with poor bioavailability. '
                'Picolinate form has the best absorption. Avoid mega-dosing — competes with copper.',
    },
    'supp_collagen': {
        'icon': '✨', 'name': 'Marine Collagen + Vitamin C',
        'dose': '10g collagen + 500mg Vit C / morning',
        'desc': 'Hydrolysed marine collagen (type I/III) + Vitamin C is the clinically studied '
                'stack for skin elasticity, joint cartilage, and tendon repair. Vit C is '
                'required for collagen cross-linking — do not take collagen without it.',
    },
    'supp_electrolytes': {
        'icon': '⚗️', 'name': 'Electrolyte Complex (Na/K/Mg)',
        'dose': '1 serving intra/post workout or with first meal',
        'desc': 'Keto diets cause rapid urinary loss of sodium, potassium, and magnesium '
                '(reduced insulin → less renal reabsorption). Electrolyte depletion causes '
                'the "keto flu". High-volume athletes also sweat significant sodium.',
    },
}


# ─────────────────────────────────────────────
# 4. PREDICT
# ─────────────────────────────────────────────

GOAL_MAP   = {'cut': 0, 'recomp': 1, 'bulk': 2, 'maintain': 3}
DIET_MAP   = {'omnivore': 0, 'vegetarian': 1, 'vegan': 2, 'keto': 3}
HEALTH_MAP = {'general': 0, 'skin': 1, 'performance': 2, 'hormones': 3}
GENDER_MAP = {'male': 1, 'female': 0}


def _range(lo, mid, hi, lo_min=None):
    """Return a tidy {min, target, max} dict with optional floor on min."""
    lo_v = round(max(lo_min, lo) if lo_min is not None else lo)
    return {'min': lo_v, 'target': round(mid), 'max': round(hi)}


def predict(age, weight, height, gender, activity, goal, diet, health,
            deficiencies: list[str] | None = None):
    for val, mapping, name in [
        (gender, GENDER_MAP, 'gender'),
        (goal,   GOAL_MAP,   'goal'),
        (diet,   DIET_MAP,   'diet'),
        (health, HEALTH_MAP, 'health'),
    ]:
        if val not in mapping:
            raise ValueError(f"{name} must be one of {list(mapping)}, got {val!r}")

    if not _trained:
        train()

    bmi = weight / (height / 100) ** 2
    bmr = (10*weight + 6.25*height - 5*age + 5  if gender == 'male'
           else 10*weight + 6.25*height - 5*age - 161)

    X = np.array([[
        age, weight, height,
        GENDER_MAP[gender], activity,
        GOAL_MAP[goal], DIET_MAP[diet], HEALTH_MAP[health],
        bmi, bmr,
    ]])
    X_scaled = _scaler.transform(X)

    models = _models
    raw = {k: float(models[k].predict(X_scaled)[0]) for k in TARGET_COLS}

    # ── Calories ─────────────────────────────────────────────────────────────
    cal_min    = max(1200, raw['cals_min'])
    cal_target = max(1200, raw['target_cals'])
    cal_max    = max(cal_target, raw['cals_max'])
    tdee       = round(raw['tdee'])

    calories = _range(cal_min, cal_target, cal_max, lo_min=1200)
    deficit  = round(cal_target - tdee)

    # ── Protein ──────────────────────────────────────────────────────────────
    protein = _range(
        max(50, raw['protein_g_min']),
        max(50, raw['protein_g']),
        max(50, raw['protein_g_max']),
        lo_min=50
    )
    # Sanity: target should not exceed 2.5 g/kg (well above any evidence-based ceiling)
    protein['target'] = min(protein['target'], round(weight * 2.5))
    protein['max']    = min(protein['max'],    round(weight * 2.5))

    # ── Fats ─────────────────────────────────────────────────────────────────
    fat = _range(max(20, raw['fat_g_min']), max(20, raw['fat_g']), raw['fat_g_max'], lo_min=20)

    # ── Carbs ────────────────────────────────────────────────────────────────
    carb_lo  = max(0, raw['carb_g_min'])
    carb_mid = max(0, raw['carb_g'])
    if diet == 'keto':
        carb_lo, carb_mid = min(carb_lo, 25), min(carb_mid, 50)
    carb = _range(carb_lo, carb_mid, max(carb_mid, raw['carb_g_max']))

    # ── Water ─────────────────────────────────────────────────────────────────
    water = {
        'min':    round(max(1.5, raw['water_l_min']), 1),
        'target': round(max(1.5, raw['water_l']),     1),
        'max':    round(max(1.5, raw['water_l_max']), 1),
    }

    # ── Supplements ──────────────────────────────────────────────────────────
    supp_scores = {k: round(min(10, max(0, raw[k])), 1) for k in SUPPLEMENT_META}

    # Boost supplements that match explicitly stated deficiencies
    deficiencies = [d.lower().strip() for d in (deficiencies or [])]
    boosts = {
        'vitamin d': 'supp_vitd', 'vit d': 'supp_vitd', 'vitamin d3': 'supp_vitd',
        'iron': 'supp_iron',
        'b12': 'supp_b12', 'vitamin b12': 'supp_b12',
        'magnesium': 'supp_magnesium', 'mag': 'supp_magnesium',
        'zinc': 'supp_zinc',
        'omega 3': 'supp_omega3', 'omega-3': 'supp_omega3', 'omega3': 'supp_omega3',
    }
    for deficiency in deficiencies:
        key = boosts.get(deficiency)
        if key:
            supp_scores[key] = min(10.0, supp_scores[key] + 2.0)

    # Build ranked supplement list
    supplements = []
    for key, meta in SUPPLEMENT_META.items():
        supplements.append({
            'key':      key,
            'icon':     meta['icon'],
            'name':     meta['name'],
            'dose':     meta['dose'],
            'desc':     meta['desc'],
            'priority': supp_scores[key],
        })
    supplements.sort(key=lambda s: s['priority'], reverse=True)

    # ── Meal distribution ────────────────────────────────────────────────────
    # Distribute target protein and carbs across meals based on goal & timing.
    meals = _meal_distribution(
        cal_target, protein['target'], carb['target'], fat['target'],
        goal, activity
    )

    return {
        'bmi':         round(bmi, 1),
        'bmr':         round(bmr),
        'tdee':        tdee,
        'deficit':     deficit,
        'calories':    calories,
        'protein':     protein,
        'fat':         fat,
        'carb':        carb,
        'water':       water,
        'supplements': supplements,
        'meals':       meals,
        # Legacy flat keys kept for backward compat with existing frontend sections
        'target_cals': calories['target'],
        'protein_g':   protein['target'],
        'fat_g':       fat['target'],
        'carb_g':      carb['target'],
        'water_l':     water['target'],
    }


def _meal_distribution(cal_target, protein_g, carb_g, fat_g, goal, activity):
    """
    Return a list of meal dicts with estimated macros per meal.
    Logic: protein spread evenly (MPS requires ~0.4g/kg per meal, ~4 meals);
    carbs front-loaded around training; fat fills remaining calories.
    """
    is_cut     = goal == 'cut'
    is_athlete = activity >= 1.725

    if is_cut:
        schedule = [
            {'time': '7:00 AM',  'name': 'Protein Breakfast',
             'note': 'High protein, low carb. Sets satiety and MPS for the day.',
             'protein_pct': 0.28, 'carb_pct': 0.15, 'fat_pct': 0.25},
            {'time': '12:30 PM', 'name': 'Lunch — Carb Anchor',
             'note': 'Biggest carb window. Rice / roti + dal + lean protein.',
             'protein_pct': 0.25, 'carb_pct': 0.40, 'fat_pct': 0.25},
            {'time': '4:00 PM',  'name': 'Pre-Workout',
             'note': 'Small carb + protein hit 45–60 min before training.',
             'protein_pct': 0.15, 'carb_pct': 0.25, 'fat_pct': 0.10},
            {'time': '7:00 PM',  'name': 'Post-Workout',
             'note': 'Fast protein + moderate carbs. Priority MPS window.',
             'protein_pct': 0.22, 'carb_pct': 0.18, 'fat_pct': 0.20},
            {'time': '9:30 PM',  'name': 'Light Dinner',
             'note': 'High protein, very low carb. Paneer / eggs / curd + salad.',
             'protein_pct': 0.10, 'carb_pct': 0.02, 'fat_pct': 0.20},
        ]
    elif is_athlete:
        schedule = [
            {'time': '7:00 AM',  'name': 'Breakfast',
             'note': 'Protein + complex carbs. Oats + eggs or whey + fruit.',
             'protein_pct': 0.22, 'carb_pct': 0.20, 'fat_pct': 0.20},
            {'time': '10:30 AM', 'name': 'Mid-Morning',
             'note': 'Sustains energy between sessions. Nuts or curd.',
             'protein_pct': 0.12, 'carb_pct': 0.10, 'fat_pct': 0.20},
            {'time': '1:00 PM',  'name': 'Lunch',
             'note': 'Heaviest meal — carbs, protein, fats all present.',
             'protein_pct': 0.25, 'carb_pct': 0.30, 'fat_pct': 0.25},
            {'time': '4:30 PM',  'name': 'Pre-Workout',
             'note': 'Carbs + protein 45 min before. Fuels performance.',
             'protein_pct': 0.18, 'carb_pct': 0.25, 'fat_pct': 0.10},
            {'time': '7:30 PM',  'name': 'Post-Workout / Dinner',
             'note': 'Protein-first to replenish MPS; carbs restore glycogen.',
             'protein_pct': 0.23, 'carb_pct': 0.15, 'fat_pct': 0.25},
        ]
    else:
        schedule = [
            {'time': '8:00 AM',  'name': 'Breakfast',
             'note': 'Start with protein. Eggs, curd, or a whey smoothie.',
             'protein_pct': 0.25, 'carb_pct': 0.25, 'fat_pct': 0.25},
            {'time': '1:00 PM',  'name': 'Lunch',
             'note': 'Biggest meal of the day. Balanced macros.',
             'protein_pct': 0.35, 'carb_pct': 0.45, 'fat_pct': 0.35},
            {'time': '4:30 PM',  'name': 'Afternoon Snack',
             'note': 'Protein + fat combo keeps insulin stable.',
             'protein_pct': 0.15, 'carb_pct': 0.10, 'fat_pct': 0.15},
            {'time': '7:30 PM',  'name': 'Dinner',
             'note': 'Protein-led, moderate carbs, healthy fats.',
             'protein_pct': 0.25, 'carb_pct': 0.20, 'fat_pct': 0.25},
        ]

    meals = []
    for m in schedule:
        meals.append({
            'time':    m['time'],
            'name':    m['name'],
            'note':    m['note'],
            'protein': round(protein_g * m['protein_pct']),
            'carbs':   round(carb_g    * m['carb_pct']),
            'fat':     round(fat_g     * m['fat_pct']),
            'cals':    round(
                protein_g * m['protein_pct'] * 4 +
                carb_g    * m['carb_pct']    * 4 +
                fat_g     * m['fat_pct']     * 9
            ),
        })
    return meals
