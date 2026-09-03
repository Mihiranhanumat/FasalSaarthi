"""
FASALSAARTHI - Yield Prediction Module (Maharashtra, Correct Design)

Dataset insight: crop_yield.csv is STATE-LEVEL data.
  - area      = total Maharashtra area under that crop (lakh ha)
  - production = total Maharashtra production (lakh tonnes)
  - yield      = production/area = kg/ha (already the correct per-ha rate)

Correct model: crop + year + season → yield (kg/ha) for Maharashtra
User inputs that feed the model: crop, year, season, district
User inputs used only for farm-level calculation: farm_area_ha
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_yield.csv')

# ── Maharashtra districts with productivity multipliers ───────────────────────
DISTRICTS = [
    'Pune', 'Nashik', 'Aurangabad', 'Nagpur', 'Solapur', 'Kolhapur',
    'Satara', 'Sangli', 'Jalgaon', 'Ahmednagar', 'Latur', 'Osmanabad',
    'Nanded', 'Amravati', 'Akola', 'Washim', 'Buldhana', 'Yavatmal',
    'Ratnagiri', 'Sindhudurg', 'Raigad', 'Thane', 'Palghar',
]

DISTRICT_FACTOR = {
    'Kolhapur': 1.18, 'Satara': 1.12, 'Nashik': 1.10, 'Ratnagiri': 1.08,
    'Sangli': 1.07,   'Pune': 1.05,   'Sindhudurg': 1.05, 'Raigad': 1.04,
    'Jalgaon': 1.02,  'Amravati': 1.01,'Nagpur': 1.00, 'Ahmednagar': 0.98,
    'Thane': 0.97,    'Buldhana': 0.96,'Akola': 0.95, 'Nanded': 0.93,
    'Aurangabad': 0.93,'Yavatmal': 0.92,'Palghar': 0.91,'Washim': 0.90,
    'Latur': 0.89,    'Osmanabad': 0.87,'Solapur': 0.85,
}

SEASONS = ['Kharif', 'Rabi', 'Whole Year', 'Summer', 'Autumn']

# Clean display names for UI
CROP_DISPLAY = {
    'Arhar/Tur':           'Arhar / Tur Dal',
    'Bajra':               'Bajra (Pearl Millet)',
    'Castor seed':         'Castor Seed',
    'Cotton(lint)':        'Cotton',
    'Gram':                'Gram (Chickpea)',
    'Groundnut':           'Groundnut',
    'Jowar':               'Jowar (Sorghum)',
    'Linseed':             'Linseed',
    'Maize':               'Maize',
    'Moong(Green Gram)':   'Moong (Green Gram)',
    'Niger seed':          'Niger Seed',
    'Other  Rabi pulses':  'Other Rabi Pulses',
    'Other Cereals':       'Other Cereals',
    'Other Kharif pulses': 'Other Kharif Pulses',
    'Ragi':                'Ragi (Finger Millet)',
    'Rapeseed &Mustard':   'Rapeseed & Mustard',
    'Rice':                'Rice',
    'Safflower':           'Safflower',
    'Sesamum':             'Sesamum (Til)',
    'Small millets':       'Small Millets',
    'Soyabean':            'Soybean',
    'Sugarcane':           'Sugarcane',
    'Sunflower':           'Sunflower',
    'Tobacco':             'Tobacco',
    'Urad':                'Urad (Black Gram)',
    'Wheat':               'Wheat',
    'other oilseeds':      'Other Oilseeds',
}
CROP_REVERSE = {v: k for k, v in CROP_DISPLAY.items()}

# Season defaults per crop (for auto-suggest)
CROP_SEASON = {
    'Rice': 'Kharif', 'Bajra (Pearl Millet)': 'Kharif', 'Jowar (Sorghum)': 'Rabi',
    'Cotton': 'Kharif', 'Soybean': 'Kharif', 'Arhar / Tur Dal': 'Kharif',
    'Wheat': 'Rabi', 'Gram (Chickpea)': 'Rabi', 'Groundnut': 'Kharif',
    'Sugarcane': 'Whole Year', 'Onion': 'Rabi', 'Maize': 'Kharif',
    'Sunflower': 'Rabi', 'Safflower': 'Rabi', 'Moong (Green Gram)': 'Kharif',
    'Urad (Black Gram)': 'Kharif', 'Ragi (Finger Millet)': 'Kharif',
}


def _load_maharashtra():
    """Load and clean Maharashtra-only state-level yield data."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df['crop']   = df['crop'].str.strip()
    df['state']  = df['state'].str.strip()
    df['season'] = df['season'].str.strip()

    # Maharashtra only
    mh = df[df['state'] == 'Maharashtra'].copy()

    # yield column is already tonnes/ha → convert to kg/ha
    mh['yield_kg_ha'] = mh['yield'] * 1000

    # Remove zero/null yields
    mh = mh[mh['yield_kg_ha'] > 0].dropna(subset=['yield_kg_ha'])

    # Per-crop outlier removal: clip to 5th–95th percentile
    rows = []
    for crop, grp in mh.groupby('crop'):
        lo = grp['yield_kg_ha'].quantile(0.05)
        hi = grp['yield_kg_ha'].quantile(0.95)
        clean = grp[(grp['yield_kg_ha'] >= lo) & (grp['yield_kg_ha'] <= hi)]
        if len(clean) >= 5:   # keep only crops with enough data
            rows.append(clean)

    return pd.concat(rows, ignore_index=True)


class YieldPredictor:
    """
    Predicts yield (kg/ha) for Maharashtra crops.
    Features: crop + year + season  → target: yield_kg_ha
    District factor applied as post-prediction multiplier.
    """

    FEATURES    = ['year', 'crop_enc', 'season_enc']
    FEAT_LABELS = ['Year', 'Crop Type', 'Season']

    def __init__(self):
        self.model        = None
        self.crop_enc     = LabelEncoder()
        self.season_enc   = LabelEncoder()
        self.is_trained   = False
        self.r2           = 0.0
        self.mae          = 0.0
        self._df          = None
        self._mh_stats    = {}      # per-crop stats from real MH data
        self._display_crops = []

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self):
        df = _load_maharashtra()
        self._df = df.copy()

        # Per-crop historical stats
        for crop, grp in df.groupby('crop'):
            y = grp['yield_kg_ha']
            self._mh_stats[crop] = {
                'mean': float(y.mean()),
                'std':  max(float(y.std()), float(y.mean()) * 0.05),
                'min':  float(y.min()),
                'max':  float(y.max()),
                'q25':  float(y.quantile(0.25)),
                'q75':  float(y.quantile(0.75)),
                'q10':  float(y.quantile(0.10)),
                'q90':  float(y.quantile(0.90)),
                'rows': int(len(y)),
                'trend': float(np.polyfit(grp['year'], y, 1)[0]),  # kg/ha per year
            }

        self._display_crops = sorted(
            [CROP_DISPLAY.get(c, c) for c in self._mh_stats.keys()]
        )

        df['crop_enc']   = self.crop_enc.fit_transform(df['crop'])
        df['season_enc'] = self.season_enc.fit_transform(df['season'])

        X = df[self.FEATURES]
        y = df['yield_kg_ha']

        self.model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,
            random_state=42,
        )
        self.model.fit(X, y)

        y_pred  = self.model.predict(X)
        self.r2  = r2_score(y, y_pred)
        self.mae = mean_absolute_error(y, y_pred)

        # ── Per-crop accuracy score (for per-crop confidence) ────────────────
        # For each crop, compute MAPE (mean absolute percentage error) on training data
        # This is stored in _mh_stats so confidence is crop-specific
        for crop, grp in df.groupby('crop'):
            if crop not in self._mh_stats:
                continue
            idx    = grp.index
            preds  = self.model.predict(grp[self.FEATURES])
            trues  = grp['yield_kg_ha'].values
            mape   = float(np.mean(np.abs((preds - trues) / (trues + 1e-10))))
            self._mh_stats[crop]['mape'] = mape          # 0 = perfect, 1 = 100% error
            # accuracy_score: 1 - mape, clipped to [0.4, 0.98]
            self._mh_stats[crop]['accuracy'] = float(np.clip(1.0 - mape, 0.40, 0.98))

        self.is_trained = True
        return round(self.r2, 4), round(self.mae, 2)

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, year: int, crop_display: str, district: str,
                farm_area_ha: float = 1.0, season: str = None) -> dict:

        if not self.is_trained:
            self.train()

        # Map display name → raw
        crop = CROP_REVERSE.get(crop_display, crop_display)

        # Auto-season if not provided
        if season is None:
            season = CROP_SEASON.get(crop_display, 'Kharif')

        # District productivity factor
        dist_factor = DISTRICT_FACTOR.get(district, 1.0)

        def _enc(enc, val, fallback=0):
            return int(enc.transform([val])[0]) if val in enc.classes_ else fallback

        crop_e   = _enc(self.crop_enc,   crop,   0)
        season_e = _enc(self.season_enc, season, 0)

        row = pd.DataFrame([{
            'year':       year,
            'crop_enc':   crop_e,
            'season_enc': season_e,
        }])

        # Base prediction (Maharashtra state average for this crop/year/season)
        raw_pred   = float(self.model.predict(row[self.FEATURES])[0])
        # Apply district factor
        prediction = max(raw_pred * dist_factor, 1.0)

        # ── Confidence calculation (crop-specific) ──────────────────────────
        stats = self._mh_stats.get(crop, {})
        if stats:
            mh_mean = stats['mean']
            mh_std  = stats['std']

            # 1) Per-crop model accuracy (MAPE-based, trained in fit step)
            model_score = stats.get('accuracy', 0.70)

            # 2) How far is prediction from historical range (z-score)
            z = abs(prediction - mh_mean) / (mh_std + 1e-10)
            range_score = max(0.0, 1.0 - min(z, 2.5) / 2.5)

            # 3) Data sufficiency (more MH rows = more reliable)
            data_score = min(1.0, stats['rows'] / 18.0)

            confidence = (model_score * 0.55 + range_score * 0.30 + data_score * 0.15) * 100
            confidence = float(np.clip(confidence, 50.0, 93.0))
        else:
            confidence = 60.0
            mh_mean = prediction

        # Feature importance
        imp = dict(zip(self.FEAT_LABELS, self.model.feature_importances_))

        # Total farm production
        total_production_tonnes = (prediction * farm_area_ha) / 1000.0

        return {
            'predicted_yield':        round(prediction),
            'confidence':             round(confidence, 1),
            'total_production_t':     round(total_production_tonnes, 2),
            'farm_area_ha':           farm_area_ha,
            'feature_importance':     imp,
            'model_r2':               round(self.r2, 3),
            'model_mae':              round(self.mae),
            'typical_min':            round(stats.get('q10', mh_mean * 0.7)),
            'typical_max':            round(stats.get('q90', mh_mean * 1.3)),
            'typical_avg':            round(stats.get('mean', mh_mean)),
            'typical_std':            round(stats.get('std',  mh_mean * 0.15)),
            'mh_data_rows':           stats.get('rows', 0),
            'trend_kg_per_year':      round(stats.get('trend', 0), 1),
            'crop':                   crop_display,
            'crop_raw':               crop,
            'district':               district,
            'dist_factor':            dist_factor,
            'season':                 season,
        }

    # ── UI helpers ────────────────────────────────────────────────────────────
    def get_crops(self):          return self._display_crops
    def get_districts(self):      return DISTRICTS
    def get_seasons(self):        return SEASONS
    def get_crop_default_season(self, crop_display):
        return CROP_SEASON.get(crop_display, 'Kharif')

    def get_historical_trend(self, crop_display: str) -> pd.DataFrame:
        if self._df is None:
            return pd.DataFrame()
        crop = CROP_REVERSE.get(crop_display, crop_display)
        sub  = self._df[self._df['crop'] == crop]
        if sub.empty:
            return pd.DataFrame()
        return (sub.groupby('year')['yield_kg_ha']
                .mean().reset_index()
                .rename(columns={'year':'Year','yield_kg_ha':'Yield_Kg_Ha'}))

    def get_crop_comparison(self, crop_display_list: list) -> pd.DataFrame:
        rows = []
        for cd in crop_display_list:
            c    = CROP_REVERSE.get(cd, cd)
            stat = self._mh_stats.get(c)
            if stat:
                rows.append({'Crop': cd,
                             'Avg (kg/ha)': round(stat['mean']),
                             'Min': round(stat['q10']),
                             'Max': round(stat['q90'])})
        return pd.DataFrame(rows)

    def get_dataset_info(self) -> dict:
        if self._df is None:
            return {}
        return {
            'total_rows':  len(self._df),
            'crops':       len(self._mh_stats),
            'year_range':  f"{int(self._df['year'].min())}–{int(self._df['year'].max())}",
            'state':       'Maharashtra Only',
        }
