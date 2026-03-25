"""
FASALSAARTHI - Crop Recommendation Module
Uses REAL Crop_recommendation_konkan_maharashtra.csv dataset (2,200 records).
Algorithm: Random Forest Classifier → recommends best crop for soil/climate conditions.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_recommendation.csv')

# ── Crop info database ─────────────────────────────────────────────────────────
CROP_INFO = {
    'rice':        {'emoji':'🌾','season':'Kharif','water':'High','hindi':'धान','marathi':'भात'},
    'maize':       {'emoji':'🌽','season':'Kharif/Rabi','water':'Medium','hindi':'मक्का','marathi':'मका'},
    'chickpea':    {'emoji':'🫘','season':'Rabi','water':'Low','hindi':'चना','marathi':'हरभरा'},
    'kidneybeans': {'emoji':'🫘','season':'Kharif','water':'Medium','hindi':'राजमा','marathi':'राजमा'},
    'pigeonpeas':  {'emoji':'🌿','season':'Kharif','water':'Low','hindi':'अरहर','marathi':'तूर'},
    'mothbeans':   {'emoji':'🫘','season':'Kharif','water':'Low','hindi':'मोठ','marathi':'मठ'},
    'mungbean':    {'emoji':'🫘','season':'Kharif','water':'Low','hindi':'मूंग','marathi':'मूग'},
    'blackgram':   {'emoji':'🫘','season':'Kharif','water':'Low','hindi':'उड़द','marathi':'उडीद'},
    'lentil':      {'emoji':'🫘','season':'Rabi','water':'Low','hindi':'मसूर','marathi':'मसूर'},
    'pomegranate': {'emoji':'🍎','season':'Perennial','water':'Low','hindi':'अनार','marathi':'डाळिंब'},
    'banana':      {'emoji':'🍌','season':'Perennial','water':'High','hindi':'केला','marathi':'केळ'},
    'mango':       {'emoji':'🥭','season':'Perennial','water':'Medium','hindi':'आम','marathi':'आंबा'},
    'grapes':      {'emoji':'🍇','season':'Perennial','water':'Medium','hindi':'अंगूर','marathi':'द्राक्ष'},
    'watermelon':  {'emoji':'🍉','season':'Summer','water':'Medium','hindi':'तरबूज','marathi':'कलिंगड'},
    'muskmelon':   {'emoji':'🍈','season':'Summer','water':'Medium','hindi':'खरबूज','marathi':'खरबूज'},
    'apple':       {'emoji':'🍏','season':'Rabi','water':'Medium','hindi':'सेब','marathi':'सफरचंद'},
    'orange':      {'emoji':'🍊','season':'Perennial','water':'Medium','hindi':'संतरा','marathi':'संत्रा'},
    'papaya':      {'emoji':'🧡','season':'Perennial','water':'Medium','hindi':'पपीता','marathi':'पपई'},
    'coconut':     {'emoji':'🥥','season':'Perennial','water':'High','hindi':'नारियल','marathi':'नारळ'},
    'cotton':      {'emoji':'☁️','season':'Kharif','water':'Medium','hindi':'कपास','marathi':'कापूस'},
    'jute':        {'emoji':'🌿','season':'Kharif','water':'High','hindi':'जूट','marathi':'ताग'},
    'coffee':      {'emoji':'☕','season':'Perennial','water':'Medium','hindi':'कॉफी','marathi':'कॉफी'},
}

SOIL_HEALTH_TIPS = {
    'N': {
        'low':  'Apply Urea (46% N) @ 50–100 kg/ha or Ammonium Sulphate @ 100–200 kg/ha.',
        'high': 'Reduce nitrogen inputs. Excess N causes lodging and disease susceptibility.',
        'ok':   'Nitrogen levels are adequate. Maintain with split application of Urea.',
    },
    'P': {
        'low':  'Apply DAP (46% P₂O₅) @ 100–150 kg/ha or SSP (16% P) @ 300–400 kg/ha.',
        'high': 'Avoid additional P fertilizer. High P locks out zinc and iron.',
        'ok':   'Phosphorus is adequate. Apply DAP as basal at recommended rates.',
    },
    'K': {
        'low':  'Apply MOP (60% K₂O) @ 50–100 kg/ha or SOP for chloride-sensitive crops.',
        'high': 'Skip K fertilizer this season. High K can reduce calcium uptake.',
        'ok':   'Potassium is adequate. Continue with standard MOP dose at sowing.',
    },
}


class CropRecommender:
    """Random Forest classifier trained on Konkan/Maharashtra soil-climate data."""

    def __init__(self):
        self.model      = None
        self.label_enc  = LabelEncoder()
        self.is_trained = False
        self.accuracy   = 0.0
        self.df         = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        self.feature_display = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)',
                                'Temperature (°C)', 'Humidity (%)', 'Soil pH', 'Rainfall (mm)']

    def train(self) -> float:
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna()
        self.df = df

        y_enc = self.label_enc.fit_transform(df['label'])
        X     = df[self.feature_names].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12,
            random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        self.accuracy   = accuracy_score(y_test, self.model.predict(X_test))
        self.is_trained = True
        return round(self.accuracy * 100, 2)

    def recommend(self, N: float, P: float, K: float, temperature: float,
                  humidity: float, ph: float, rainfall: float) -> dict:
        if not self.is_trained:
            self.train()

        X = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        probs      = self.model.predict_proba(X)[0]
        top_idx    = np.argsort(probs)[::-1][:5]
        top_crops  = [(self.label_enc.classes_[i], round(probs[i] * 100, 1)) for i in top_idx]

        best_crop  = top_crops[0][0]
        best_conf  = top_crops[0][1]
        info       = CROP_INFO.get(best_crop, {})

        # Feature importance
        imp = dict(zip(self.feature_display, self.model.feature_importances_))

        # Soil health tips
        tips = {}
        for nut, val, name in [('N', N, 'Nitrogen'), ('P', P, 'Phosphorus'), ('K', K, 'Potassium')]:
            status = 'low' if val < 25 else 'high' if val > 100 else 'ok'
            tips[name] = {'status': status, 'tip': SOIL_HEALTH_TIPS[nut][status]}

        return {
            'best_crop':        best_crop,
            'confidence':       best_conf,
            'top_5':            top_crops,
            'season':           info.get('season', 'N/A'),
            'water_need':       info.get('water', 'Medium'),
            'emoji':            info.get('emoji', '🌾'),
            'hindi':            info.get('hindi', best_crop),
            'marathi':          info.get('marathi', best_crop),
            'soil_tips':        tips,
            'feature_importance': imp,
            'model_accuracy':   round(self.accuracy * 100, 1),
            'input_summary':    {
                'N': N, 'P': P, 'K': K,
                'Temperature': temperature,
                'Humidity': humidity,
                'pH': ph,
                'Rainfall': rainfall,
            },
        }

    def get_dataset_stats(self) -> dict:
        if self.df is None:
            return {}
        return {
            'rows':   len(self.df),
            'crops':  self.df['label'].nunique(),
            'region': 'Konkan / Maharashtra',
        }

    def get_all_crops(self) -> list:
        if self.is_trained:
            return sorted(self.label_enc.classes_)
        return sorted(CROP_INFO.keys())

    def get_crop_ranges(self) -> pd.DataFrame:
        """Return mean N,P,K,temp,humidity,pH,rainfall per crop — for reference table."""
        if self.df is None:
            return pd.DataFrame()
        return self.df.groupby('label')[self.feature_names].mean().round(1).reset_index()
