"""
FASALSAARTHI - Smart Crop Recommender
Uses Crop_recommendation_konkan_maharashtra.csv with N,P,K,temp,humidity,pH,rainfall
to recommend best crops for given soil & weather conditions.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'crop_recommendation.csv')

# Crop info cards — label → details
CROP_INFO = {
    'rice':        {'emoji':'🌾','season':'Kharif','water':'High','hindi':'चावल','marathi':'भात'},
    'maize':       {'emoji':'🌽','season':'Kharif/Rabi','water':'Medium','hindi':'मक्का','marathi':'मका'},
    'chickpea':    {'emoji':'🟡','season':'Rabi','water':'Low','hindi':'चना','marathi':'हरभरा'},
    'kidneybeans': {'emoji':'🫘','season':'Kharif','water':'Medium','hindi':'राजमा','marathi':'राजमा'},
    'pigeonpeas':  {'emoji':'🌿','season':'Kharif','water':'Low','hindi':'अरहर/तूर','marathi':'तूर'},
    'mothbeans':   {'emoji':'🌱','season':'Kharif','water':'Low','hindi':'मोठ','marathi':'मोठ'},
    'mungbean':    {'emoji':'💚','season':'Kharif','water':'Low','hindi':'मूंग','marathi':'मूग'},
    'blackgram':   {'emoji':'⚫','season':'Kharif','water':'Low','hindi':'उड़द','marathi':'उडीद'},
    'lentil':      {'emoji':'🟤','season':'Rabi','water':'Low','hindi':'मसूर','marathi':'मसूर'},
    'pomegranate': {'emoji':'🍎','season':'Perennial','water':'Low','hindi':'अनार','marathi':'डाळिंब'},
    'banana':      {'emoji':'🍌','season':'Perennial','water':'High','hindi':'केला','marathi':'केळी'},
    'mango':       {'emoji':'🥭','season':'Summer','water':'Medium','hindi':'आम','marathi':'आंबा'},
    'grapes':      {'emoji':'🍇','season':'Rabi','water':'Medium','hindi':'अंगूर','marathi':'द्राक्षे'},
    'watermelon':  {'emoji':'🍉','season':'Summer','water':'Medium','hindi':'तरबूज','marathi':'कलिंगड'},
    'muskmelon':   {'emoji':'🍈','season':'Summer','water':'Medium','hindi':'खरबूजा','marathi':'खरबूज'},
    'apple':       {'emoji':'🍏','season':'Rabi','water':'Medium','hindi':'सेब','marathi':'सफरचंद'},
    'orange':      {'emoji':'🍊','season':'Perennial','water':'Medium','hindi':'संतरा','marathi':'संत्रा'},
    'papaya':      {'emoji':'🧡','season':'Perennial','water':'Medium','hindi':'पपीता','marathi':'पपई'},
    'coconut':     {'emoji':'🥥','season':'Perennial','water':'High','hindi':'नारियल','marathi':'नारळ'},
    'cotton':      {'emoji':'🤍','season':'Kharif','water':'Medium','hindi':'कपास','marathi':'कापूस'},
    'jute':        {'emoji':'🟫','season':'Kharif','water':'High','hindi':'जूट','marathi':'ताग'},
    'coffee':      {'emoji':'☕','season':'Perennial','water':'High','hindi':'कॉफी','marathi':'कॉफी'},
}

IDEAL_CONDITIONS = {
    'rice':        {'N':(80,120),'P':(40,60),'K':(40,60),'pH':(5.5,7.0),'temp':(20,35),'rain':(150,300)},
    'wheat':       {'N':(100,140),'P':(50,70),'K':(30,50),'pH':(6.0,7.5),'temp':(10,25),'rain':(40,100)},
    'maize':       {'N':(100,130),'P':(60,80),'K':(50,70),'pH':(5.8,7.0),'temp':(18,27),'rain':(50,100)},
    'cotton':      {'N':(100,140),'P':(50,70),'K':(50,70),'pH':(6.0,7.5),'temp':(21,30),'rain':(60,110)},
    'sugarcane':   {'N':(150,250),'P':(60,100),'K':(80,120),'pH':(6.0,8.0),'temp':(20,35),'rain':(75,200)},
    'banana':      {'N':(100,150),'P':(80,100),'K':(180,220),'pH':(5.5,7.0),'temp':(20,35),'rain':(75,100)},
    'chickpea':    {'N':(20,40),'P':(50,70),'K':(40,60),'pH':(6.0,8.0),'temp':(10,25),'rain':(65,95)},
    'pigeonpeas':  {'N':(20,40),'P':(50,70),'K':(30,50),'pH':(6.0,7.5),'temp':(25,35),'rain':(65,100)},
}


class CropRecommender:
    def __init__(self):
        self.model     = None
        self.label_enc = LabelEncoder()
        self.is_trained= False
        self.accuracy  = 0.0
        self._crops    = []

    def train(self):
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip().lower() for c in df.columns]

        self._crops = sorted(df['label'].unique().tolist())
        X = df[['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = self.label_enc.fit_transform(df['label'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        self.accuracy  = accuracy_score(y_test, self.model.predict(X_test))
        self.is_trained = True
        return round(self.accuracy * 100, 1)

    def recommend(self, N: float, P: float, K: float, temperature: float,
                  humidity: float, ph: float, rainfall: float) -> dict:
        if not self.is_trained:
            self.train()

        X = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                         columns=['n','p','k','temperature','humidity','ph','rainfall'])
        proba = self.model.predict_proba(X)[0]

        # Top 5
        top_idx = np.argsort(proba)[::-1][:5]
        top5    = []
        for idx in top_idx:
            label = self.label_enc.inverse_transform([idx])[0]
            info  = CROP_INFO.get(label, {'emoji':'🌾','season':'N/A','water':'N/A',
                                          'hindi':label,'marathi':label})
            top5.append({
                'crop':     label.title(),
                'label':    label,
                'prob':     round(float(proba[idx]) * 100, 1),
                'emoji':    info['emoji'],
                'season':   info['season'],
                'water':    info['water'],
                'hindi':    info['hindi'],
                'marathi':  info['marathi'],
            })

        best    = top5[0]
        cond    = IDEAL_CONDITIONS.get(best['label'], {})

        # Input suitability feedback
        tips = []
        if cond:
            if N < cond.get('N', (0,999))[0]:   tips.append(f"⬆️ Increase Nitrogen (current {N:.0f}, ideal {cond['N'][0]}–{cond['N'][1]})")
            if P < cond.get('P', (0,999))[0]:   tips.append(f"⬆️ Increase Phosphorus (current {P:.0f})")
            if K < cond.get('K', (0,999))[0]:   tips.append(f"⬆️ Increase Potassium (current {K:.0f})")
            if ph < cond.get('pH', (0,14))[0]:  tips.append(f"⬆️ pH too low ({ph:.1f}); apply lime")
            if ph > cond.get('pH', (0,14))[1]:  tips.append(f"⬇️ pH too high ({ph:.1f}); apply gypsum")
        if not tips:
            tips.append("✅ Soil conditions are suitable for the recommended crop!")

        return {
            'top5':          top5,
            'best_crop':     best['crop'],
            'best_emoji':    best['emoji'],
            'best_prob':     best['prob'],
            'best_season':   best['season'],
            'best_hindi':    best['hindi'],
            'best_marathi':  best['marathi'],
            'model_accuracy':round(self.accuracy * 100, 1),
            'tips':          tips,
            'inputs':        {'N':N,'P':P,'K':K,'Temp':temperature,
                              'Humidity':humidity,'pH':ph,'Rainfall':rainfall},
        }

    def get_crops(self):
        return [c.title() for c in self._crops]

    def get_dataset_info(self):
        df = pd.read_csv(DATA_PATH)
        return {'rows': len(df), 'crops': df['label'].nunique(),
                'accuracy': round(self.accuracy * 100, 1)}
