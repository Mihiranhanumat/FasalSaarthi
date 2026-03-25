"""
FASALSAARTHI - Fertilizer Recommendation Module
Rule-based NPK calculator with cost estimation.
"""

from dataclasses import dataclass

FERTILIZERS = {
    'Urea':               {'N':46,'P':0, 'K':0,  'price':6.5,  'hindi':'यूरिया'},
    'Ammonium Sulphate':  {'N':21,'P':0, 'K':0,  'price':12.0, 'hindi':'अमोनियम सल्फेट'},
    'DAP':                {'N':18,'P':46,'K':0,  'price':27.0, 'hindi':'डीएपी'},
    'SSP':                {'N':0, 'P':16,'K':0,  'price':8.0,  'hindi':'सिंगल सुपर फॉस्फेट'},
    'TSP':                {'N':0, 'P':46,'K':0,  'price':25.0, 'hindi':'ट्रिपल सुपर फॉस्फेट'},
    'MOP':                {'N':0, 'P':0, 'K':60, 'price':17.0, 'hindi':'पोटाश (MOP)'},
    'SOP':                {'N':0, 'P':0, 'K':50, 'price':45.0, 'hindi':'सल्फेट ऑफ पोटाश'},
    'NPK 10-26-26':       {'N':10,'P':26,'K':26, 'price':22.0, 'hindi':'NPK 10-26-26'},
    'NPK 12-32-16':       {'N':12,'P':32,'K':16, 'price':24.0, 'hindi':'NPK 12-32-16'},
}

CROP_REQUIREMENTS = {
    'Rice':       {'N':120,'P':60, 'K':60,  'notes':'Split N in 3 doses'},
    'Wheat':      {'N':120,'P':60, 'K':40,  'notes':'Apply half N at sowing'},
    'Sugarcane':  {'N':250,'P':115,'K':115, 'notes':'High K demand, apply in splits'},
    'Cotton':     {'N':100,'P':50, 'K':50,  'notes':'Apply N in 3–4 splits'},
    'Soybean':    {'N':30, 'P':60, 'K':40,  'notes':'Use Rhizobium seed treatment'},
    'Jowar':      {'N':90, 'P':40, 'K':40,  'notes':'Top dress N at knee height'},
    'Bajra':      {'N':90, 'P':45, 'K':45,  'notes':'N in 2 splits'},
    'Tur Dal':    {'N':20, 'P':50, 'K':30,  'notes':'Inoculate with Rhizobium'},
    'Onion':      {'N':100,'P':50, 'K':75,  'notes':'High K improves bulb quality'},
    'Tomato':     {'N':120,'P':60, 'K':60,  'notes':'Fertigate for best results'},
    'Potato':     {'N':180,'P':80, 'K':120, 'notes':'Split K; high K for tuber quality'},
    'Maize':      {'N':150,'P':75, 'K':40,  'notes':'Top dress 50% N at knee height'},
    'Groundnut':  {'N':25, 'P':50, 'K':50,  'notes':'Rhizobium + gypsum @ 500 kg/ha'},
    'Soyabean':   {'N':30, 'P':60, 'K':40,  'notes':'Rhizobium seed treatment'},
    'Sunflower':  {'N':90, 'P':60, 'K':60,  'notes':'Boron @ 1.5 kg/ha critical'},
    'Banana':     {'N':200,'P':60, 'K':200, 'notes':'High K for bunch quality'},
    'Grapes':     {'N':100,'P':50, 'K':100, 'notes':'Fertigate; split K in 3 parts'},
}

SOIL_FACTORS = {
    'Low':    {'N':1.25,'P':1.25,'K':1.25},
    'Medium': {'N':1.00,'P':1.00,'K':1.00},
    'High':   {'N':0.70,'P':0.50,'K':0.50},
}

SCHEDULES = {
    'Rice':      ['Basal at transplanting (P + K + 1/3 N)',
                  'First top-dress at 25 days (1/3 N)',
                  'Second top-dress at panicle initiation (1/3 N)'],
    'Wheat':     ['Basal at sowing (P + K + 1/2 N)',
                  'Top-dress at crown root initiation – 21 DAS (1/2 N)'],
    'Sugarcane': ['Basal at planting (P + 1/4 N)',
                  '30 days (1/4 N + 1/2 K)',
                  '90 days (1/4 N + 1/2 K)',
                  '150 days (1/2 N)'],
    'Cotton':    ['Basal at sowing (P + K + 1/3 N)',
                  '45 DAS (1/3 N)',
                  '90 DAS (1/3 N)'],
    'Tomato':    ['Basal (P + K + 1/3 N)',
                  'Flowering – 1/3 N',
                  'Fruit setting – 1/3 N'],
    'Potato':    ['Basal at planting (P + 1/2 K + 1/2 N)',
                  'Earthing up – remaining N + K'],
    'Maize':     ['Basal at sowing (P + K + 1/3 N)',
                  'Knee height – 1/3 N',
                  'Tasselling – 1/3 N'],
}
DEFAULT_SCHEDULE = ['Basal at sowing (P + K + 1/2 N)', 'Top-dress at 30 DAS (1/2 N)']

ORGANIC_TIPS = {
    'Rice':      'Apply 5–10 t/ha FYM before transplanting. Use Azolla as green manure.',
    'Wheat':     'Apply 10 t/ha FYM before sowing. Seed treatment with Azotobacter.',
    'Sugarcane': 'Apply 25 t/ha FYM + pressmud compost @ 5 t/ha.',
    'Soybean':   'Rhizobium seed treatment reduces N requirement by 25 kg/ha.',
    'Soyabean':  'Rhizobium seed treatment reduces N requirement by 25 kg/ha.',
    'Cotton':    'Apply 10 t/ha FYM. Use Trichoderma to improve soil health.',
    'Tomato':    'Apply 15–20 t/ha FYM + vermicompost @ 5 t/ha.',
    'Potato':    'Apply 20 t/ha FYM. Use balanced micronutrient mix.',
    'Maize':     'Apply 10 t/ha FYM before sowing. Crop residue incorporation adds N.',
    'Onion':     'Apply 15 t/ha FYM. Foliar spray of 0.5% ZnSO₄ improves bulb size.',
    'Banana':    'Apply 20 t/ha FYM + vermicompost 5 t/ha. Drip fertigation ideal.',
}
DEFAULT_ORGANIC = 'Apply 5–10 t/ha FYM before sowing to improve soil organic carbon.'


class FertilizerRecommender:
    def recommend(self, crop, area, soil_n='Medium', soil_p='Medium', soil_k='Medium') -> dict:
        if crop not in CROP_REQUIREMENTS:
            # Try case-insensitive match
            match = next((k for k in CROP_REQUIREMENTS if k.lower() == crop.lower()), None)
            if not match:
                return {'error': f'Crop "{crop}" not in database.'}
            crop = match

        base = CROP_REQUIREMENTS[crop]
        sf   = {'N': SOIL_FACTORS[soil_n]['N'],
                'P': SOIL_FACTORS[soil_p]['P'],
                'K': SOIL_FACTORS[soil_k]['K']}

        n_need = base['N'] * sf['N'] * area
        p_need = base['P'] * sf['P'] * area
        k_need = base['K'] * sf['K'] * area

        recs  = []
        total = 0.0

        # Nitrogen
        urea_kg = (n_need / FERTILIZERS['Urea']['N']) * 100
        as_kg   = (n_need / FERTILIZERS['Ammonium Sulphate']['N']) * 100
        r_n = {'nutrient':'Nitrogen (N)','soil_status':soil_n,
               'nutrient_kg_total':round(n_need,1),'nutrient_kg_per_ha':round(base['N']*sf['N'],1),
               'primary_fert':'Urea','primary_qty_kg':round(urea_kg,1),
               'primary_hindi':FERTILIZERS['Urea']['hindi'],
               'alt_fert':'Ammonium Sulphate','alt_qty_kg':round(as_kg,1),
               'application':'Split: ½ basal + ¼ at 25 days + ¼ at 50 days',
               'cost':round(urea_kg*FERTILIZERS['Urea']['price'],0),'icon':'🟦'}
        recs.append(r_n); total += r_n['cost']

        # Phosphorus
        dap_kg = (p_need / FERTILIZERS['DAP']['P']) * 100
        ssp_kg = (p_need / FERTILIZERS['SSP']['P']) * 100
        r_p = {'nutrient':'Phosphorus (P₂O₅)','soil_status':soil_p,
               'nutrient_kg_total':round(p_need,1),'nutrient_kg_per_ha':round(base['P']*sf['P'],1),
               'primary_fert':'DAP','primary_qty_kg':round(dap_kg,1),
               'primary_hindi':FERTILIZERS['DAP']['hindi'],
               'alt_fert':'SSP','alt_qty_kg':round(ssp_kg,1),
               'application':'Full dose at sowing/transplanting',
               'cost':round(dap_kg*FERTILIZERS['DAP']['price'],0),'icon':'🟨'}
        recs.append(r_p); total += r_p['cost']

        # Potassium
        mop_kg = (k_need / FERTILIZERS['MOP']['K']) * 100
        sop_kg = (k_need / FERTILIZERS['SOP']['K']) * 100
        r_k = {'nutrient':'Potassium (K₂O)','soil_status':soil_k,
               'nutrient_kg_total':round(k_need,1),'nutrient_kg_per_ha':round(base['K']*sf['K'],1),
               'primary_fert':'MOP','primary_qty_kg':round(mop_kg,1),
               'primary_hindi':FERTILIZERS['MOP']['hindi'],
               'alt_fert':'SOP (chloride-sensitive crops)','alt_qty_kg':round(sop_kg,1),
               'application':'½ basal + ½ at 30 days',
               'cost':round(mop_kg*FERTILIZERS['MOP']['price'],0),'icon':'🟩'}
        recs.append(r_k); total += r_k['cost']

        return {
            'crop':             crop,
            'area':             area,
            'recommendations':  recs,
            'total_cost':       round(total, 0),
            'crop_notes':       base['notes'],
            'schedule':         SCHEDULES.get(crop, DEFAULT_SCHEDULE),
            'organic_tip':      ORGANIC_TIPS.get(crop, DEFAULT_ORGANIC),
            'nutrient_summary': {'N':round(n_need,1),'P':round(p_need,1),'K':round(k_need,1)},
        }

    def get_crops(self):
        return list(CROP_REQUIREMENTS.keys())
