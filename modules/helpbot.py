"""
FASALSAARTHI - AI Helpbot Module
Intent-based chatbot for farmer assistance in English, Hindi and Marathi.
"""

import re
from datetime import datetime

# ── Intent definitions ────────────────────────────────────────────────────────
INTENTS = {
    'greeting': {
        'keywords': [
            'hello', 'hi', 'namaste', 'hey', 'good morning', 'good evening',
            'good afternoon', 'namaskar', 'vandana',
            'नमस्ते', 'हेलो', 'नमस्कार', 'प्रणाम', 'राम राम',
            'नमस्कार', 'जय हिंद',
        ],
        'response': (
            "🙏 Namaste! I am **FASALSAARTHI**, your AI farming companion.\n\n"
            "I can help you with:\n"
            "🌿 **Crop Disease Detection** – upload a leaf photo\n"
            "📊 **Yield Prediction** – know expected output\n"
            "🌱 **Fertilizer Guidance** – NPK recommendations\n"
            "💰 **Market Prices** – mandi rates\n"
            "🌧️ **Irrigation & Weather** – seasonal tips\n\n"
            "How can I help you today?"
        ),
        'action': None,
        'suggestions': ['Disease detection', 'Predict yield', 'Fertilizer advice', 'Market prices'],
    },
    'disease': {
        'keywords': [
            'disease', 'spot', 'blight', 'rust', 'yellow leaf', 'brown leaf', 'sick',
            'dying', 'wilting', 'mold', 'fungus', 'pest', 'insect', 'worm',
            'caterpillar', 'aphid', 'whitefly', 'mite', 'lesion', 'rot',
            'रोग', 'धब्बा', 'पीला', 'पीले', 'पीली', 'बीमार', 'कीड़ा', 'कीट', 'इल्ली',
            'सूखना', 'मुरझाना', 'झुलसा', 'करपा', 'फफूंद', 'कवक', 'विषाणु', 'वायरस',
            'पत्ते पीले', 'पत्ती पीली', 'फसल खराब', 'नुकसान', 'दवाई', 'दवा',
            'रोग', 'डाग', 'आजार', 'किडे', 'बुरशी', 'पिवळे', 'करपा', 'पाने पिवळी',
            'पीक खराब', 'नुकसान', 'औषध', 'फवारणी', 'कीटक',
        ],
        'response': (
            "🌿 Leaf or crop symptoms detected in your message!\n\n"
            "Please go to the **Disease Detection** module and upload a clear photo "
            "of the affected leaf. The AI will identify the disease and suggest treatment.\n\n"
            "📸 **Photo Tips:** Good lighting, clear focus, show the affected area clearly."
        ),
        'action': 'open_disease_detector',
        'suggestions': ['Upload leaf photo', 'What diseases affect rice?', 'Fungicide list'],
    },
    'yield': {
        'keywords': [
            'yield', 'production', 'harvest', 'output', 'crop quantity', 'how much',
            'predict', 'estimate', 'tonnes', 'quintal', 'kg per acre',
            'उपज', 'पैदावार', 'उत्पादन', 'कापणी', 'किती', 'अंदाज',
            'उत्पन्न', 'माल',
        ],
        'response': (
            "📊 I can predict your crop yield!\n\n"
            "Go to the **Yield Prediction** module and enter:\n"
            "• Crop name\n"
            "• District\n"
            "• Area (hectares)\n"
            "• Estimated production\n\n"
            "The AI will predict yield in kg/ha with a confidence score and explain what factors matter most."
        ),
        'action': 'open_yield_predictor',
        'suggestions': ['Predict rice yield', 'Wheat yield estimate', 'What affects yield?'],
    },
    'fertilizer': {
        'keywords': [
            'fertilizer', 'fertiliser', 'nutrient', 'urea', 'dap', 'potash', 'npk',
            'nitrogen', 'phosphorus', 'potassium', 'manure', 'compost', 'khad',
            'खाद', 'उर्वरक', 'यूरिया', 'डीएपी', 'पोषण', 'नाइट्रोजन',
            'खाद डालना', 'खाद कितना', 'उर्वरक मात्रा', 'पोटाश', 'फास्फोरस',
            'खत', 'युरिया', 'नायट्रोजन', 'खत किती', 'फवारणी खत', 'माती परीक्षण',
        ],
        'response': (
            "🌱 I can help with fertilizer recommendations!\n\n"
            "Go to the **Fertilizer Guide** module. You'll need:\n"
            "• Your crop type\n"
            "• Area in hectares\n"
            "• Soil test status (Low/Medium/High N, P, K)\n\n"
            "💡 **Quick Guide:**\n"
            "Low N → Apply Urea (46% N)\n"
            "Low P → Apply DAP (46% P₂O₅)\n"
            "Low K → Apply MOP (60% K₂O)"
        ),
        'action': 'open_fertilizer',
        'suggestions': ['Fertilizer for wheat', 'Urea for 1 acre', 'Organic alternatives'],
    },
    'market': {
        'keywords': [
            'price', 'market', 'sell', 'mandi', 'rate', 'msp', 'minimum support',
            'buyer', 'trader', 'apm', 'apmc',
            'कीमत', 'मंडी', 'बाजार', 'भाव', 'एमएसपी', 'बेचना',
            'बाजारभाव', 'विक्री', 'भाव',
        ],
        'response': (
            "💰 For current market prices and MSP:\n\n"
            "📱 **Apps to use:**\n"
            "• **eNAM** – enam.gov.in (national mandi prices)\n"
            "• **Agmarknet** – agmarknet.gov.in\n"
            "• **NAFED App** – for MSP information\n"
            "• **Kisan Suvidha** – all-in-one farmer app\n\n"
            "📞 **Mandi helpline:** 1800-270-0224\n\n"
            "MSP 2024-25 key crops: Rice ₹2300/q | Wheat ₹2275/q | Soybean ₹4892/q"
        ),
        'action': None,
        'suggestions': ['Rice MSP 2024', 'Wheat mandi price', 'Soybean rate today'],
    },
    'weather': {
        'keywords': [
            'weather', 'rain', 'temperature', 'monsoon', 'drought', 'flood',
            'forecast', 'climate', 'humidity',
            'मौसम', 'बारिश', 'तापमान', 'पाऊस', 'हवामान', 'वर्षा', 'सूखा',
        ],
        'response': (
            "🌧️ For accurate weather forecasts:\n\n"
            "📱 **Recommended apps / websites:**\n"
            "• **Meghdoot App** – crop-specific weather advisories\n"
            "• **IMD** – mausam.imd.gov.in\n"
            "• **Damini App** – lightning alerts\n"
            "• **Kisan Suvidha** – weather + farming tips\n\n"
            "🌾 **Current season tip:** During June-September, watch for excess moisture which promotes fungal diseases. Apply preventive fungicide sprays."
        ),
        'action': None,
        'suggestions': ['Monsoon forecast', 'How to manage drought', 'Irrigation schedule'],
    },
    'irrigation': {
        'keywords': [
            'water', 'irrigation', 'drip', 'sprinkler', 'canal', 'bore', 'well',
            'moisture', 'flood irrigation', 'furrow',
            'पानी', 'सिंचाई', 'नहर', 'कुआं', 'ड्रिप', 'स्प्रिंकलर',
            'पाणी', 'ठिबक', 'फवारणी',
        ],
        'response': (
            "💧 Irrigation guidance:\n\n"
            "**Drip Irrigation** – saves 40–50% water; best for vegetables, fruits, sugarcane.\n"
            "**Sprinkler** – suitable for wheat, onion, oilseeds; uneven terrain.\n"
            "**Flood/Furrow** – traditional; suitable for rice.\n\n"
            "🏛️ **Subsidy:** Check **Pradhan Mantri Krishi Sinchai Yojana (PMKSY)** for up to 55% subsidy on drip/sprinkler.\n\n"
            "📞 Horticulture/Irrigation dept: 1800-180-1551"
        ),
        'action': None,
        'suggestions': ['Drip irrigation for onion', 'PMKSY subsidy', 'Water requirement for rice'],
    },
    'government': {
        'keywords': [
            'scheme', 'subsidy', 'loan', 'kcc', 'kisan credit', 'pm kisan', 'fasal bima',
            'insurance', 'government', 'yojana', 'sarkar',
            'योजना', 'सब्सिडी', 'कर्ज', 'बीमा', 'किसान क्रेडिट',
            'अनुदान', 'सरकार', 'योजना',
        ],
        'response': (
            "🏛️ Key Government Schemes for Farmers:\n\n"
            "1. **PM-KISAN** – ₹6,000/year income support (pmkisan.gov.in)\n"
            "2. **PM Fasal Bima Yojana** – Crop insurance at 2% premium\n"
            "3. **Kisan Credit Card (KCC)** – Credit up to ₹3 lakh @ 4%\n"
            "4. **Soil Health Card** – Free soil testing\n"
            "5. **PMKSY** – Irrigation subsidy (55% for drip)\n"
            "6. **eNAM** – Online mandi platform\n\n"
            "📞 Kisan Call Center: **1800-180-1551** (free, 24×7)"
        ),
        'action': None,
        'suggestions': ['PM Kisan registration', 'Fasal Bima claim', 'KCC application'],
    },
    'soil': {
        'keywords': [
            'soil', 'land', 'ph', 'organic matter', 'clay', 'loam', 'sand',
            'soil test', 'soil health',
            'मिट्टी', 'भूमि', 'जमीन', 'माती', 'pH', 'मृदा',
        ],
        'response': (
            "🌱 Soil health guidance:\n\n"
            "1. **Get a Soil Test** – Contact nearest KVK or use Soil Health Card scheme (free).\n"
            "2. **Ideal pH:** 6.0–7.5 for most crops. Add lime for acidic soil, gypsum for alkaline.\n"
            "3. **Organic Carbon:** Target >0.75%. Apply FYM/compost every year.\n"
            "4. **Micro-nutrients:** Zinc deficiency common in Maharashtra — apply ZnSO₄ @ 25 kg/ha.\n\n"
            "📲 **Soil Health Card App** – available on Google Play for testing records."
        ),
        'action': None,
        'suggestions': ['Soil test near me', 'How to improve soil pH', 'Zinc deficiency treatment'],
    },
    'help': {
        'keywords': [
            'help', 'what can you do', 'features', 'menu', 'options',
            'मदद', 'सहायता', 'मदत', 'काय करू',
        ],
        'response': (
            "🌾 Here's what I can help you with:\n\n"
            "| Module | What it does |\n"
            "|--------|-------------|\n"
            "| 🌿 Disease Detection | Upload leaf photo → AI diagnoses |\n"
            "| 📊 Yield Prediction | Estimate crop output with AI |\n"
            "| 🌱 Fertilizer Guide | NPK doses & cost estimate |\n"
            "| 🤖 Helpbot (this) | Any farming question |\n\n"
            "You can type in **English**, **हिंदी** or **मराठी**!"
        ),
        'action': 'show_help',
        'suggestions': ['Disease detection', 'Predict my yield', 'Fertilizer doses', 'Market prices'],
    },
}

# ── Crop-specific quick facts ─────────────────────────────────────────────────
CROP_FACTS = {
    'rice':      'Rice needs 1200–1500 mm water. Best planted in June–July in Maharashtra.',
    'wheat':     'Wheat sown October–November. Needs 6 irrigations in rabi season.',
    'cotton':    'Cotton is a 180-day crop. Bollworm is the biggest threat — use BT cotton.',
    'soybean':   'Soybean fixes nitrogen; inoculate seeds with Rhizobium before sowing.',
    'sugarcane': 'Sugarcane has 12–18 month cycle. Ratoon crop saves replanting cost.',
    'onion':     'Onion needs well-drained soil. Excess water causes neck rot.',
    'tomato':    'Tomato is prone to late blight in rainy season. Stake plants for support.',
    'potato':    'Potato needs cool weather. Harvest when tops die back naturally.',
}


class HelpBot:
    """Intent-based conversational assistant for farmers."""

    def __init__(self):
        self.history: list[dict] = []
        self._greet_done = False

    # ── Intent detection ────────────────────────────────────────────────────
    def detect_intent(self, message: str) -> str:
        msg_lower = message.lower()
        scores: dict[str, int] = {}

        for intent, data in INTENTS.items():
            score = sum(1 for kw in data['keywords'] if kw in msg_lower)
            if score:
                scores[intent] = score

        if scores:
            return max(scores, key=scores.get)

        # Check crop-specific queries
        for crop in CROP_FACTS:
            if crop in msg_lower:
                return 'crop_fact_' + crop

        return 'unknown'

    # ── Response generation ─────────────────────────────────────────────────
    def get_response(self, message: str) -> dict:
        intent = self.detect_intent(message)

        self.history.append({
            'role': 'user', 'text': message,
            'intent': intent, 'time': datetime.now().strftime('%H:%M')
        })

        # Crop-specific fact
        if intent.startswith('crop_fact_'):
            crop_key = intent.replace('crop_fact_', '')
            response    = f"🌾 About **{crop_key.title()}**:\n\n{CROP_FACTS[crop_key]}"
            action      = None
            suggestions = ['Disease in ' + crop_key, crop_key.title() + ' fertilizer', 'Yield prediction']
        elif intent in INTENTS:
            response    = INTENTS[intent]['response']
            action      = INTENTS[intent]['action']
            suggestions = INTENTS[intent]['suggestions']
        else:
            response = (
                "I understand you have a farming question. For best results, try asking about:\n\n"
                "• 🌿 **Disease** – describe leaf symptoms\n"
                "• 📊 **Yield** – predict crop output\n"
                "• 🌱 **Fertilizer** – NPK recommendations\n"
                "• 💰 **Market** – price information\n"
                "• 🏛️ **Schemes** – government subsidies\n\n"
                "You can also type in हिंदी or मराठी!"
            )
            action      = None
            suggestions = ['Disease detection', 'Yield prediction', 'Fertilizer guide', 'Help']

        self.history.append({
            'role': 'bot', 'text': response,
            'time': datetime.now().strftime('%H:%M')
        })

        return {
            'response':    response,
            'intent':      intent,
            'action':      action,
            'suggestions': suggestions,
        }

    def clear_history(self):
        self.history.clear()
        self._greet_done = False

    def get_history(self) -> list:
        return self.history
