"""Cab-display messages in English, Hindi and Telugu.

Short imperative sentences so an operator can take them in at a glance. The Hindi and
Telugu wording is simple and direct, but it has not been reviewed by native-speaking
operators; have it checked (and tested on the actual display font) before field use.

Placeholders are filled with ``str.format``: {gear}, {pct}, {target}, {limit}, {dist}, {temp}.
"""

from __future__ import annotations

LANGS = ("en", "hi", "te")

# code -> (priority, {lang: text}); priority 1 = safety, 2 = fault, 3 = efficiency
MESSAGES: dict[str, tuple[int, dict[str, str]]] = {
    "OVERSPEED": (
        1,
        {
            "en": "Over the speed limit. Slow down to {limit} km/h",
            "hi": "गति सीमा से ज़्यादा। {limit} किमी/घंटा पर आएँ",
            "te": "వేగ పరిమితి దాటింది. {limit} కిమీ/గంకి తగ్గించండి",
        },
    ),
    "NO_NEUTRAL": (
        1,
        {
            "en": "Do not coast in neutral. Use engine brake",
            "hi": "न्यूट्रल में गाड़ी न चलाएँ। इंजन ब्रेक का उपयोग करें",
            "te": "న్యూట్రల్‌లో నడపవద్దు. ఇంజిన్ బ్రేక్ వాడండి",
        },
    ),
    "BUMP_AHEAD": (
        1,
        {
            "en": "Speed breaker in {dist} m. Slow to {limit} km/h",
            "hi": "{dist} मीटर आगे स्पीड ब्रेकर। {limit} किमी/घंटा पर आएँ",
            "te": "{dist} మీటర్ల ముందు స్పీడ్ బ్రేకర్. {limit} కిమీ/గంకి తగ్గించండి",
        },
    ),
    "SMOOTH_BRAKE": (
        1,
        {
            "en": "Hard braking. Brake earlier and gently",
            "hi": "तेज़ ब्रेक। पहले और धीरे से ब्रेक लगाएँ",
            "te": "గట్టి బ్రేకింగ్. ముందుగా, మెల్లగా బ్రేక్ వేయండి",
        },
    ),
    "BRAKE_HOT": (
        2,
        {
            "en": "Brakes hot ({temp} C). Stop to cool, use a low gear",
            "hi": "ब्रेक गर्म ({temp} °C)। रुककर ठंडा करें, लो गियर लगाएँ",
            "te": "బ్రేకులు వేడిగా ఉన్నాయి ({temp} °C). ఆగి చల్లార్చండి, లో గేర్ వాడండి",
        },
    ),
    "ENGINE_HOT": (
        2,
        {
            "en": "Engine overheating ({temp} C). Stop safely",
            "hi": "इंजन ज़्यादा गर्म ({temp} °C)। सुरक्षित जगह रुकें",
            "te": "ఇంజిన్ ఎక్కువగా వేడెక్కింది ({temp} °C). సురక్షితంగా ఆపండి",
        },
    ),
    "LOW_OIL": (
        2,
        {
            "en": "Low oil pressure. Stop the engine now",
            "hi": "ऑयल प्रेशर कम। इंजन तुरंत बंद करें",
            "te": "ఆయిల్ ప్రెషర్ తక్కువ. వెంటనే ఇంజిన్ ఆపండి",
        },
    ),
    "OVERLOAD": (
        2,
        {
            "en": "Overloaded by {pct}%. Unload before driving",
            "hi": "{pct}% ओवरलोड। चलने से पहले माल उतारें",
            "te": "{pct}% ఓవర్‌లోడ్. బయలుదేరే ముందు లోడ్ తగ్గించండి",
        },
    ),
    "FUEL_LOSS": (
        2,
        {
            "en": "Fuel level dropping fast. Check for a leak",
            "hi": "डीज़ल का स्तर तेज़ी से गिर रहा है। लीकेज जाँचें",
            "te": "డీజిల్ స్థాయి వేగంగా తగ్గుతోంది. లీక్ ఉందేమో చూడండి",
        },
    ),
    "SHIFT_UP": (
        3,
        {
            "en": "Shift up to gear {gear}. Saves {pct}% fuel",
            "hi": "गियर {gear} में शिफ्ट करें। {pct}% डीज़ल बचेगा",
            "te": "గేర్ {gear}కి మార్చండి. {pct}% డీజిల్ ఆదా",
        },
    ),
    "SHIFT_DOWN": (
        3,
        {
            "en": "Shift down to gear {gear}. Engine is straining",
            "hi": "गियर {gear} पर आएँ। इंजन पर ज़ोर पड़ रहा है",
            "te": "గేర్ {gear}కి తగ్గించండి. ఇంజిన్‌పై భారం ఎక్కువ",
        },
    ),
    "EASE_OFF": (
        3,
        {
            "en": "Ease off. Target {target} km/h saves {pct}% fuel",
            "hi": "रफ़्तार कम करें। {target} किमी/घंटा पर {pct}% डीज़ल बचेगा",
            "te": "వేగం తగ్గించండి. {target} కిమీ/గంకి {pct}% డీజిల్ ఆదా",
        },
    ),
    "STOP_AHEAD": (
        3,
        {
            "en": "Stop ahead in {dist} m. Lift off the accelerator early",
            "hi": "{dist} मीटर आगे रुकना है। पहले से एक्सीलेटर छोड़ें",
            "te": "{dist} మీటర్ల ముందు ఆగాలి. ముందుగానే యాక్సిలరేటర్ వదలండి",
        },
    ),
    "DESCENT_GEAR": (
        3,
        {
            "en": "Steep descent. Use gear {gear} (engine brake)",
            "hi": "तीखी ढलान। गियर {gear} (इंजन ब्रेक) लगाएँ",
            "te": "నిటారైన ఏటవాలు. గేర్ {gear} (ఇంజిన్ బ్రేక్) వాడండి",
        },
    ),
    "IDLE_OFF": (
        3,
        {
            "en": "Waiting? Switch the engine off",
            "hi": "रुके हैं? इंजन बंद करें",
            "te": "ఆగి ఉన్నారా? ఇంజిన్ ఆపండి",
        },
    ),
}


def render(code: str, lang: str = "en", **params) -> tuple[int, str]:
    """Return (priority, text) for a message code in the requested language."""
    priority, texts = MESSAGES[code]
    text = texts.get(lang) or texts["en"]
    try:
        return priority, text.format(**params)
    except KeyError as exc:
        raise KeyError(f"message {code!r} needs parameter {exc}") from None
