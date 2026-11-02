"""
app.py
------
Flask web application for SpamGuard SMS Spam Detector.

Routes:
  GET  /          - serves the main HTML page
  POST /predict   - accepts JSON {"text": "..."}, returns prediction
  GET  /info      - returns model metadata as JSON

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import html
import json
import os
import re
import sys

import joblib
from flask import Flask, jsonify, render_template, request

# ─────────────────────────────────────────────────────────────────────────────
# Startup - load model artefacts
# ─────────────────────────────────────────────────────────────────────────────
# Use the directory that contains THIS file as the root so that gunicorn
# launched from any working directory can still locate model/ and templates/.
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
_REQUIRED = ["model.joblib", "vectorizer.joblib", "metadata.json"]

for fname in _REQUIRED:
    if not os.path.exists(os.path.join(MODEL_DIR, fname)):
        print(
            f"\n[ERROR] '{MODEL_DIR}/{fname}' not found.\n"
            "  Please run the training script first:\n\n"
            "      python train_model.py\n"
        )
        sys.exit(1)

MODEL      = joblib.load(os.path.join(MODEL_DIR, "model.joblib"))
VECTORIZER = joblib.load(os.path.join(MODEL_DIR, "vectorizer.joblib"))

with open(os.path.join(MODEL_DIR, "metadata.json"), "r") as _f:
    METADATA = json.load(_f)

TOP_SPAM_WORDS: set = set(METADATA.get("top_spam_words", []))

# ─────────────────────────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)


# ── Helper ────────────────────────────────────────────────────────────────────
def find_spam_words_in_text(text: str, spam_word_set: set) -> list:
    text_lower = text.lower()
    found = []
    tokens = re.findall(r"[a-z]+", text_lower)
    for token in tokens:
        if token in spam_word_set and token not in found:
            found.append(token)
    for i in range(len(tokens) - 1):
        bigram = tokens[i] + " " + tokens[i + 1]
        if bigram in spam_word_set and bigram not in found:
            found.append(bigram)
    return found


def highlight_spam_words(text: str, spam_words: list) -> str:
    safe = html.escape(text)
    if not spam_words:
        return safe
    sorted_words = sorted(spam_words, key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(w) for w in sorted_words) + r")\b"

    def replacer(match):
        return f'<mark class="spam-highlight">{match.group(0)}</mark>'

    return re.sub(pattern, replacer, safe, flags=re.IGNORECASE)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or not isinstance(data.get("text", ""), str):
        return jsonify({"error": "Request must be JSON with a 'text' field."}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Text cannot be empty."}), 400

    vec      = VECTORIZER.transform([text])
    label    = MODEL.predict(vec)[0]
    proba    = MODEL.predict_proba(vec)[0]
    classes  = list(MODEL.classes_)
    spam_idx = classes.index("spam")
    spam_pct = round(float(proba[spam_idx]) * 100, 2)

    spam_words  = find_spam_words_in_text(text, TOP_SPAM_WORDS)
    highlighted = highlight_spam_words(text, spam_words)

    return jsonify({
        "label"       : label,
        "spam_percent": spam_pct,
        "spam_words"  : spam_words,
        "highlighted" : highlighted,
    })


@app.route("/info")
def info():
    return jsonify(METADATA)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # debug=True only when run directly for local development.
    # gunicorn (used on Render) never calls this block.
    print("\n[START] SpamGuard is running at http://127.0.0.1:5000")
    print("  Press CTRL+C to stop.\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
