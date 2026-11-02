"""
verify_routes.py  —  tests all three routes using Flask's built-in test client.
Works on Windows and Linux. Run: python verify_routes.py
This is equivalent to what gunicorn does but without needing gunicorn to run on Windows.
"""
import json
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# --- import the app ---
try:
    from app import app, METADATA, TOP_SPAM_WORDS, VECTORIZER, MODEL
    print("[PASS]  app imported successfully")
    print(f"        Flask app name : {app.name}")
    print(f"        Best model     : {METADATA['best_model']}")
    print(f"        Vocab size     : {METADATA.get('tfidf_vocab_size', '?')}")
    print(f"        Top spam words : {len(TOP_SPAM_WORDS)}")
except Exception as e:
    print(f"[FAIL]  import failed: {e}")
    sys.exit(1)

client = app.test_client()
ok = True

def check(label, passed, detail=""):
    global ok
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"  {tag}  {label:40s}  {detail}")
    if not passed:
        ok = False

print()
print("=" * 60)
print("  Route tests (Flask test client)")
print("=" * 60)

# --- GET / ---
r = client.get("/")
check(f"GET /  -> {r.status_code}", r.status_code == 200,
      f"HTML {len(r.data):,} bytes  contains SpamGuard={'SpamGuard' in r.data.decode()}")

# --- POST /predict  spam ---
r = client.post("/predict",
                data=json.dumps({"text": "WINNER!! Claim your free prize now!"}),
                content_type="application/json")
d = r.get_json()
check(f"POST /predict (spam)  -> {r.status_code}", r.status_code == 200,
      f"label={d.get('label')}  spam%={d.get('spam_percent')}")

# --- POST /predict  ham ---
r = client.post("/predict",
                data=json.dumps({"text": "Hey, are you coming to the party tonight?"}),
                content_type="application/json")
d = r.get_json()
check(f"POST /predict (ham)   -> {r.status_code}", r.status_code == 200,
      f"label={d.get('label')}  spam%={d.get('spam_percent')}")

# --- POST /predict  empty body must return 400 ---
r = client.post("/predict",
                data=json.dumps({"text": "   "}),
                content_type="application/json")
check(f"POST /predict (empty) -> {r.status_code}", r.status_code == 400,
      "400 returned as expected")

# --- POST /predict  no JSON must return 400 ---
r = client.post("/predict", data="not json", content_type="text/plain")
check(f"POST /predict (bad)   -> {r.status_code}", r.status_code == 400,
      "400 returned as expected")

# --- GET /info ---
r = client.get("/info")
d = r.get_json()
check(f"GET /info  -> {r.status_code}", r.status_code == 200,
      f"best_model={d.get('best_model')}  top_words={len(d.get('top_spam_words', []))}")

# --- highlight check ---
r = client.post("/predict",
                data=json.dumps({"text": "Claim your free prize now!"}),
                content_type="application/json")
d = r.get_json()
has_mark = "<mark" in d.get("highlighted", "")
check("highlight words in response", has_mark,
      f"highlighted={d.get('highlighted','')[:60]}")

print("=" * 60)
print(f"  OVERALL: {'ALL PASS -- ready to deploy to Render' if ok else 'SOME FAILURES -- fix before deploying'}")
print("=" * 60)
sys.exit(0 if ok else 1)
