"""
smoke_test.py  —  verify gunicorn + all routes work before deploying to Render.
Run from the spamguard/ folder:  python smoke_test.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

PORT = 5099
BASE = f"http://127.0.0.1:{PORT}"

# Locate gunicorn on this machine
GUNICORN_CANDIDATES = [
    r"C:\Users\yetur\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python311\Scripts\gunicorn.exe",
    "gunicorn",  # fallback: if it's on PATH
]

gunicorn_cmd = None
for candidate in GUNICORN_CANDIDATES:
    if os.path.isfile(candidate) or candidate == "gunicorn":
        gunicorn_cmd = candidate
        break

if gunicorn_cmd is None:
    print("[FAIL] gunicorn executable not found. Run: pip install gunicorn")
    sys.exit(1)

print(f"[INFO] Starting gunicorn on port {PORT} ...")
proc = subprocess.Popen(
    [gunicorn_cmd, "app:app", "--bind", f"127.0.0.1:{PORT}",
     "--workers", "1", "--timeout", "30"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

time.sleep(4)  # give it a moment to start

results = []
ok = True


def check(label, status, detail=""):
    global ok
    tag = "PASS" if "FAIL" not in label else "FAIL"
    marker = "[PASS]" if "FAIL" not in label else "[FAIL]"
    line = f"  {marker}  {label:30s}  {detail}"
    results.append(line)
    if "FAIL" in label:
        ok = False


# ── Test 1: GET / ────────────────────────────────────────────────────────────
try:
    r = urllib.request.urlopen(f"{BASE}/", timeout=5)
    body = r.read()
    check(f"GET /  -> {r.status}", "OK", f"HTML {len(body):,} bytes")
except Exception as e:
    check("FAIL GET /", "FAIL", str(e))

# ── Test 2: POST /predict (spam message) ─────────────────────────────────────
try:
    payload = json.dumps({"text": "WINNER!! Claim your free prize now!"}).encode()
    req = urllib.request.Request(
        f"{BASE}/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    r = urllib.request.urlopen(req, timeout=5)
    d = json.loads(r.read())
    check(
        f"POST /predict (spam) -> {r.status}", "OK",
        f"label={d['label']}  spam%={d['spam_percent']}"
    )
except Exception as e:
    check("FAIL POST /predict (spam)", "FAIL", str(e))

# ── Test 3: POST /predict (ham message) ──────────────────────────────────────
try:
    payload = json.dumps({"text": "Hey, are you coming to the party tonight?"}).encode()
    req = urllib.request.Request(
        f"{BASE}/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    r = urllib.request.urlopen(req, timeout=5)
    d = json.loads(r.read())
    check(
        f"POST /predict (ham)  -> {r.status}", "OK",
        f"label={d['label']}  spam%={d['spam_percent']}"
    )
except Exception as e:
    check("FAIL POST /predict (ham)", "FAIL", str(e))

# ── Test 4: POST /predict empty body  must return 400 ─────────────────────────
try:
    payload = json.dumps({"text": ""}).encode()
    req = urllib.request.Request(
        f"{BASE}/predict", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        check("FAIL POST /predict empty", "FAIL", "expected 400 but got 200")
    except urllib.error.HTTPError as he:
        check(f"POST /predict (empty) -> {he.code}", "OK", "400 returned as expected")
except Exception as e:
    check("FAIL POST /predict empty", "FAIL", str(e))

# ── Test 5: GET /info ─────────────────────────────────────────────────────────
try:
    r = urllib.request.urlopen(f"{BASE}/info", timeout=5)
    d = json.loads(r.read())
    check(
        f"GET /info  -> {r.status}", "OK",
        f"best_model={d['best_model']}  vocab={d.get('tfidf_vocab_size', '?')}"
    )
except Exception as e:
    check("FAIL GET /info", "FAIL", str(e))

proc.terminate()
proc.wait()

print()
print("=" * 56)
print("  gunicorn smoke-test results")
print("=" * 56)
for line in results:
    print(line)
print("=" * 56)
print(f"  OVERALL: {'ALL PASS — safe to deploy' if ok else 'SOME FAILURES — fix before deploying'}")
print("=" * 56)
sys.exit(0 if ok else 1)
