# 🛡️ SpamGuard – SMS Spam Detector

**B.Tech AI & ML Mini Project**

An end-to-end machine learning web application that classifies SMS messages as
**Spam** or **Not Spam** using TF-IDF features and two ML models (Multinomial
Naive Bayes & Logistic Regression).

---

## 📁 Project Structure

```
spamguard/
├── data/                  ← place your dataset here
│   └── (SMSSpamCollection or spam.csv)
├── model/                 ← auto-created by train_model.py
│   ├── model.joblib
│   ├── vectorizer.joblib
│   └── metadata.json
├── templates/
│   └── index.html         ← single-page web UI
├── app.py                 ← Flask server
├── train_model.py         ← ML training pipeline
├── make_sample_data.py    ← generates fake practice data
├── requirements.txt       ← Python dependencies
└── README.md              ← this file
```

---

## 🚀 Quick Start (Windows)

### Step 1 – Install Python 3.10+
1. Go to https://www.python.org/downloads/
2. Download the latest Python 3.x for Windows
3. Run the installer and **tick "Add Python to PATH"** before clicking Install

### Step 2 – Open the project folder
1. Press `Win + R`, type `cmd`, press Enter
2. Navigate to the folder:
   ```
   cd C:\Users\yetur\Downloads\Documents\spamguard
   ```

### Step 3 – Install dependencies
```
pip install -r requirements.txt
```

### Step 4 – Get a dataset (choose one)

**Option A – Practice data (no download needed):**
```
python make_sample_data.py
```
⚠️ Fake data only – for testing the pipeline, not for real results.

**Option B – Real UCI dataset:**
1. Download from https://archive.ics.uci.edu/dataset/228/sms+spam+collection
2. Extract `SMSSpamCollection` into the `data/` folder

**Option C – Kaggle dataset:**
1. Download from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Put `spam.csv` into the `data/` folder

### Step 5 – Train the model
```
python train_model.py
```
This prints accuracy, F1, and the confusion matrix, then saves the model.

### Step 6 – Start the web app
```
python app.py
```
Open your browser and go to: **http://127.0.0.1:5000**

---

## 🌐 API Endpoints

| Method | Route      | Description                                |
|--------|------------|--------------------------------------------|
| GET    | `/`        | Main web page                              |
| POST   | `/predict` | Classify a message (JSON body)             |
| GET    | `/info`    | Return model metrics and metadata as JSON  |

### POST /predict – Example

**Request:**
```json
{ "text": "WINNER!! You have won £1000. Call NOW." }
```

**Response:**
```json
{
  "label"       : "spam",
  "spam_percent": 99.87,
  "spam_words"  : ["winner", "won", "call"],
  "highlighted" : "WINNER!! You have <mark class=\"spam-highlight\">won</mark> ..."
}
```

---

## 🧠 ML Details

| Setting          | Value                                |
|------------------|--------------------------------------|
| Vectoriser       | TF-IDF (unigrams + bigrams, min_df=2)|
| Model 1          | Multinomial Naive Bayes (α = 0.1)   |
| Model 2          | Logistic Regression (balanced class) |
| Train/Test split | 80% / 20%, stratified               |
| Selection metric | Spam F1-Score (not accuracy)         |

**Why F1, not accuracy?**
Spam is only ~13% of messages. A lazy model that always says "ham" would be
87% accurate yet completely useless. F1-score rewards both catching real spam
(recall) and not wrongly flagging normal messages (precision).

---

## ⚠️ Limitations

- Trained on **English SMS** data only
- May miss **new spam tricks**, slang, or coded language
- **Not** a replacement for a production spam filter
- Performance on non-English messages is untested

---

## 📚 Dataset Credit

SMS Spam Collection Dataset – UCI Machine Learning Repository  
Almeida, T.A., Hidalgo, J.M.G., Yamakami, A. (2011)  
https://archive.ics.uci.edu/dataset/228/sms+spam+collection

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** – ML models and TF-IDF
- **pandas / numpy** – data handling
- **Flask** – web server
- **joblib** – model serialisation
- **HTML + CSS + Vanilla JS** – frontend (no heavy framework)
