# Spam Email Classifier

A simple Streamlit-based demo for spam vs ham email classification using UCI Spambase dataset and Naive Bayes model. This project demonstrates basic ML model deployment and interactive web apps.

## Features

* Slider-based feature demo for UCI Spambase numeric features
* Text-based classifier for raw email/SMS content
* Pre-trained Naive Bayes models with vectorizers
* Clean Streamlit UI with probability metrics
* Lightweight and easy to deploy

## Project Structure

```
spam-classifier/
│
├── src/                # Inference and model logic
├── models/             # Saved models and vectorizers
├── config.py           # Paths and model configuration
├── app.py              # Slider-based numeric demo
├── text_app.py         # Text input classifier
├── requirements.txt    # Python dependencies
└── README.md
```

## Installation

1. Clone the repository

```
git clone [YOUR-REPO-URL]
cd spam-classifier
```

2. Create virtual environment and install dependencies

```
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

## Training Models

**Numeric model (UCI Spambase):**
```
python src/data_preprocessing.py
python -m src.model_training
```

**Text model:**
```
python src/text_model_training.py
```

## Usage

**Slider demo (numeric features):**
```
streamlit run app.py
```

**Text classifier:**
```
streamlit run text_app.py
```

## Example Usage

**Slider Demo:** Adjust `word_freq_free`, `word_freq_your`, `capital_run_length` sliders and click Predict.

**Text Demo:** 
```
Input: "Congratulations! You won $1000 gift card. Click here now!"
Output: Spam 🛑 (95.2% probability)
```

## Technologies Used

* Python, Streamlit
* Scikit-learn (Naive Bayes)
* NumPy, Pandas
* Joblib (model persistence)
* UCI Spambase dataset

## Author

**Nandakishore**  
GitHub: [https://github.com/NANDAKISHORE58](https://github.com/NANDAKISHORE58)
