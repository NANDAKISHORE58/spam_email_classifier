python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/data_preprocessing.py
python -m src.model_training
streamlit run app.py
