# Training Package → Element-level Skill Statements (BART) Web App

A Streamlit web app to:
1) Upload a training package CSV (any package)
2) Auto-detect the best extractor (or manually pick one)
3) Normalize to Unit → Element → Performance Criteria groups
4) Generate one element-level BART skill statement per element with QA + auto-rewrite
5) Download:
   - rsd_output.csv (RSD template format)
   - traceability.csv (PCs, prompt, QA checks)

## Quick start (local)
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env to include OPENAI_API_KEY (and optionally OPENAI_MODEL)

streamlit run app.py
```

## Deploy to Streamlit Cloud
- Push this repo to GitHub
- In Streamlit Cloud: set secrets:
```toml
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-5.2"
```

## Supported CSV formats (via Extractor Registry)
- **training.gov.au blob**: a column containing both "Elements" and "Performance Criteria" text
- **row-per-performance-criteria**: explicit Element + Performance Criteria columns (one row per PC)

Add more extractors in `core/extractors/` and register them in `core/extractor.py`.
