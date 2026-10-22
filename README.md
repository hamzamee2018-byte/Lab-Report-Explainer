# 🧪 Lab Report Explainer

A Streamlit app that explains a medical lab report in plain language. Upload a PDF, JPG or PNG and the app (powered by a Google Gemini Flash model) shows:

- a short **summary** of the report
- a table of values **outside the normal range**, marked **High** or **Low**
- a simple **explanation** of each abnormal value
- **general lifestyle tips**
- **questions to ask your doctor**

> ⚠️ **Disclaimer:** This app is for general education only. It is **not a medical diagnosis**, it is not medical advice, and it **does not replace a doctor**. AI can make mistakes. Always discuss your results with a qualified healthcare professional.

## How it works

1. **Upload**: the file is checked (not empty, under 10 MB, valid PDF/JPG/PNG, PDFs up to 15 pages, no password).
2. **Text extraction**:
   - Digital PDFs: text is read locally with `pypdf`.
   - Scanned PDFs and images: Gemini's vision capability transcribes the text (OCR). If the scan is unreadable, the app says so.
3. **Analysis**: the extracted text is sent to Gemini, which returns structured JSON that the app validates and displays.

## Project files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Run locally

Requires Python 3.10+.

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. add your Gemini API key (get one at https://aistudio.google.com/apikey)
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<'EOF'
GEMINI_API_KEY = "your-key-here"
EOF

# 4. run
streamlit run app.py
```

Instead of a secrets file you can use an environment variable:

```bash
export GEMINI_API_KEY="your-key-here"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key-here"
streamlit run app.py
```

**Never put your API key in the code and never commit `.streamlit/secrets.toml` to GitHub.**

## Configuration

| Setting | Required | Where to set it | Default |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | Streamlit secrets or environment variable | none |
| `GEMINI_MODEL` | No | Streamlit secrets or environment variable | `gemini-3.7-flash` |

Gemini model names change over time. If you see a "model was not found" message, set `GEMINI_MODEL` to a Flash model that is available to your key (see the [Gemini models page](https://ai.google.dev/gemini-api/docs/models)).

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository (without `.streamlit/secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub and click **Create app**.
3. Choose your repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Click **Deploy**.

## Error handling

The app shows a clear message for: a missing or rejected API key, an empty file, an unsupported or corrupt file, a password-protected PDF, a PDF that is too long, an unreadable/blurry scan, a file that is not a lab report, API rate limits or outages, and an unexpected AI response format.

## Privacy

Uploaded files are sent to Google's Gemini API for processing. This app does not store files or results, but you should still avoid uploading anything you are not comfortable sharing (consider covering your name, ID number and address first). Review Google's Gemini API data-use terms for your account type, especially for free-tier keys.

## Limitations

- Accuracy depends on the report's quality and the AI model; always verify against the original report.
- Reference ranges differ between labs, so the app relies on the ranges printed in your report.
- The app cannot diagnose, recommend treatment, or replace professional medical care.
