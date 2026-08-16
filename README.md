# SWIFT Auditor - Multi-Agent SWIFT Message Extractor & Reasoning System

A Proof of Concept (POC) multi-agent system built using **LangGraph**, **Python**, and **Ollama** that extracts fields from raw SWIFT MT103 messages and reasons why transactions failed due to missing or invalid fields.

This application runs entirely locally on your machine and provides a sleek web interface to drop/upload SWIFT files (Text, PDF, Images) or paste raw text.

---

## Architecture

The system utilizes a multi-agent orchestration pattern built on LangGraph:

1. **Document Preprocessor**: Detects file types and extracts raw text. Integrates native PDF parsing and optional OCR.
2. **SWIFT Extractor Agent (Sub-Agent)**: Local LLM parses raw text blocks into structured JSON tags.
3. **Reasoning Agent (Sub-Agent)**: Logical checker checks mandatory requirements (KYC, accounts, details) and produces a transaction failure compliance report.

---

## Setup & Running the POC

Setting up the POC on a new machine takes under 5 minutes. Follow these simple steps:

### Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com/) (for running the LLM locally)

### Step 1: Start Ollama and Pull the Model
Open your terminal (Command Prompt, PowerShell, or Bash) and run:

```bash
# Pull the default lightweight model (3B parameters, ~2GB download)
ollama pull llama3.2
```

Ensure Ollama is running in the background. By default, it runs on `http://localhost:11434`.

### Step 2: Install Python Dependencies
In the project directory, run:

```bash
pip install -r requirements.txt
```

*(Optional)* If you wish to test OCR text extraction from **images** (PNG, JPG), install EasyOCR:
```bash
pip install easyocr
```

### Step 3: Run the FastAPI Server
Run the application server:

```bash
python main.py
```

The server will start at: `http://127.0.0.1:8000`

---

## Testing the POC

1. Open your web browser and navigate to `http://localhost:8000`.
2. Locate the pre-prepared sample files in the `samples/` directory:
   - **`sample_valid.txt`**: A correct SWIFT message (Passes verification).
   - **`sample_missing_beneficiary.txt`**: Missing tag `:59:` (Fails due to missing recipient/IBAN).
   - **`sample_missing_amount.txt`**: Missing tag `:32A:` (Fails due to missing date, amount, or currency).
3. Switch between **Upload Document** and **Paste Raw SWIFT Text** tab.
4. Upload or drag-and-drop any of the sample files, or copy/paste their content into the text area.
5. Click **Extract and Analyze Message**.
6. View the live agent execution steps. Once completed, the **Extracted SWIFT Fields (JSON)** and **Transaction Failure Analysis (Markdown Report)** will render.

---

## Customizing the Local LLM

If you prefer to use a different model that you already have pulled (e.g., `qwen2.5:3b`, `mistral`, or `llama3`), you can easily select it from the dropdown menu in the upper-right corner of the web interface.
