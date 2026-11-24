# How to Set Up and Run This Project

This guide provides step-by-step instructions for anyone receiving this project folder to get it up and running on their machine (macOS, Windows, or Linux).

---

## Prerequisites

Before starting, ensure you have:
1. **Python 3.10 to 3.14** installed on your system.
   - Verify by running: `python3 --version` (or `python --version` on Windows).
2. **A Free Google Gemini API Key**:
   - Get one in 1 minute from [Google AI Studio](https://aistudio.google.com/).

---

## Step 1: Open the Project Directory

Open your Terminal (macOS/Linux) or Command Prompt / PowerShell (Windows) and navigate into the `rag-app` directory:

```bash
cd path/to/rag-app
```

---

## Step 2: Create and Activate a Virtual Environment

Isolate the project dependencies from your global Python installation:

### On macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### On Windows (Command Prompt):
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(You will see `(.venv)` appear at the beginning of your command line prompt).*

---

## Step 3: Install Required Dependencies

Upgrade `pip` and install all necessary packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Step 4: Configure Your Gemini API Key

1. Look for the file named `.env.example` in the `rag-app` folder.
2. Create a copy of it and name the new file `.env`:
   - **macOS / Linux:**
     ```bash
     cp .env.example .env
     ```
   - **Windows:**
     ```cmd
     copy .env.example .env
     ```
3. Open the `.env` file in any text editor and replace the placeholder with your actual Gemini API key:
   ```env
   GEMINI_API_KEY="your_actual_gemini_api_key_here"
   LLM_MODEL="gemini-3.1-flash-lite"
   ```

---

## Step 5: Start the Application

Run the FastAPI local server with Uvicorn:

```bash
uvicorn main:app --reload --port 8000
```

You should see output similar to:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Application startup complete.
```

---

## Step 6: Access the Web Interface

1. Open your web browser (Chrome, Edge, Safari, Firefox).
2. Go to: **[http://localhost:8000](http://localhost:8000)**

---

## How to Use the Application

1. **Upload Documents:**
   - In the left pane (**Document Store**), click the upload area or drag and drop any `.pdf` or `.txt` file.
   - Click **Index Document**.
   - The file will be extracted, split into vector chunks, embedded, and stored locally in ChromaDB.
   - You will see it appear under **Indexed Corpus** with its chunk count.

2. **Ask Questions:**
   - In the right pane (**Query Console**), type any question related to the uploaded documents.
   - Press **`⌘ + Enter`** (or **`Ctrl + Enter`** on Windows) or click **Query**.
   - The system retrieves the most relevant passages and generates a grounded answer citing the source document filenames.

3. **Delete Documents:**
   - Click the trash icon next to any document in the list to purge both the file and its vector embeddings from ChromaDB.

---

## Common Questions & Troubleshooting

- **Error: "Model is experiencing high demand (503)"**
  - The default model is set to `gemini-3.1-flash-lite` in `.env`, which is fast and rarely encounters free-tier congestion. Avoid switching to heavy flagship models on free-tier keys.
- **Red squiggly lines under `import` statements in VS Code:**
  - Press `Cmd + Shift + P` (or `Ctrl + Shift + P`), type `Python: Select Interpreter`, and select the interpreter inside `./.venv/bin/python`.
- **How to stop the server:**
  - Press `Ctrl + C` in the terminal where Uvicorn is running.
