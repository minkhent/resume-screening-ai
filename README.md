
---

# 🧠 Resume Screening AI

An enterprise-ready **AI Resume Screening System** that goes beyond keyword matching. It leverages **Semantic Search & Skill Normalization** to understand candidate context, detect skill gaps, and align profiles with predefined job roles.

---

## 🚀 Key Features

* **Semantic Skill Matching:** Uses `all-MiniLM-L6-v2` Transformers to map resumes to job competencies, including synonyms and aliases.
* **Skill Alias Normalization:** Detects variant expressions like `"multi-gpu"` ≈ `"distributed training"` or `"Jira"` ≈ `"agile management"`.
* **Experience & Context Analysis:** Evaluates seniority, leadership indicators, and role relevance.
* **Interactive Dashboard:** Streamlit + Plotly visualizations for candidate-job alignment.
* **Microservices Architecture:** FastAPI backend for inference; Streamlit frontend for reporting.
* **Performance Monitoring:** Tracks RAM, CPU/GPU usage, and inference latency.

---

## Dataset 
I use this dataset to train model
```text
 https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
```

## 🎬 Demo Video
[AI_resume_screening_demo.webm](https://github.com/user-attachments/assets/a7ff475d-4c9d-4c19-8b04-e33bf783be30)




## 🏗️ Project Structure

```text
resume-screening-ai/
├── api/
│   └── main.py           # FastAPI backend & hardware monitoring
├── app/
│   └── ui.py             # Streamlit frontend dashboard
├── data/
│   ├── cleaned_resume.csv
│   ├── resume.csv
│   └── job_descriptions.json
├── models/
│   └── resume_model_v1/  # Trained model, metrics, plots
├── scripts/
│   ├── model.py          # Model definition & training utilities
│   ├── train.py          # Main training entrypoint
│   ├── plot_per_class_metrics.py
│   └── utils/            # Helper modules: data, metrics, logging, seed
├── src/
│   ├── parser.py         # Resume text extraction (PDF/DOCX)
│   └── engine.py         # Semantic scoring & skill analysis
├── notebooks/
│   └── 01_dataset_analysis.ipynb
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

This project is optimized for **Python 3.10**.

### 1. Create Conda Environment

```bash
conda create -n resume_ai python=3.10 -y
conda activate resume_ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Install SpaCy language model:

```bash
pip install en_core_web_md-3.7.1-py3-none-any.whl
```

### 3. Run the Application

**Backend (FastAPI):**

```bash
uvicorn api.main:app --reload
```

**Frontend (Streamlit):**

```bash
streamlit run app/ui.py
```

---

## 🛠️ Core Workflow

1. **Preprocessing & Parsing:** Extract text from PDFs/DOCX resumes using `PyMuPDF` and `python-docx`.
2. **Skill Extraction:**

   * Exact keyword matches
   * Semantic matches via **sentence embeddings** (`SentenceTransformer`)
   * Normalization using **skill alias mapping** from `job_descriptions.json`
3. **Experience Analysis:** Detects years of experience, seniority, and leadership markers.
4. **Context Evaluation:** Compares resume content to job competencies and benchmarks.
5. **Weighted Scoring:** Combines skill coverage, experience, context, and keywords into a **final match percentage**.

---

## 📂 Model Artifacts (`models/resume_model_v1`)

This folder contains the trained resume classification model, evaluation metrics, and transformer components.

```text
models/resume_model_v1/
├── classifier.pt                  # PyTorch model weights
├── label_encoder.pkl              # Encodes job roles / classes
├── classification_report.csv      # Per-class precision, recall, F1
├── metrics.csv                    # Training & validation metrics
├── train.log                      # Training logs
├── plots/                         # Visualizations of model performance
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   ├── per_class_precision.png
│   ├── per_class_recall.png
│   ├── per_class_f1.png
│   └── topk_accuracy.png
└── transformer/                   # Sentence Transformer model for embeddings
    ├── model.safetensors          # Pretrained transformer weights
    ├── tokenizer.json             # Tokenizer data
    ├── vocab.txt                  # Vocabulary
    ├── config.json                # Model configuration
    ├── modules.json               # Transformer modules structure
    ├── sentence_bert_config.json  # SBERT config
    └── 1_Pooling/2_Normalize/...  # Optional layers for embeddings
```

### 📝 Key Artifacts

* **`classifier.pt`** – Trained PyTorch classifier for mapping resume embeddings to job roles.
* **`label_encoder.pkl`** – Maps numerical labels to job role names.
* **`metrics.csv` & `classification_report.csv`** – Evaluation metrics: accuracy, precision, recall, F1, and per-class performance.
* **`plots/`** – Training and evaluation visualizations (accuracy/loss curves, confusion matrix, per-class metrics).
* **`transformer/`** – Sentence Transformer used to generate semantic embeddings from resumes. Includes weights, tokenizer, configuration, and optional pooling/normalization layers.
* **`train.log`** – Detailed log of training progress and statistics.

> This structure allows seamless inference: load the transformer, encode resumes, and classify them into job roles.

---

## 📊 Scoring Formula

* **Skill Coverage (35%)** – Semantic + keyword match
* **Experience Signal (20%)** – Seniority detection
* **Context Similarity (25%)** – Embedding similarity
* **Keyword Boost (20%)** – Presence of key role keywords

**Semantic similarity (cosine):**

$$
\text{similarity} = \frac{A \cdot B}{|A| |B|}
$$

---

## 📦 Tech Stack

* **Python 3.10**
* **NLP / AI:** `sentence-transformers`, `spacy`
* **Web & Dashboard:** `FastAPI`, `Streamlit`, `Plotly`
* **File Parsing:** `PyMuPDF`, `python-docx`
* **Data Handling:** `numpy`, `pandas`
* **Monitoring:** `psutil` for RAM & CPU usage
* **Optional Containerization:** `Docker`, `Kubernetes`

---
