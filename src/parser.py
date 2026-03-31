import json
import os
import fitz  # PyMuPDF
import docx
from io import BytesIO

# ---------------- LOAD PREDEFINED JOBS ----------------
def load_predefined_jobs():
    """
    Load predefined job descriptions from JSON file.
    Returns a list of job dictionaries.
    """
    path = os.path.join("data", "job_descriptions.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return []
        except Exception as e:
            print(f"Error loading job descriptions: {e}")
            return []
    print(f"Job descriptions file not found at {path}")
    return []

# ---------------- EXTRACT TEXT FROM FILE BYTES ----------------
def extract_text_from_bytes(file_bytes, extension):
    """
    Extract text content from PDF or DOCX file bytes.
    
    Parameters:
        file_bytes (bytes): Raw file content
        extension (str): File extension, e.g., "pdf" or "docx"
    
    Returns:
        str: Extracted text, or error message
    """
    text = ""
    ext = extension.lower().lstrip(".")  # normalize extension

    try:
        if ext == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
        elif ext == "docx":
            doc = docx.Document(BytesIO(file_bytes))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            return f"Unsupported file type: {extension}"
    except Exception as e:
        return f"Error parsing file: {str(e)}"

    return text.strip()