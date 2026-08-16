import io
import os

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError(
            "The 'pypdf' library is required to extract text from PDF files. "
            "Please install it using: pip install pypdf"
        )
    
    pdf_file = io.BytesIO(pdf_bytes)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    if not text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "The PDF might contain scanned images. If so, please convert it to an image and try OCR, or use a text-based PDF."
        )
    return text

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extracts text from an image using EasyOCR or PyTesseract, with graceful failure if not installed."""
    # Attempt EasyOCR first
    try:
        import easyocr
        import numpy as np
        from PIL import Image
        
        img = Image.open(io.BytesIO(image_bytes))
        img_np = np.array(img)
        
        # Initialize easyocr Reader (English) - note that it will download weights on first run
        # We specify gpu=False by default to avoid CUDA errors on machines without dedicated GPU
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_np)
        
        text = "\n".join([res[1] for res in results])
        if text.strip():
            return text
    except ImportError:
        pass  # Fallback to PyTesseract
        
    # Attempt PyTesseract fallback
    try:
        import pytesseract
        from PIL import Image
        
        # Auto-detect Tesseract installation directory on Windows
        if os.name == 'nt':
            standard_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe")
            ]
            for path in standard_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(img)
    except (ImportError, Exception):
        # Graceful fallback explaining how to enable OCR
        raise RuntimeError(
            "OCR (Optical Character Recognition) libraries are not installed or configured on this machine.\n\n"
            "To enable lightweight and fast image OCR (Windows, ~40MB total, no PyTorch needed):\n"
            "1. Install Tesseract OCR: Run `winget install UB-Mannheim.TesseractOCR` in your terminal\n"
            "2. Install PyTesseract: Run `pip install pytesseract` in your terminal\n"
            "3. Restart the FastAPI server\n\n"
            "Alternatively, please upload a Text (.txt) or PDF (.pdf) file, or paste raw text instead."
        )

def preprocess_document(content: bytes, filename: str) -> str:
    """Preprocesses files of different formats (PDF, Image, Text) and returns raw text content."""
    ext = os.path.splitext(filename.lower())[1]
    
    if ext == '.pdf':
        return extract_text_from_pdf(content)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        return extract_text_from_image(content)
    elif ext in ['.txt', '.swift', '.json', '.log']:
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return content.decode('latin-1')
    else:
        # Attempt text decoding as a default fallback
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                "Please upload a text file (.txt), PDF (.pdf), or an image (.png, .jpg, .jpeg)."
            )
