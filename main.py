import os
import uvicorn
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from preprocessor import preprocess_document
from graph import compiled_app

app = FastAPI(
    title="SWIFT Extractor & reasoning agent POC",
    description="FastAPI backend serving the LangGraph multi-agent SWIFT auditing system."
)

# Serve the static UI files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def read_index():
    """Serves the main single-page application interface."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend static UI file index.html not found.")

@app.post("/process")
async def process_swift_message(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    model_name: str = Form("llama3.2"),
    base_url: str = Form("http://localhost:11434")
):
    """
    Accepts raw SWIFT message (as plain text or uploaded txt/pdf/image file)
    and passes it through the document preprocessor and LangGraph multi-agent flow.
    """
    raw_text = ""
    
    # 1. Document Preprocessing Step
    try:
        if file is not None:
            # Read content from upload
            content_bytes = await file.read()
            raw_text = preprocess_document(content_bytes, file.filename)
        elif text is not None and text.strip():
            raw_text = text
        else:
            raise HTTPException(
                status_code=400, 
                detail="Invalid input. Please upload a file (TXT/PDF/Image) or paste raw SWIFT message text."
            )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="The extracted SWIFT message text is empty.")

    # 2. Run LangGraph Multi-Agent Workflow
    try:
        initial_state = {
            "raw_message": raw_text,
            "model_name": model_name,
            "base_url": base_url,
            "extracted_fields": {},
            "reasoning_report": "",
            "errors": []
        }
        
        # Execute compiled LangGraph workflow synchronously
        final_state = compiled_app.invoke(initial_state)
        
        return {
            "extracted_fields": final_state.get("extracted_fields", {}),
            "reasoning_report": final_state.get("reasoning_report", ""),
            "errors": final_state.get("errors", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-Agent system error: {str(e)}")

if __name__ == "__main__":
    # Start the FastAPI server locally on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
