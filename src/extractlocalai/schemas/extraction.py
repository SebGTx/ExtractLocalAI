# schemas/extraction.py

from pydantic import BaseModel

class DocumentExtraction(BaseModel):
    pages: int
    contains_ocr_text: bool
    contains_native_text: bool
    extraction_mode: str
    extract_time_ms: int
    text: str
