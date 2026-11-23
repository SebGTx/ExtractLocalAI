# schemas/document.py

from pydantic import BaseModel
from extractlocalai.schemas.data import DataField

class DocumentAnalysis(BaseModel):
    filename: str
    pages: int
    contains_ocr_text: bool
    contains_native_text: bool
    extraction_mode: str
    extract_model: str
    extract_time_ms: int
    text_length: int
    truncated_text: str
    document_type: str
    confidence: float
    classification_llm_model: str
    classification_time_ms: int
    sub_type: str | None = None
    sub_confidence: float | None = None
    sub_classification_llm_model: str | None = None
    sub_classification_time_ms: int | None = None
    data: dict[str, DataField]
    data_llm_model: str
    data_time_ms: int
    total_time_ms: int
