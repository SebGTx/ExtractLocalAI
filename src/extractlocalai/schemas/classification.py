# schemas/classification.py

from pydantic import BaseModel

class DocumentClassification(BaseModel):    
    document_type: str
    confidence: float
    classification_llm_model: str
    classification_time_ms: int
    text_length: int