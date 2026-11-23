# schemas/sub_classification.py

from pydantic import BaseModel


class DocumentSubClassification(BaseModel):
    sub_type: str
    sub_confidence: float
    sub_classification_llm_model: str
    sub_classification_time_ms: int