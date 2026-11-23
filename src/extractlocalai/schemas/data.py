# schemas/data.py

from pydantic import BaseModel
from typing import Any

class DataField(BaseModel):
    value: Any
    confidence: float

class DocumentData(BaseModel):
    data: dict[str, DataField]
    data_llm_model: str
    data_time_ms: int