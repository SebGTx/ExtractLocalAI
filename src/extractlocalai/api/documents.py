# api/documents.py

from fastapi import APIRouter, UploadFile, File

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)

# Endpoint d'analyse
from pathlib import Path
from extractlocalai.schemas.document import DocumentAnalysis
from extractlocalai.services.docling_service import DoclingService
from extractlocalai.services.ollama_service import OllamaService

@router.post("/analyse")
async def classify_document(
    file: UploadFile = File(...)
):

    content = await file.read()

    # Text extraction and OCR
    docling_result = DoclingService.extract_text(content)

    # Document classification
    classification_result = OllamaService.classify(docling_result.text)

    # Document sub classification
    sub_classification_result = None
    classifier = (
        OllamaService.load_classifier(
            classification_result.document_type
        )
    )
    if classifier:
        sub_classification_result = (
            OllamaService.sub_classify(
                docling_result.text,
                classification_result.document_type
            )
        )

    # Data extraction
    document_type = (
        classification_result.document_type
    )
    if sub_classification_result:
        document_type = (
            f"{document_type}_"
            f"{sub_classification_result.sub_type}"
        )
    data_result = OllamaService.extract_data(docling_result.text, document_type)

    return DocumentAnalysis(
        filename=file.filename,
        pages=docling_result.pages,
        contains_ocr_text=docling_result.contains_ocr_text,
        contains_native_text=docling_result.contains_native_text,
        extraction_mode=docling_result.extraction_mode,
        extract_model="docling",
        extract_time_ms=docling_result.extract_time_ms,
        text_length=len(docling_result.text),
        truncated_text=docling_result.text[:500],
        document_type=classification_result.document_type,
        confidence=classification_result.confidence,
        classification_llm_model=classification_result.classification_llm_model,
        classification_time_ms=classification_result.classification_time_ms,
        sub_type=(
            sub_classification_result.sub_type
            if sub_classification_result
            else None
        ),
        sub_confidence=(
            sub_classification_result.sub_confidence
            if sub_classification_result
            else None
        ),
        sub_classification_llm_model=(
            sub_classification_result.sub_classification_llm_model
            if sub_classification_result
            else None
        ),
        sub_classification_time_ms=(
            sub_classification_result.sub_classification_time_ms
            if sub_classification_result
            else None
        ),
        data=data_result.data,
        data_llm_model=data_result.data_llm_model,
        data_time_ms=data_result.data_time_ms,
        total_time_ms=(
            docling_result.extract_time_ms + 
            classification_result.classification_time_ms + (
                sub_classification_result.sub_classification_time_ms
                if sub_classification_result
                else 0
            ) +
            data_result.data_time_ms
        )
    )
