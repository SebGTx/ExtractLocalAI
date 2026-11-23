# services/docling_service.py

import time

from pathlib import Path
from tempfile import NamedTemporaryFile

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrMode
from extractlocalai.schemas.extraction import DocumentExtraction

class DoclingService:

    @staticmethod
    def extract_text(content: bytes) -> DocumentExtraction:

        start = time.perf_counter()

        with NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(content)

            pdf_path = Path(tmp.name)

        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options.mode = (
                OcrMode.FULL_PAGE
            )
            pipeline_options.do_table_structure = False

            pdf_option = PdfFormatOption(pipeline_options=pipeline_options)

            converter = DocumentConverter(
                format_options={
                    "pdf": pdf_option
                }
            )

            result = converter.convert(pdf_path)

            document = result.document.export_to_dict()

            texts = document.get("texts", [])

            text = "\n".join(
                item["text"]
                for item in texts
                if item.get("text")
            )

            pages = len(document.get("pages", {}))

            pictures = document.get("pictures", [])

            contains_ocr_text = any(
                picture.get("children")
                for picture in pictures
            )

            contains_native_text = any(
                not text.get("parent", {})
                    .get("$ref", "")
                    .startswith("#/pictures/")
                for text in texts
            )

            if contains_native_text and contains_ocr_text:
                extraction_mode = "mixed"
            elif contains_native_text:
                extraction_mode = "text"
            elif contains_ocr_text:
                extraction_mode = "ocr"
            else:
                extraction_mode = "unknown"

            extract_time_ms = int(
                (time.perf_counter() - start) * 1000
            )
    
            return DocumentExtraction(
                pages=pages,
                contains_ocr_text=contains_ocr_text,
                contains_native_text=contains_native_text,
                extraction_mode=extraction_mode,
                extract_time_ms=extract_time_ms,
                text=text
            )

        finally:
            pdf_path.unlink(missing_ok=True)        