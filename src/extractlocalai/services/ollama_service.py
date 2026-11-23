# services/ollama_service.py

import json
import time

from pathlib import Path
from ollama import chat
from extractlocalai.schemas.classification import DocumentClassification
from extractlocalai.schemas.sub_classification import DocumentSubClassification
from extractlocalai.schemas.data import DocumentData
from extractlocalai.core.config import settings

class OllamaService:

    @staticmethod
    def load_classifier(
        classifier_name: str
    ) -> dict | None:

        file_path = (
            Path(__file__).parent.parent
            / "schemas"
            / "classifiers"
            / f"{classifier_name}.json"
        )

        if not file_path.exists():
            return None

        with open(
            file_path,
            encoding="utf-8"
        ) as file:
            return json.load(file)
    
    @staticmethod
    def classify(text: str) -> DocumentClassification:

        text = text[:settings.max_document_length]

        definition = OllamaService.load_classifier("main")

        prompt = f"""
Tu es un expert documentaire français.

Tu dois choisir exactement UN type parmi cette liste :

{chr(10).join(f"- {item}" for item in definition["document_types"])}

IMPORTANT :

Le type "inconnu" doit être utilisé dès qu'il existe
un doute significatif.

Ne classe jamais un document dans une catégorie
simplement parce qu'elle semble être la plus proche.

Un document doit contenir des éléments explicites
permettant de confirmer la catégorie choisie.

Si aucun élément explicite ne permet de confirmer
une catégorie, réponds "inconnu".

Réponds uniquement avec un JSON valide :

{json.dumps(
    definition["response"],
    indent=2,
    ensure_ascii=False
)}

Document :

{text}
"""
        
        start_time = time.perf_counter()

        chat_options = (
            {"num_gpu": 0}
            if settings.ollama_force_cpu
            else {}
        )

        response = chat(
            model=settings.ollama_model,
            format="json",
            options=chat_options,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        classification = json.loads(response.message.content)

        classification_time_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        text_length = len(text)

        return DocumentClassification(
            document_type=classification ["document_type"],
            confidence=classification ["confidence"],
            classification_llm_model=settings.ollama_model,
            classification_time_ms=classification_time_ms,
            text_length=text_length
        )

    @staticmethod
    def sub_classify(
        text: str,
        classifier_name: str
    ) -> DocumentSubClassification:

        definition = (
            OllamaService.load_classifier(
                classifier_name
            )
        )

        prompt_definition = "\n".join(definition["prompt"])

        prompt = f"""
{prompt_definition}

Réponds uniquement avec un JSON valide :

{json.dumps(
    definition["response"],
    indent=2,
    ensure_ascii=False
)}

Document :

{text}
"""

        start = time.perf_counter()

        chat_options = (
            {"num_gpu": 0}
            if settings.ollama_force_cpu
            else {}
        )

        response = chat(
            model=settings.ollama_model,
            format="json",
            options=chat_options,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        result = json.loads(
            response.message.content
        )

        return DocumentSubClassification(
            sub_type=result["sub_type"],
            sub_confidence=result["confidence"],
            sub_classification_llm_model=settings.ollama_model,
            sub_classification_time_ms=int(
                (time.perf_counter() - start) * 1000
            )
        )

    @staticmethod
    def load_prompt(
        document_type: str
    ) -> dict | None:

        file_path = (
            Path(__file__)
            .parent.parent
            / "schemas"
            / "prompts"
            / f"{document_type}.json"
        )

        if not file_path.exists():
            return None

        with open(
            file_path,
            encoding="utf-8"
        ) as file:
            return json.load(file)

    @staticmethod
    def extract_data(
        text: str,
        document_type: str
    ) -> dict:

        definition = (
            OllamaService.load_prompt(
                document_type
            )
        )

        if definition is None:
            return DocumentData(
                data={},
                data_llm_model=settings.ollama_model,
                data_time_ms=0
            )

        prompt = f"""
{definition["prompt"]}

Réponds uniquement avec un JSON valide.

Schéma :

{json.dumps(
    definition["schema"],
    indent=2,
    ensure_ascii=False
)}

Document :

{text}
"""
        start_time = time.perf_counter()

        chat_options = (
            {"num_gpu": 0}
            if settings.ollama_force_cpu
            else {}
        )

        response = chat(
            model=settings.ollama_model,
            format="json",
            options=chat_options,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        payload = json.loads(response.message.content)

        data_time_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        return DocumentData(
            data=payload.get("data", {}),
            data_llm_model=settings.ollama_model,
            data_time_ms=data_time_ms
        )