import json
from pathlib import Path

import pytest

from extractlocalai.services.docling_service import DoclingService
from extractlocalai.services.ollama_service import OllamaService


FIXTURES_DIR = Path("tests/fixtures")

PDF_FILES = sorted(
    FIXTURES_DIR.rglob("*.pdf")
)

def load_expected(pdf_path: Path) -> dict:
    specific_expected = pdf_path.with_suffix(
        ".expected.json"
    )

    if specific_expected.exists():
        expected_path = specific_expected

    else:
        expected_path = (
            pdf_path.parent
            / "expected.json"
        )

    if not expected_path.exists():
        raise FileNotFoundError(
            f"Aucun fichier expected trouvé pour "
            f"{pdf_path.name}"
        )

    with open(
        expected_path,
        encoding="utf-8"
    ) as file:
        return json.load(file)


@pytest.mark.parametrize(
    "pdf_path",
    PDF_FILES,
    ids=lambda path: (
        f"{path.parent.name}/"
        f"{path.stem}"
    )
)
def test_document_analysis(
    pdf_path: Path
):

    expected = load_expected(pdf_path)

    content = pdf_path.read_bytes()

    #
    # Extraction Docling
    #
    extraction = (
        DoclingService.extract_text(
            content
        )
    )

    assert extraction.text

    #
    # Classification principale
    #
    classification = (
        OllamaService.classify(
            extraction.text
        )
    )

    assert (
        classification.document_type
        == expected["document_type"]
    )

    assert (
        classification.confidence
        >= expected.get(
            "min_confidence",
            0.0
        )
    )

    #
    # Sous-classification
    #
    sub_classification = None

    if (
        "sub_type" in expected
        and OllamaService.load_classifier(
            classification.document_type
        )
    ):

        sub_classification = (
            OllamaService.sub_classify(
                extraction.text,
                classification.document_type
            )
        )

        print(
            f"{pdf_path.name} "
            f"-> {classification.document_type} "
            f"({classification.confidence})"
            f"-> {sub_classification.sub_type} "
            f"({sub_classification.sub_confidence})"
        )

        assert (
            sub_classification.sub_type
            == expected["sub_type"]
        )

        assert (
            sub_classification.sub_confidence
            >= expected.get(
                "min_sub_confidence",
                0.0
            )
        )
    else:
        print(
            f"{pdf_path.name} "
            f"-> {classification.document_type} "
            f"({classification.confidence})"
        )
    #
    # Construction du nom du schéma
    #
    document_schema = (
        classification.document_type
    )

    if sub_classification:

        document_schema = (
            f"{document_schema}_"
            f"{sub_classification.sub_type}"
        )

    #
    # Extraction des données
    #
    data_result = (
        OllamaService.extract_data(
            extraction.text,
            document_schema
        )
    )

    #
    # Vérification des champs attendus
    #
    for field in expected.get(
        "required_fields",
        []
    ):

        assert (
            field
            in data_result.data
        ), (
            f"Champ absent : "
            f"{field}"
        )

        assert (
            data_result
            .data[field]
            .value
        ), (
            f"Champ vide : "
            f"{field}"
        )

    #
    # Vérification des valeurs attendues
    #
    for field, expected_value in (
        expected.get(
            "expected_values",
            {}
        ).items()
    ):

        assert (
            data_result
            .data[field]
            .value
            == expected_value
        ), (
            f"Valeur inattendue "
            f"pour '{field}'"
        )

    #
    # Vérification des niveaux de confiance
    #
    for field, min_confidence in (
        expected.get(
            "field_confidence",
            {}
        ).items()
    ):

        assert (
            data_result
            .data[field]
            .confidence
            >= min_confidence
        ), (
            f"Confiance trop faible "
            f"pour '{field}'"
        )