"""
clinical_cases.py — Curated clinical vignette dataset for SympathyAI evaluation.

Each case contains:
  - A clinical presentation (the prompt given to the LLM).
  - A reference good response (expert-aligned).
  - A reference poor response (plausible-sounding but flawed).
  - Guideline keywords required for correct reasoning.
  - Symptom and diagnosis term lists for causality scoring.

This dataset is intended for:
  1. Demonstrating the SympathyAI re-ranking pipeline.
  2. Validating that sympathetic scores correlate with expert judgment.
  3. Benchmarking future improvements to the scoring functions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class ClinicalCase:
    case_id: str
    presentation: str
    keywords: List[str]
    symptoms: List[str]
    diagnosis_terms: List[str]
    good_response: str
    poor_response: str
    condition: str = ""
    source: str = "SympathyAI synthetic benchmark v0.1"


CLINICAL_CASES: List[ClinicalCase] = [
    ClinicalCase(
        case_id="MDD_001",
        condition="Major Depressive Disorder",
        presentation=(
            "A 25-year-old presents with persistent sadness, insomnia, "
            "loss of interest in daily activities, and fatigue lasting 3 weeks."
        ),
        keywords=["depression", "DSM", "criteria", "anhedonia", "two weeks"],
        symptoms=["sadness", "insomnia", "loss of interest", "fatigue"],
        diagnosis_terms=["major depressive disorder", "MDD", "depression"],
        good_response=(
            "The presentation is consistent with major depressive disorder. "
            "The patient meets DSM-5 criteria: depressed mood, anhedonia, "
            "insomnia, and fatigue persisting for more than two weeks with "
            "functional impairment. A PHQ-9 assessment and referral to "
            "psychiatry are indicated."
        ),
        poor_response=(
            "This might be stress or a difficult life period. The diagnosis "
            "is unclear — it could be many things. Perhaps some rest would help."
        ),
    ),
    ClinicalCase(
        case_id="MI_001",
        condition="Myocardial Infarction",
        presentation=(
            "A 58-year-old male presents with sudden chest pain radiating to "
            "the left arm, diaphoresis, and nausea for the past 45 minutes."
        ),
        keywords=["myocardial infarction", "ECG", "troponin", "STEMI", "cardiac"],
        symptoms=["chest pain", "radiation to left arm", "diaphoresis", "nausea"],
        diagnosis_terms=["myocardial infarction", "MI", "STEMI", "NSTEMI", "ACS"],
        good_response=(
            "This presentation is highly concerning for acute myocardial infarction. "
            "Immediate 12-lead ECG and troponin levels are indicated. Aspirin should "
            "be administered, and the patient should be prepared for potential "
            "percutaneous coronary intervention if STEMI is confirmed."
        ),
        poor_response=(
            "The chest pain could be anxiety or musculoskeletal in origin. "
            "It is unclear what is causing the symptoms. The patient should "
            "rest and return if symptoms worsen."
        ),
    ),
    ClinicalCase(
        case_id="PNE_001",
        condition="Community-Acquired Pneumonia",
        presentation=(
            "A 40-year-old presents with fever (38.9°C), productive cough, "
            "shortness of breath, and right-sided pleuritic chest pain for 4 days."
        ),
        keywords=["pneumonia", "antibiotics", "chest X-ray", "consolidation", "infection"],
        symptoms=["fever", "cough", "shortness of breath", "pleuritic chest pain"],
        diagnosis_terms=["pneumonia", "community-acquired pneumonia", "CAP"],
        good_response=(
            "The clinical picture is consistent with community-acquired pneumonia. "
            "A chest X-ray should be obtained to look for consolidation. "
            "Empirical antibiotics (e.g., amoxicillin-clavulanate) are indicated, "
            "with severity assessed using CRB-65 to guide inpatient vs. outpatient management."
        ),
        poor_response=(
            "This is likely nothing serious — probably a viral cold. "
            "The patient should drink fluids and take paracetamol. "
            "There is no need for further investigation at this stage."
        ),
    ),
    ClinicalCase(
        case_id="GAD_001",
        condition="Generalised Anxiety Disorder",
        presentation=(
            "A 32-year-old reports excessive worry about work and family, "
            "muscle tension, difficulty concentrating, and sleep disturbance "
            "persisting for 8 months despite no identifiable stressor."
        ),
        keywords=["anxiety", "GAD", "DSM", "worry", "six months", "impairment"],
        symptoms=["worry", "muscle tension", "concentration difficulty", "insomnia"],
        diagnosis_terms=["generalised anxiety disorder", "GAD", "anxiety disorder"],
        good_response=(
            "This presentation meets DSM-5 criteria for Generalised Anxiety Disorder: "
            "excessive, difficult-to-control worry occurring more days than not for "
            "at least six months, with multiple somatic symptoms and functional impairment. "
            "CBT and/or an SSRI/SNRI are first-line treatments."
        ),
        poor_response=(
            "The patient seems stressed about life. This is normal and may resolve "
            "on its own. It is unclear whether this requires formal treatment."
        ),
    ),
    ClinicalCase(
        case_id="DM2_001",
        condition="Type 2 Diabetes Mellitus",
        presentation=(
            "A 52-year-old with obesity and a family history of diabetes presents "
            "with polyuria, polydipsia, blurred vision, and fatigue. "
            "Fasting glucose is 8.2 mmol/L on two occasions."
        ),
        keywords=["diabetes", "HbA1c", "glucose", "metformin", "lifestyle"],
        symptoms=["polyuria", "polydipsia", "blurred vision", "fatigue"],
        diagnosis_terms=["type 2 diabetes", "diabetes mellitus", "T2DM"],
        good_response=(
            "Two fasting glucose readings above 7.0 mmol/L confirm a diagnosis of "
            "type 2 diabetes mellitus. HbA1c should be obtained to assess glycaemic "
            "control. First-line management includes lifestyle modification and "
            "initiation of metformin, with referral to a diabetes care team."
        ),
        poor_response=(
            "The elevated glucose might be due to a large meal or stress. "
            "The diagnosis is uncertain — we should just monitor the patient "
            "and see if glucose normalises on its own."
        ),
    ),
]


def load_cases() -> List[ClinicalCase]:
    """Return all curated clinical cases."""
    return CLINICAL_CASES


def get_case(case_id: str) -> ClinicalCase:
    """Retrieve a single case by ID."""
    for case in CLINICAL_CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(f"Case '{case_id}' not found.")
