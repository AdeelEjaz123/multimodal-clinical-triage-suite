"""
Multimodal Clinical Triage Suite (MCTS)
Emergency Department AI Pipeline: DenseNet-121 + Llama 3 8B
"""

__version__ = "1.0.0"
__author__ = "Adeel Ejaz"

from .vision_model import DenseNetTriageClassifier, CHEXPERT_LABELS, OPTIMAL_THRESHOLDS
from .reasoner import ClinicalReasoner
from .triage_pipeline import MultimodalTriagePipeline

__all__ = [
    "DenseNetTriageClassifier",
    "CHEXPERT_LABELS",
    "OPTIMAL_THRESHOLDS",
    "ClinicalReasoner",
    "MultimodalTriagePipeline",
]
