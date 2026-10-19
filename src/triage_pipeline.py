"""
Unified Multimodal Clinical Triage Pipeline
Integrates DenseNet-121 Vision Classifier and Llama 3 8B Reasoner.
"""

import os
import gc
from typing import Dict, Any, Optional
from .vision_model import DenseNetTriageClassifier
from .reasoner import ClinicalReasoner


class MultimodalTriagePipeline:
    """Unified pipeline orchestrating vision inference, memory management, and LLM triage report generation."""

    def __init__(
        self,
        vision_weights_path: str = "densenet121_chexpert_sota.pth",
        llm_model_path: str = "triage_llama3_lora",
        base_model_name: str = "unsloth/llama-3-8b-Instruct-bnb-4bit",
        device: str = "auto",
        single_gpu: bool = False
    ):
        try:
            import torch
            self.torch = torch
            has_cuda = torch.cuda.is_available()
            num_gpus = torch.cuda.device_count() if has_cuda else 0
        except ImportError:
            self.torch = None
            has_cuda = False
            num_gpus = 0

        # Enable expandable segments to eliminate virtual memory fragmentation deadlocks
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if has_cuda and self.torch is not None:
            self.torch.cuda.empty_cache()
        gc.collect()

        # Multi-GPU / Single-GPU / CPU device distribution
        if device != "auto":
            self.vision_device = device
            self.llm_device = device
        elif num_gpus > 1 and not single_gpu:
            self.vision_device = "cuda:1"
            self.llm_device = "cuda:0"
        elif num_gpus == 1 or single_gpu:
            self.vision_device = "cuda:0" if num_gpus >= 1 else "cpu"
            self.llm_device = "cuda:0" if num_gpus >= 1 else "cpu"
        else:
            self.vision_device = "cpu"
            self.llm_device = "cpu"

        print(f"[Pipeline] Hardware allocation -> Vision: {self.vision_device} | Reasoner: {self.llm_device}")

        # Initialize Vision Layer
        self.vision_classifier = DenseNetTriageClassifier(
            weights_path=vision_weights_path,
            device=self.vision_device
        )

        # Initialize Reasoning Layer
        self.clinical_reasoner = ClinicalReasoner(
            model_path_or_name=llm_model_path,
            base_model_name=base_model_name,
            device_map=self.llm_device
        )

    def run_triage(self, image_path: str, clinical_notes: str, clinical_question: str) -> Dict[str, Any]:
        """Executes the complete multimodal triage evaluation on a patient case."""
        print("\n" + "=" * 55)
        print("STEP 1: VISION CLASSIFICATION (DenseNet-121)")
        print("=" * 55)
        detected_pathologies, all_probs = self.vision_classifier.predict_image(image_path)
        findings_str = ", ".join(detected_pathologies) if detected_pathologies else "No acute abnormality detected"
        print(f"Detected Pathologies: {findings_str}")

        # Clean memory before heavy LLM generation
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        gc.collect()

        print("\n" + "=" * 55)
        print("STEP 2: CLINICAL LLM SYNTHESIS (Llama 3 8B)")
        print("=" * 55)
        report = self.clinical_reasoner.generate_triage_report(
            visual_detections=detected_pathologies,
            clinical_notes=clinical_notes,
            clinical_question=clinical_question
        )

        print("\n" + "=" * 55)
        print("           CLINICAL TRIAGE REPORT")
        print("=" * 55)
        print(report)
        print("=" * 55)

        return {
            "image_path": image_path,
            "detected_pathologies": detected_pathologies,
            "pathology_probabilities": all_probs,
            "clinical_notes": clinical_notes,
            "clinical_question": clinical_question,
            "clinical_report": report
        }
