"""
CLI Demo for Multimodal Clinical Triage Suite
Usage:
    python inference_demo.py \
        --image path/to/xray.jpg \
        --notes "68-year-old female with acute orthopnea" \
        --vision-weights densenet121_chexpert_sota.pth \
        --llm-model triage_llama3_lora
"""

import os
import json
import argparse
from src.triage_pipeline import MultimodalTriagePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multimodal Clinical Triage Suite (MCTS) Inference CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image",
        type=str,
        default="sample_xray_demo.jpg",
        help="Path to input chest radiograph image (JPEG/PNG/DICOM-exported)"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="68-year-old female presenting to emergency room with progressive shortness of breath, bilateral lower extremity swelling, and orthopnea over 4 days.",
        help="Raw EHR physician clinical notes or path to a .txt file containing notes"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What is the primary diagnostic consideration, and does the radiological assessment suggest pulmonary decompensation requiring admission?",
        help="Clinical triage query"
    )
    parser.add_argument(
        "--vision-weights",
        type=str,
        default="densenet121_chexpert_sota.pth",
        help="Path to DenseNet-121 weights checkpoint (.pth file)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="triage_llama3_lora",
        help="Path to fine-tuned Llama 3 directory / LoRA adapter folder"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/llama-3-8b-Instruct-bnb-4bit",
        help="Base Llama 3 model name or path (for HF PEFT loading)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "cuda:0", "cuda:1"],
        help="Hardware execution device"
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Force single-GPU execution mode"
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to export triage results as a JSON file"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  🏥 Multimodal Clinical Triage Suite (MCTS) — Inference CLI")
    print("=" * 60)
    print(f"• Vision Weights Path : {args.vision_weights}")
    print(f"• LLM Model Path      : {args.llm_model}")
    print(f"• Target Device       : {args.device}")

    # Check for clinical notes input from file if applicable
    clinical_notes = args.notes
    if os.path.exists(clinical_notes) and os.path.isfile(clinical_notes):
        with open(clinical_notes, "r", encoding="utf-8") as f:
            clinical_notes = f.read().strip()

    # Verify or generate sample image if not provided
    sample_img_path = args.image
    if not os.path.exists(sample_img_path):
        from PIL import Image
        import numpy as np
        print(f"[Notice] Specified image not found at '{sample_img_path}'. Creating synthetic demo radiograph...")
        sample_img_path = "sample_xray_demo.jpg"
        arr = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(sample_img_path)

    # Initialize Multimodal Pipeline
    pipeline = MultimodalTriagePipeline(
        vision_weights_path=args.vision_weights,
        llm_model_path=args.llm_model,
        base_model_name=args.base_model,
        device=args.device,
        single_gpu=args.single_gpu
    )

    # Run inference
    result = pipeline.run_triage(
        image_path=sample_img_path,
        clinical_notes=clinical_notes,
        clinical_question=args.question
    )

    # Export results to JSON if requested
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[Export] Full triage result successfully saved to: '{args.output_json}'")

    print("\n[MCTS Pipeline Execution Finished]")


if __name__ == "__main__":
    main()
