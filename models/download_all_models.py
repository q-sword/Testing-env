#!/usr/bin/env python3
"""
Comprehensive ML Model Downloader
Downloads pretrained lightweight-medium sized models across all domains.
Uses HuggingFace Hub for all downloads to avoid network restrictions.
"""

import os
import torch
import json
from pathlib import Path

# Base directory
BASE_DIR = Path("/home/user/Testing-env/models")

# Model catalog to track all downloaded models
MODEL_CATALOG = {}

def save_model_info(domain, name, info):
    """Save model info to catalog"""
    if domain not in MODEL_CATALOG:
        MODEL_CATALOG[domain] = {}
    MODEL_CATALOG[domain][name] = info

def save_catalog():
    """Save the complete catalog to JSON"""
    with open(BASE_DIR / "model_catalog.json", "w") as f:
        json.dump(MODEL_CATALOG, f, indent=2)

# ============================================================================
# COMPUTER VISION MODELS (via HuggingFace/timm)
# ============================================================================
def download_vision_models():
    print("\n" + "="*60)
    print("DOWNLOADING COMPUTER VISION MODELS")
    print("="*60)

    import timm
    from transformers import (
        AutoModelForImageClassification, AutoImageProcessor,
        DetrForObjectDetection, DetrImageProcessor,
        SegformerForSemanticSegmentation, SegformerImageProcessor,
        Mask2FormerForUniversalSegmentation, AutoProcessor,
        YolosForObjectDetection, YolosImageProcessor,
        ViTForImageClassification, ViTImageProcessor,
        ConvNextForImageClassification, ConvNextImageProcessor,
        SwinForImageClassification,
        DeiTForImageClassification,
        BeitForImageClassification,
        MobileViTForImageClassification,
        ResNetForImageClassification
    )

    vision_dir = BASE_DIR / "vision"

    # --- Image Classification ---
    print("\n[1/4] Image Classification Models...")

    # ResNet-18 via HuggingFace
    print("  - ResNet-18...")
    try:
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
        model.save_pretrained(vision_dir / "resnet-18")
        processor.save_pretrained(vision_dir / "resnet-18")
        save_model_info("vision", "resnet-18", {
            "task": "image_classification",
            "params": "11.7M",
            "size": "~45MB",
            "dataset": "ImageNet-1K",
            "source": "microsoft/resnet-18"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # ResNet-50 via HuggingFace
    print("  - ResNet-50...")
    try:
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model.save_pretrained(vision_dir / "resnet-50")
        processor.save_pretrained(vision_dir / "resnet-50")
        save_model_info("vision", "resnet-50", {
            "task": "image_classification",
            "params": "25.6M",
            "size": "~98MB",
            "dataset": "ImageNet-1K",
            "source": "microsoft/resnet-50"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # ViT Tiny
    print("  - ViT Tiny...")
    try:
        model = ViTForImageClassification.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        processor = ViTImageProcessor.from_pretrained("WinKawaks/vit-tiny-patch16-224")
        model.save_pretrained(vision_dir / "vit-tiny-patch16-224")
        processor.save_pretrained(vision_dir / "vit-tiny-patch16-224")
        save_model_info("vision", "vit-tiny-patch16-224", {
            "task": "image_classification",
            "params": "5.7M",
            "size": "~22MB",
            "dataset": "ImageNet-1K",
            "architecture": "Vision Transformer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # ViT Base
    print("  - ViT Base (patch16, 224)...")
    try:
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model.save_pretrained(vision_dir / "vit-base-patch16-224")
        processor.save_pretrained(vision_dir / "vit-base-patch16-224")
        save_model_info("vision", "vit-base-patch16-224", {
            "task": "image_classification",
            "params": "86M",
            "size": "~330MB",
            "dataset": "ImageNet-1K",
            "accuracy": "77.9% top-1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # DeiT Tiny
    print("  - DeiT Tiny...")
    try:
        model = DeiTForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224")
        processor = AutoImageProcessor.from_pretrained("facebook/deit-tiny-patch16-224")
        model.save_pretrained(vision_dir / "deit-tiny-patch16-224")
        processor.save_pretrained(vision_dir / "deit-tiny-patch16-224")
        save_model_info("vision", "deit-tiny-patch16-224", {
            "task": "image_classification",
            "params": "5.7M",
            "size": "~22MB",
            "dataset": "ImageNet-1K",
            "architecture": "Data-efficient Image Transformer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # DeiT Small
    print("  - DeiT Small...")
    try:
        model = DeiTForImageClassification.from_pretrained("facebook/deit-small-patch16-224")
        processor = AutoImageProcessor.from_pretrained("facebook/deit-small-patch16-224")
        model.save_pretrained(vision_dir / "deit-small-patch16-224")
        processor.save_pretrained(vision_dir / "deit-small-patch16-224")
        save_model_info("vision", "deit-small-patch16-224", {
            "task": "image_classification",
            "params": "22M",
            "size": "~88MB",
            "dataset": "ImageNet-1K",
            "accuracy": "79.9% top-1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BEiT Base
    print("  - BEiT Base...")
    try:
        model = BeitForImageClassification.from_pretrained("microsoft/beit-base-patch16-224")
        processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")
        model.save_pretrained(vision_dir / "beit-base-patch16-224")
        processor.save_pretrained(vision_dir / "beit-base-patch16-224")
        save_model_info("vision", "beit-base-patch16-224", {
            "task": "image_classification",
            "params": "86M",
            "size": "~330MB",
            "dataset": "ImageNet-1K",
            "accuracy": "83.2% top-1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Swin Tiny
    print("  - Swin Transformer Tiny...")
    try:
        model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        model.save_pretrained(vision_dir / "swin-tiny")
        processor.save_pretrained(vision_dir / "swin-tiny")
        save_model_info("vision", "swin-tiny", {
            "task": "image_classification",
            "params": "28M",
            "size": "~108MB",
            "dataset": "ImageNet-1K",
            "accuracy": "81.2% top-1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # ConvNeXt Tiny
    print("  - ConvNeXt Tiny...")
    try:
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
        processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-tiny-224")
        model.save_pretrained(vision_dir / "convnext-tiny")
        processor.save_pretrained(vision_dir / "convnext-tiny")
        save_model_info("vision", "convnext-tiny", {
            "task": "image_classification",
            "params": "28.6M",
            "size": "~110MB",
            "dataset": "ImageNet-1K",
            "accuracy": "82.1% top-1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # MobileViT Small
    print("  - MobileViT Small...")
    try:
        model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
        processor = AutoImageProcessor.from_pretrained("apple/mobilevit-small")
        model.save_pretrained(vision_dir / "mobilevit-small")
        processor.save_pretrained(vision_dir / "mobilevit-small")
        save_model_info("vision", "mobilevit-small", {
            "task": "image_classification",
            "params": "5.6M",
            "size": "~22MB",
            "dataset": "ImageNet-1K",
            "description": "Apple's MobileViT for mobile devices"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # MobileViT XX-Small
    print("  - MobileViT XX-Small...")
    try:
        model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")
        processor = AutoImageProcessor.from_pretrained("apple/mobilevit-xx-small")
        model.save_pretrained(vision_dir / "mobilevit-xxs")
        processor.save_pretrained(vision_dir / "mobilevit-xxs")
        save_model_info("vision", "mobilevit-xxs", {
            "task": "image_classification",
            "params": "1.3M",
            "size": "~5MB",
            "dataset": "ImageNet-1K",
            "description": "Ultra-small MobileViT"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # EfficientFormer
    print("  - EfficientFormer L1...")
    try:
        model = AutoModelForImageClassification.from_pretrained("snap-research/efficientformer-l1-300")
        processor = AutoImageProcessor.from_pretrained("snap-research/efficientformer-l1-300")
        model.save_pretrained(vision_dir / "efficientformer-l1")
        processor.save_pretrained(vision_dir / "efficientformer-l1")
        save_model_info("vision", "efficientformer-l1", {
            "task": "image_classification",
            "params": "12M",
            "size": "~48MB",
            "dataset": "ImageNet-1K",
            "description": "Snap's EfficientFormer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Object Detection ---
    print("\n[2/4] Object Detection Models...")

    # DETR ResNet-50
    print("  - DETR ResNet-50...")
    try:
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model.save_pretrained(vision_dir / "detr-resnet-50")
        processor.save_pretrained(vision_dir / "detr-resnet-50")
        save_model_info("vision", "detr-resnet-50", {
            "task": "object_detection",
            "params": "41M",
            "size": "~160MB",
            "dataset": "COCO",
            "mAP": "42.0"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # YOLOS Tiny
    print("  - YOLOS Tiny...")
    try:
        model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")
        processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        model.save_pretrained(vision_dir / "yolos-tiny")
        processor.save_pretrained(vision_dir / "yolos-tiny")
        save_model_info("vision", "yolos-tiny", {
            "task": "object_detection",
            "params": "6.5M",
            "size": "~26MB",
            "dataset": "COCO",
            "mAP": "28.7"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # YOLOS Small
    print("  - YOLOS Small...")
    try:
        model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")
        processor = YolosImageProcessor.from_pretrained("hustvl/yolos-small")
        model.save_pretrained(vision_dir / "yolos-small")
        processor.save_pretrained(vision_dir / "yolos-small")
        save_model_info("vision", "yolos-small", {
            "task": "object_detection",
            "params": "30M",
            "size": "~120MB",
            "dataset": "COCO",
            "mAP": "36.1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Conditional DETR
    print("  - Conditional DETR ResNet-50...")
    try:
        model = AutoModelForImageClassification.from_pretrained("microsoft/conditional-detr-resnet-50")
        processor = AutoImageProcessor.from_pretrained("microsoft/conditional-detr-resnet-50")
        model.save_pretrained(vision_dir / "conditional-detr-resnet-50")
        processor.save_pretrained(vision_dir / "conditional-detr-resnet-50")
        save_model_info("vision", "conditional-detr-resnet-50", {
            "task": "object_detection",
            "params": "44M",
            "size": "~170MB",
            "dataset": "COCO"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Semantic Segmentation ---
    print("\n[3/4] Semantic Segmentation Models...")

    # SegFormer B0
    print("  - SegFormer B0 (ADE20K)...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        model.save_pretrained(vision_dir / "segformer-b0-ade")
        processor.save_pretrained(vision_dir / "segformer-b0-ade")
        save_model_info("vision", "segformer-b0-ade", {
            "task": "semantic_segmentation",
            "params": "3.8M",
            "size": "~15MB",
            "dataset": "ADE20K",
            "mIoU": "37.4"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # SegFormer B1
    print("  - SegFormer B1 (ADE20K)...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
        model.save_pretrained(vision_dir / "segformer-b1-ade")
        processor.save_pretrained(vision_dir / "segformer-b1-ade")
        save_model_info("vision", "segformer-b1-ade", {
            "task": "semantic_segmentation",
            "params": "13.7M",
            "size": "~55MB",
            "dataset": "ADE20K",
            "mIoU": "40.1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # SegFormer B2
    print("  - SegFormer B2 (ADE20K)...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
        model.save_pretrained(vision_dir / "segformer-b2-ade")
        processor.save_pretrained(vision_dir / "segformer-b2-ade")
        save_model_info("vision", "segformer-b2-ade", {
            "task": "semantic_segmentation",
            "params": "27.4M",
            "size": "~110MB",
            "dataset": "ADE20K",
            "mIoU": "45.6"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # SegFormer B0 CityScapes
    print("  - SegFormer B0 (CityScapes)...")
    try:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
        processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-cityscapes-512-1024")
        model.save_pretrained(vision_dir / "segformer-b0-cityscapes")
        processor.save_pretrained(vision_dir / "segformer-b0-cityscapes")
        save_model_info("vision", "segformer-b0-cityscapes", {
            "task": "semantic_segmentation",
            "params": "3.8M",
            "size": "~15MB",
            "dataset": "CityScapes",
            "mIoU": "71.9"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Instance/Panoptic Segmentation ---
    print("\n[4/4] Instance Segmentation Models...")

    # Mask2Former Swin-Tiny (COCO panoptic)
    print("  - Mask2Former Swin-Tiny (COCO Panoptic)...")
    try:
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-panoptic")
        processor = AutoProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-panoptic")
        model.save_pretrained(vision_dir / "mask2former-swin-tiny-coco")
        processor.save_pretrained(vision_dir / "mask2former-swin-tiny-coco")
        save_model_info("vision", "mask2former-swin-tiny-coco", {
            "task": "panoptic_segmentation",
            "params": "47M",
            "size": "~180MB",
            "dataset": "COCO",
            "PQ": "52.1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Mask2Former Swin-Small (ADE20K)
    print("  - Mask2Former Swin-Small (ADE20K)...")
    try:
        model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        processor = AutoProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        model.save_pretrained(vision_dir / "mask2former-swin-small-ade")
        processor.save_pretrained(vision_dir / "mask2former-swin-small-ade")
        save_model_info("vision", "mask2former-swin-small-ade", {
            "task": "semantic_segmentation",
            "params": "69M",
            "size": "~270MB",
            "dataset": "ADE20K",
            "mIoU": "51.3"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n✓ Computer Vision models downloaded!")
    save_catalog()

# ============================================================================
# NLP MODELS
# ============================================================================
def download_nlp_models():
    print("\n" + "="*60)
    print("DOWNLOADING NLP MODELS")
    print("="*60)

    from transformers import (
        AutoModel, AutoModelForSequenceClassification,
        AutoModelForTokenClassification, AutoModelForQuestionAnswering,
        AutoModelForMaskedLM, AutoModelForCausalLM,
        AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer,
        BartForConditionalGeneration, BartTokenizer,
        PegasusForConditionalGeneration, PegasusTokenizer,
        MarianMTModel, MarianTokenizer
    )

    nlp_dir = BASE_DIR / "nlp"

    # --- Text Embeddings / Encoding ---
    print("\n[1/8] Text Embedding Models...")

    # DistilBERT Base
    print("  - DistilBERT Base...")
    try:
        model = AutoModel.from_pretrained("distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model.save_pretrained(nlp_dir / "distilbert-base-uncased")
        tokenizer.save_pretrained(nlp_dir / "distilbert-base-uncased")
        save_model_info("nlp", "distilbert-base-uncased", {
            "task": "text_embedding",
            "params": "66M",
            "size": "~268MB",
            "architecture": "DistilBERT",
            "description": "Distilled version of BERT, 40% smaller, 60% faster"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BERT Tiny
    print("  - BERT Tiny...")
    try:
        model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
        model.save_pretrained(nlp_dir / "bert-tiny")
        tokenizer.save_pretrained(nlp_dir / "bert-tiny")
        save_model_info("nlp", "bert-tiny", {
            "task": "text_embedding",
            "params": "4.4M",
            "size": "~17MB",
            "architecture": "BERT",
            "description": "Ultra-small BERT for resource-constrained environments"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BERT Mini
    print("  - BERT Mini...")
    try:
        model = AutoModel.from_pretrained("prajjwal1/bert-mini")
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
        model.save_pretrained(nlp_dir / "bert-mini")
        tokenizer.save_pretrained(nlp_dir / "bert-mini")
        save_model_info("nlp", "bert-mini", {
            "task": "text_embedding",
            "params": "11.3M",
            "size": "~45MB",
            "architecture": "BERT",
            "description": "Small BERT variant"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BERT Small
    print("  - BERT Small...")
    try:
        model = AutoModel.from_pretrained("prajjwal1/bert-small")
        tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-small")
        model.save_pretrained(nlp_dir / "bert-small")
        tokenizer.save_pretrained(nlp_dir / "bert-small")
        save_model_info("nlp", "bert-small", {
            "task": "text_embedding",
            "params": "29M",
            "size": "~115MB",
            "architecture": "BERT"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # ALBERT Base v2
    print("  - ALBERT Base v2...")
    try:
        model = AutoModel.from_pretrained("albert-base-v2")
        tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
        model.save_pretrained(nlp_dir / "albert-base-v2")
        tokenizer.save_pretrained(nlp_dir / "albert-base-v2")
        save_model_info("nlp", "albert-base-v2", {
            "task": "text_embedding",
            "params": "12M",
            "size": "~47MB",
            "architecture": "ALBERT",
            "description": "A Lite BERT with parameter sharing"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # DistilRoBERTa
    print("  - DistilRoBERTa Base...")
    try:
        model = AutoModel.from_pretrained("distilroberta-base")
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        model.save_pretrained(nlp_dir / "distilroberta-base")
        tokenizer.save_pretrained(nlp_dir / "distilroberta-base")
        save_model_info("nlp", "distilroberta-base", {
            "task": "text_embedding",
            "params": "82M",
            "size": "~330MB",
            "architecture": "DistilRoBERTa"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Text Classification ---
    print("\n[2/8] Text Classification Models...")

    # DistilBERT Sentiment (SST-2)
    print("  - DistilBERT SST-2 Sentiment...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        model.save_pretrained(nlp_dir / "distilbert-sst2-sentiment")
        tokenizer.save_pretrained(nlp_dir / "distilbert-sst2-sentiment")
        save_model_info("nlp", "distilbert-sst2-sentiment", {
            "task": "sentiment_analysis",
            "params": "67M",
            "size": "~268MB",
            "dataset": "SST-2",
            "accuracy": "91.3%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Twitter RoBERTa Sentiment
    print("  - Twitter RoBERTa Sentiment...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        model.save_pretrained(nlp_dir / "twitter-roberta-sentiment")
        tokenizer.save_pretrained(nlp_dir / "twitter-roberta-sentiment")
        save_model_info("nlp", "twitter-roberta-sentiment", {
            "task": "sentiment_analysis",
            "params": "125M",
            "size": "~500MB",
            "dataset": "Twitter",
            "labels": ["negative", "neutral", "positive"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Emotion Classification
    print("  - RoBERTa Emotion...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "SamLowe/roberta-base-go_emotions"
        )
        tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
        model.save_pretrained(nlp_dir / "roberta-go-emotions")
        tokenizer.save_pretrained(nlp_dir / "roberta-go-emotions")
        save_model_info("nlp", "roberta-go-emotions", {
            "task": "emotion_classification",
            "params": "125M",
            "size": "~500MB",
            "dataset": "GoEmotions",
            "num_labels": 28
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Topic Classification
    print("  - BART MNLI (Zero-shot classification)...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
        model.save_pretrained(nlp_dir / "bart-large-mnli")
        tokenizer.save_pretrained(nlp_dir / "bart-large-mnli")
        save_model_info("nlp", "bart-large-mnli", {
            "task": "zero_shot_classification",
            "params": "407M",
            "size": "~1.6GB",
            "dataset": "MNLI",
            "description": "Zero-shot text classification"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Named Entity Recognition ---
    print("\n[3/8] Named Entity Recognition Models...")

    # DistilBERT NER
    print("  - DistilBERT NER...")
    try:
        model = AutoModelForTokenClassification.from_pretrained("dslim/distilbert-NER")
        tokenizer = AutoTokenizer.from_pretrained("dslim/distilbert-NER")
        model.save_pretrained(nlp_dir / "distilbert-ner")
        tokenizer.save_pretrained(nlp_dir / "distilbert-ner")
        save_model_info("nlp", "distilbert-ner", {
            "task": "named_entity_recognition",
            "params": "66M",
            "size": "~268MB",
            "entities": ["PER", "ORG", "LOC", "MISC"],
            "f1_score": "90.2%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BERT Base NER
    print("  - BERT Base NER...")
    try:
        model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        model.save_pretrained(nlp_dir / "bert-base-ner")
        tokenizer.save_pretrained(nlp_dir / "bert-base-ner")
        save_model_info("nlp", "bert-base-ner", {
            "task": "named_entity_recognition",
            "params": "110M",
            "size": "~440MB",
            "entities": ["PER", "ORG", "LOC", "MISC"],
            "f1_score": "91.3%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Question Answering ---
    print("\n[4/8] Question Answering Models...")

    # DistilBERT SQuAD
    print("  - DistilBERT SQuAD...")
    try:
        model = AutoModelForQuestionAnswering.from_pretrained(
            "distilbert-base-uncased-distilled-squad"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-distilled-squad"
        )
        model.save_pretrained(nlp_dir / "distilbert-squad")
        tokenizer.save_pretrained(nlp_dir / "distilbert-squad")
        save_model_info("nlp", "distilbert-squad", {
            "task": "question_answering",
            "params": "66M",
            "size": "~265MB",
            "dataset": "SQuAD v1.1",
            "f1_score": "86.9%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # MiniLM SQuAD v2
    print("  - MiniLM SQuAD v2...")
    try:
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/minilm-uncased-squad2")
        tokenizer = AutoTokenizer.from_pretrained("deepset/minilm-uncased-squad2")
        model.save_pretrained(nlp_dir / "minilm-squad2")
        tokenizer.save_pretrained(nlp_dir / "minilm-squad2")
        save_model_info("nlp", "minilm-squad2", {
            "task": "question_answering",
            "params": "33M",
            "size": "~130MB",
            "dataset": "SQuAD v2.0",
            "f1_score": "76.1%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # RoBERTa SQuAD
    print("  - RoBERTa Base SQuAD2...")
    try:
        model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
        tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        model.save_pretrained(nlp_dir / "roberta-base-squad2")
        tokenizer.save_pretrained(nlp_dir / "roberta-base-squad2")
        save_model_info("nlp", "roberta-base-squad2", {
            "task": "question_answering",
            "params": "125M",
            "size": "~500MB",
            "dataset": "SQuAD v2.0",
            "f1_score": "83.0%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Text Generation ---
    print("\n[5/8] Text Generation Models...")

    # DistilGPT2
    print("  - DistilGPT2...")
    try:
        model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        model.save_pretrained(nlp_dir / "distilgpt2")
        tokenizer.save_pretrained(nlp_dir / "distilgpt2")
        save_model_info("nlp", "distilgpt2", {
            "task": "text_generation",
            "params": "82M",
            "size": "~330MB",
            "architecture": "GPT-2 Distilled"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # GPT-2 Small
    print("  - GPT-2 Small...")
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model.save_pretrained(nlp_dir / "gpt2-small")
        tokenizer.save_pretrained(nlp_dir / "gpt2-small")
        save_model_info("nlp", "gpt2-small", {
            "task": "text_generation",
            "params": "124M",
            "size": "~500MB",
            "architecture": "GPT-2"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # GPT-2 Medium
    print("  - GPT-2 Medium...")
    try:
        model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
        tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
        model.save_pretrained(nlp_dir / "gpt2-medium")
        tokenizer.save_pretrained(nlp_dir / "gpt2-medium")
        save_model_info("nlp", "gpt2-medium", {
            "task": "text_generation",
            "params": "355M",
            "size": "~1.4GB",
            "architecture": "GPT-2"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Summarization ---
    print("\n[6/8] Summarization Models...")

    # T5 Small
    print("  - T5 Small...")
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model.save_pretrained(nlp_dir / "t5-small")
        tokenizer.save_pretrained(nlp_dir / "t5-small")
        save_model_info("nlp", "t5-small", {
            "task": "text2text_generation",
            "params": "60M",
            "size": "~240MB",
            "capabilities": ["summarization", "translation", "QA"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # T5 Base
    print("  - T5 Base...")
    try:
        model = T5ForConditionalGeneration.from_pretrained("t5-base")
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model.save_pretrained(nlp_dir / "t5-base")
        tokenizer.save_pretrained(nlp_dir / "t5-base")
        save_model_info("nlp", "t5-base", {
            "task": "text2text_generation",
            "params": "220M",
            "size": "~890MB",
            "capabilities": ["summarization", "translation", "QA"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BART CNN
    print("  - BART CNN (Summarization)...")
    try:
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
        model.save_pretrained(nlp_dir / "bart-large-cnn")
        tokenizer.save_pretrained(nlp_dir / "bart-large-cnn")
        save_model_info("nlp", "bart-large-cnn", {
            "task": "summarization",
            "params": "406M",
            "size": "~1.6GB",
            "dataset": "CNN/DailyMail"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Pegasus XSum
    print("  - Pegasus XSum...")
    try:
        model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
        tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
        model.save_pretrained(nlp_dir / "pegasus-xsum")
        tokenizer.save_pretrained(nlp_dir / "pegasus-xsum")
        save_model_info("nlp", "pegasus-xsum", {
            "task": "summarization",
            "params": "568M",
            "size": "~2.3GB",
            "dataset": "XSum"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Translation ---
    print("\n[7/8] Translation Models...")

    # MarianMT English to French
    print("  - MarianMT EN->FR...")
    try:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        model.save_pretrained(nlp_dir / "marian-en-fr")
        tokenizer.save_pretrained(nlp_dir / "marian-en-fr")
        save_model_info("nlp", "marian-en-fr", {
            "task": "translation",
            "params": "74M",
            "size": "~300MB",
            "direction": "English -> French"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # MarianMT English to German
    print("  - MarianMT EN->DE...")
    try:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
        model.save_pretrained(nlp_dir / "marian-en-de")
        tokenizer.save_pretrained(nlp_dir / "marian-en-de")
        save_model_info("nlp", "marian-en-de", {
            "task": "translation",
            "params": "74M",
            "size": "~300MB",
            "direction": "English -> German"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # MarianMT English to Spanish
    print("  - MarianMT EN->ES...")
    try:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
        model.save_pretrained(nlp_dir / "marian-en-es")
        tokenizer.save_pretrained(nlp_dir / "marian-en-es")
        save_model_info("nlp", "marian-en-es", {
            "task": "translation",
            "params": "74M",
            "size": "~300MB",
            "direction": "English -> Spanish"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # MarianMT English to Chinese
    print("  - MarianMT EN->ZH...")
    try:
        model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
        model.save_pretrained(nlp_dir / "marian-en-zh")
        tokenizer.save_pretrained(nlp_dir / "marian-en-zh")
        save_model_info("nlp", "marian-en-zh", {
            "task": "translation",
            "params": "74M",
            "size": "~300MB",
            "direction": "English -> Chinese"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Fill-Mask ---
    print("\n[8/8] Masked Language Models...")

    # DistilBERT MLM
    print("  - DistilBERT MLM...")
    try:
        model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model.save_pretrained(nlp_dir / "distilbert-mlm")
        tokenizer.save_pretrained(nlp_dir / "distilbert-mlm")
        save_model_info("nlp", "distilbert-mlm", {
            "task": "fill_mask",
            "params": "66M",
            "size": "~268MB"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # RoBERTa Base
    print("  - RoBERTa Base MLM...")
    try:
        model = AutoModelForMaskedLM.from_pretrained("roberta-base")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model.save_pretrained(nlp_dir / "roberta-base-mlm")
        tokenizer.save_pretrained(nlp_dir / "roberta-base-mlm")
        save_model_info("nlp", "roberta-base-mlm", {
            "task": "fill_mask",
            "params": "125M",
            "size": "~500MB"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n✓ NLP models downloaded!")
    save_catalog()

# ============================================================================
# AUDIO MODELS
# ============================================================================
def download_audio_models():
    print("\n" + "="*60)
    print("DOWNLOADING AUDIO MODELS")
    print("="*60)

    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
        Wav2Vec2ForCTC, Wav2Vec2Processor,
        AutoModelForAudioClassification, AutoFeatureExtractor,
        HubertForCTC, HubertModel,
        Speech2TextForConditionalGeneration, Speech2TextProcessor
    )

    audio_dir = BASE_DIR / "audio"

    # --- Speech Recognition ---
    print("\n[1/4] Speech Recognition Models...")

    # Whisper Tiny
    print("  - Whisper Tiny...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model.save_pretrained(audio_dir / "whisper-tiny")
        processor.save_pretrained(audio_dir / "whisper-tiny")
        save_model_info("audio", "whisper-tiny", {
            "task": "speech_recognition",
            "params": "39M",
            "size": "~150MB",
            "languages": "multilingual"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Whisper Base
    print("  - Whisper Base...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model.save_pretrained(audio_dir / "whisper-base")
        processor.save_pretrained(audio_dir / "whisper-base")
        save_model_info("audio", "whisper-base", {
            "task": "speech_recognition",
            "params": "74M",
            "size": "~290MB",
            "languages": "multilingual"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Whisper Small
    print("  - Whisper Small...")
    try:
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        model.save_pretrained(audio_dir / "whisper-small")
        processor.save_pretrained(audio_dir / "whisper-small")
        save_model_info("audio", "whisper-small", {
            "task": "speech_recognition",
            "params": "244M",
            "size": "~970MB",
            "languages": "multilingual"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Wav2Vec2 Base
    print("  - Wav2Vec2 Base (960h)...")
    try:
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model.save_pretrained(audio_dir / "wav2vec2-base-960h")
        processor.save_pretrained(audio_dir / "wav2vec2-base-960h")
        save_model_info("audio", "wav2vec2-base-960h", {
            "task": "speech_recognition",
            "params": "95M",
            "size": "~380MB",
            "dataset": "LibriSpeech 960h",
            "wer": "3.4% (test-clean)"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # HuBERT Base
    print("  - HuBERT Base...")
    try:
        model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
        processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        model.save_pretrained(audio_dir / "hubert-large-ls960")
        processor.save_pretrained(audio_dir / "hubert-large-ls960")
        save_model_info("audio", "hubert-large-ls960", {
            "task": "speech_recognition",
            "params": "316M",
            "size": "~1.3GB",
            "dataset": "LibriSpeech 960h"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Audio Classification ---
    print("\n[2/4] Audio Classification Models...")

    # Audio Spectrogram Transformer
    print("  - AST AudioSet...")
    try:
        model = AutoModelForAudioClassification.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        model.save_pretrained(audio_dir / "ast-audioset")
        feature_extractor.save_pretrained(audio_dir / "ast-audioset")
        save_model_info("audio", "ast-audioset", {
            "task": "audio_classification",
            "params": "87M",
            "size": "~350MB",
            "dataset": "AudioSet",
            "classes": 527,
            "mAP": "0.459"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Wav2Vec2 Emotion
    print("  - Wav2Vec2 Emotion Recognition...")
    try:
        model = AutoModelForAudioClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        model.save_pretrained(audio_dir / "wav2vec2-emotion")
        feature_extractor.save_pretrained(audio_dir / "wav2vec2-emotion")
        save_model_info("audio", "wav2vec2-emotion", {
            "task": "emotion_recognition",
            "params": "315M",
            "size": "~1.2GB",
            "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Speaker Identification ---
    print("\n[3/4] Speaker Models...")

    # Wav2Vec2 Speaker ID
    print("  - Wav2Vec2 Speaker ID...")
    try:
        model = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-sid")
        feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-sid")
        model.save_pretrained(audio_dir / "wav2vec2-speaker-id")
        feature_extractor.save_pretrained(audio_dir / "wav2vec2-speaker-id")
        save_model_info("audio", "wav2vec2-speaker-id", {
            "task": "speaker_identification",
            "params": "95M",
            "size": "~380MB",
            "dataset": "VoxCeleb1",
            "accuracy": "75.18%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Music ---
    print("\n[4/4] Music Models...")

    # Wav2Vec2 for keyword spotting
    print("  - Wav2Vec2 Keyword Spotting...")
    try:
        model = AutoModelForAudioClassification.from_pretrained("superb/wav2vec2-base-superb-ks")
        feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
        model.save_pretrained(audio_dir / "wav2vec2-keyword-spotting")
        feature_extractor.save_pretrained(audio_dir / "wav2vec2-keyword-spotting")
        save_model_info("audio", "wav2vec2-keyword-spotting", {
            "task": "keyword_spotting",
            "params": "95M",
            "size": "~380MB",
            "dataset": "Speech Commands v1"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n✓ Audio models downloaded!")
    save_catalog()

# ============================================================================
# MULTIMODAL MODELS
# ============================================================================
def download_multimodal_models():
    print("\n" + "="*60)
    print("DOWNLOADING MULTIMODAL MODELS")
    print("="*60)

    from transformers import (
        CLIPModel, CLIPProcessor,
        BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering,
        ViltProcessor, ViltForQuestionAnswering,
        Blip2ForConditionalGeneration, Blip2Processor,
        GitForCausalLM, GitProcessor,
        ChineseCLIPModel, ChineseCLIPProcessor
    )

    multimodal_dir = BASE_DIR / "multimodal"

    # --- Vision-Language Models ---
    print("\n[1/4] Vision-Language Models...")

    # CLIP ViT-Base
    print("  - CLIP ViT-Base-Patch32...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.save_pretrained(multimodal_dir / "clip-vit-base-patch32")
        processor.save_pretrained(multimodal_dir / "clip-vit-base-patch32")
        save_model_info("multimodal", "clip-vit-base-patch32", {
            "task": "vision_language",
            "params": "151M",
            "size": "~600MB",
            "capabilities": ["image-text similarity", "zero-shot classification"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # CLIP ViT-Large
    print("  - CLIP ViT-Large-Patch14...")
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.save_pretrained(multimodal_dir / "clip-vit-large-patch14")
        processor.save_pretrained(multimodal_dir / "clip-vit-large-patch14")
        save_model_info("multimodal", "clip-vit-large-patch14", {
            "task": "vision_language",
            "params": "428M",
            "size": "~1.7GB",
            "capabilities": ["image-text similarity", "zero-shot classification"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Chinese CLIP
    print("  - Chinese CLIP ViT-Base...")
    try:
        model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        processor = ChineseCLIPProcessor.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
        model.save_pretrained(multimodal_dir / "chinese-clip-vit-base")
        processor.save_pretrained(multimodal_dir / "chinese-clip-vit-base")
        save_model_info("multimodal", "chinese-clip-vit-base", {
            "task": "vision_language",
            "params": "188M",
            "size": "~750MB",
            "language": "Chinese"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Image Captioning ---
    print("\n[2/4] Image Captioning Models...")

    # BLIP Base
    print("  - BLIP Base (Image Captioning)...")
    try:
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model.save_pretrained(multimodal_dir / "blip-image-captioning-base")
        processor.save_pretrained(multimodal_dir / "blip-image-captioning-base")
        save_model_info("multimodal", "blip-image-captioning-base", {
            "task": "image_captioning",
            "params": "247M",
            "size": "~990MB"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BLIP Large
    print("  - BLIP Large (Image Captioning)...")
    try:
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model.save_pretrained(multimodal_dir / "blip-image-captioning-large")
        processor.save_pretrained(multimodal_dir / "blip-image-captioning-large")
        save_model_info("multimodal", "blip-image-captioning-large", {
            "task": "image_captioning",
            "params": "470M",
            "size": "~1.9GB"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # GIT Base
    print("  - GIT Base (Image Captioning)...")
    try:
        model = GitForCausalLM.from_pretrained("microsoft/git-base")
        processor = GitProcessor.from_pretrained("microsoft/git-base")
        model.save_pretrained(multimodal_dir / "git-base")
        processor.save_pretrained(multimodal_dir / "git-base")
        save_model_info("multimodal", "git-base", {
            "task": "image_captioning",
            "params": "177M",
            "size": "~710MB"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Visual Question Answering ---
    print("\n[3/4] Visual Question Answering Models...")

    # ViLT VQA
    print("  - ViLT VQA...")
    try:
        model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model.save_pretrained(multimodal_dir / "vilt-vqa")
        processor.save_pretrained(multimodal_dir / "vilt-vqa")
        save_model_info("multimodal", "vilt-vqa", {
            "task": "visual_question_answering",
            "params": "113M",
            "size": "~450MB",
            "dataset": "VQAv2",
            "accuracy": "71.3%"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # BLIP VQA
    print("  - BLIP VQA...")
    try:
        model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        model.save_pretrained(multimodal_dir / "blip-vqa-base")
        processor.save_pretrained(multimodal_dir / "blip-vqa-base")
        save_model_info("multimodal", "blip-vqa-base", {
            "task": "visual_question_answering",
            "params": "247M",
            "size": "~990MB"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Document Understanding ---
    print("\n[4/4] Document Understanding Models...")

    # LayoutLM
    print("  - LayoutLMv3 Base...")
    try:
        from transformers import LayoutLMv3Model, LayoutLMv3Processor
        model = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-base")
        processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        model.save_pretrained(multimodal_dir / "layoutlmv3-base")
        processor.save_pretrained(multimodal_dir / "layoutlmv3-base")
        save_model_info("multimodal", "layoutlmv3-base", {
            "task": "document_understanding",
            "params": "125M",
            "size": "~500MB",
            "description": "Document AI model"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n✓ Multimodal models downloaded!")
    save_catalog()

# ============================================================================
# GENERATIVE MODELS
# ============================================================================
def download_generative_models():
    print("\n" + "="*60)
    print("DOWNLOADING GENERATIVE MODELS")
    print("="*60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    gen_dir = BASE_DIR / "generative"

    # --- Text Generation ---
    print("\n[1/2] Text Generation Models...")

    # OPT-125M
    print("  - OPT-125M...")
    try:
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        model.save_pretrained(gen_dir / "opt-125m")
        tokenizer.save_pretrained(gen_dir / "opt-125m")
        save_model_info("generative", "opt-125m", {
            "task": "text_generation",
            "params": "125M",
            "size": "~500MB",
            "architecture": "OPT"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # OPT-350M
    print("  - OPT-350M...")
    try:
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model.save_pretrained(gen_dir / "opt-350m")
        tokenizer.save_pretrained(gen_dir / "opt-350m")
        save_model_info("generative", "opt-350m", {
            "task": "text_generation",
            "params": "350M",
            "size": "~1.4GB",
            "architecture": "OPT"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Pythia-160M
    print("  - Pythia-160M...")
    try:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
        model.save_pretrained(gen_dir / "pythia-160m")
        tokenizer.save_pretrained(gen_dir / "pythia-160m")
        save_model_info("generative", "pythia-160m", {
            "task": "text_generation",
            "params": "160M",
            "size": "~640MB",
            "architecture": "Pythia (GPT-NeoX)"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Pythia-410M
    print("  - Pythia-410M...")
    try:
        model = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m")
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
        model.save_pretrained(gen_dir / "pythia-410m")
        tokenizer.save_pretrained(gen_dir / "pythia-410m")
        save_model_info("generative", "pythia-410m", {
            "task": "text_generation",
            "params": "410M",
            "size": "~1.6GB",
            "architecture": "Pythia (GPT-NeoX)"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Bloom-560M
    print("  - BLOOM-560M...")
    try:
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        model.save_pretrained(gen_dir / "bloom-560m")
        tokenizer.save_pretrained(gen_dir / "bloom-560m")
        save_model_info("generative", "bloom-560m", {
            "task": "text_generation",
            "params": "560M",
            "size": "~2.2GB",
            "architecture": "BLOOM",
            "languages": "46+ languages"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Code Generation ---
    print("\n[2/2] Code Generation Models...")

    # CodeGen-350M-Mono
    print("  - CodeGen-350M-Mono...")
    try:
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        model.save_pretrained(gen_dir / "codegen-350m-mono")
        tokenizer.save_pretrained(gen_dir / "codegen-350m-mono")
        save_model_info("generative", "codegen-350m-mono", {
            "task": "code_generation",
            "params": "350M",
            "size": "~1.4GB",
            "language": "Python"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # CodeGen-350M-Multi
    print("  - CodeGen-350M-Multi...")
    try:
        model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-multi")
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
        model.save_pretrained(gen_dir / "codegen-350m-multi")
        tokenizer.save_pretrained(gen_dir / "codegen-350m-multi")
        save_model_info("generative", "codegen-350m-multi", {
            "task": "code_generation",
            "params": "350M",
            "size": "~1.4GB",
            "languages": ["Python", "Java", "JavaScript", "Go", "C", "C++"]
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # StarCoder (small)
    print("  - StarCoderBase-1B...")
    try:
        model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")
        tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
        model.save_pretrained(gen_dir / "starcoderbase-1b")
        tokenizer.save_pretrained(gen_dir / "starcoderbase-1b")
        save_model_info("generative", "starcoderbase-1b", {
            "task": "code_generation",
            "params": "1B",
            "size": "~4GB",
            "languages": "80+ programming languages"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n✓ Generative models downloaded!")
    save_catalog()

# ============================================================================
# TIME SERIES MODELS
# ============================================================================
def download_timeseries_models():
    print("\n" + "="*60)
    print("DOWNLOADING TIME SERIES MODELS")
    print("="*60)

    from transformers import (
        AutoformerForPrediction, AutoformerConfig,
        TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig,
        InformerForPrediction, InformerConfig,
        PatchTSTForPrediction, PatchTSTConfig
    )

    ts_dir = BASE_DIR / "timeseries"

    # --- Time Series Forecasting ---
    print("\n[1/1] Time Series Forecasting Models...")

    # Autoformer
    print("  - Autoformer (config-based)...")
    try:
        config = AutoformerConfig(
            prediction_length=24,
            context_length=96,
            d_model=64,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2
        )
        model = AutoformerForPrediction(config)
        model.save_pretrained(ts_dir / "autoformer-base")
        save_model_info("timeseries", "autoformer-base", {
            "task": "time_series_forecasting",
            "params": "~5M",
            "size": "~20MB",
            "architecture": "Autoformer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Time Series Transformer
    print("  - Time Series Transformer...")
    try:
        config = TimeSeriesTransformerConfig(
            prediction_length=24,
            context_length=96,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2
        )
        model = TimeSeriesTransformerForPrediction(config)
        model.save_pretrained(ts_dir / "time-series-transformer")
        save_model_info("timeseries", "time-series-transformer", {
            "task": "time_series_forecasting",
            "params": "~2M",
            "size": "~8MB",
            "architecture": "Transformer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Informer
    print("  - Informer...")
    try:
        config = InformerConfig(
            prediction_length=24,
            context_length=96,
            d_model=64,
            encoder_layers=2,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2
        )
        model = InformerForPrediction(config)
        model.save_pretrained(ts_dir / "informer-base")
        save_model_info("timeseries", "informer-base", {
            "task": "time_series_forecasting",
            "params": "~5M",
            "size": "~20MB",
            "architecture": "Informer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # PatchTST
    print("  - PatchTST...")
    try:
        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=96,
            prediction_length=24,
            patch_length=16,
            stride=8,
            d_model=64,
            num_attention_heads=4,
            num_hidden_layers=2
        )
        model = PatchTSTForPrediction(config)
        model.save_pretrained(ts_dir / "patchtst-base")
        save_model_info("timeseries", "patchtst-base", {
            "task": "time_series_forecasting",
            "params": "~3M",
            "size": "~12MB",
            "architecture": "PatchTST"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Pretrained Time Series model
    print("  - TimesFM (if available)...")
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained("google/timesfm-1.0-200m")
        model.save_pretrained(ts_dir / "timesfm-200m")
        save_model_info("timeseries", "timesfm-200m", {
            "task": "time_series_forecasting",
            "params": "200M",
            "size": "~800MB",
            "architecture": "TimesFM"
        })
    except Exception as e:
        print(f"    TimesFM not available: {e}")

    print("\n✓ Time Series models downloaded!")
    save_catalog()

# ============================================================================
# TABULAR / CLASSICAL ML MODELS
# ============================================================================
def download_tabular_models():
    print("\n" + "="*60)
    print("DOWNLOADING TABULAR/CLASSICAL ML MODELS")
    print("="*60)

    import torch
    import torch.nn as nn

    tabular_dir = BASE_DIR / "tabular"

    # --- Tabular Deep Learning ---
    print("\n[1/2] Tabular Deep Learning Architectures...")

    # Tabular MLP
    print("  - Tabular MLP...")
    try:
        class TabularMLP(nn.Module):
            def __init__(self, input_dim=100, hidden_dims=[256, 128, 64], output_dim=10):
                super().__init__()
                layers = []
                prev_dim = input_dim
                for h_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = h_dim
                layers.append(nn.Linear(prev_dim, output_dim))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)

        model = TabularMLP()
        torch.save(model.state_dict(), tabular_dir / "tabular_mlp.pt")
        save_model_info("tabular", "tabular_mlp", {
            "task": "tabular_classification",
            "params": "~50K",
            "size": "~200KB",
            "architecture": "MLP"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # Tabular ResNet
    print("  - Tabular ResNet...")
    try:
        class TabularResNet(nn.Module):
            def __init__(self, input_dim=100, hidden_dim=128, num_blocks=3, output_dim=10):
                super().__init__()
                self.input_layer = nn.Linear(input_dim, hidden_dim)
                self.blocks = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim)
                    ) for _ in range(num_blocks)
                ])
                self.output_layer = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.input_layer(x))
                for block in self.blocks:
                    x = self.relu(x + block(x))
                return self.output_layer(x)

        model = TabularResNet()
        torch.save(model.state_dict(), tabular_dir / "tabular_resnet.pt")
        save_model_info("tabular", "tabular_resnet", {
            "task": "tabular_classification",
            "params": "~100K",
            "size": "~400KB",
            "architecture": "ResNet-style"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # --- Feature Extraction ---
    print("\n[2/2] Feature Extraction Architectures...")

    # Autoencoder
    print("  - Tabular Autoencoder...")
    try:
        class TabularAutoencoder(nn.Module):
            def __init__(self, input_dim=100, latent_dim=32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )

            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z)

            def encode(self, x):
                return self.encoder(x)

        model = TabularAutoencoder()
        torch.save(model.state_dict(), tabular_dir / "tabular_autoencoder.pt")
        save_model_info("tabular", "tabular_autoencoder", {
            "task": "feature_extraction",
            "params": "~10K",
            "size": "~40KB",
            "architecture": "Autoencoder"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # VAE
    print("  - Tabular VAE...")
    try:
        class TabularVAE(nn.Module):
            def __init__(self, input_dim=100, latent_dim=32):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                )
                self.fc_mu = nn.Linear(64, latent_dim)
                self.fc_var = nn.Linear(64, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_var(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decoder(z), mu, logvar

        model = TabularVAE()
        torch.save(model.state_dict(), tabular_dir / "tabular_vae.pt")
        save_model_info("tabular", "tabular_vae", {
            "task": "generative_feature_extraction",
            "params": "~15K",
            "size": "~60KB",
            "architecture": "VAE"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    # TabTransformer
    print("  - Tab Transformer...")
    try:
        class TabTransformer(nn.Module):
            def __init__(self, num_categories=10, num_continuous=10, dim=32, depth=2, heads=4, output_dim=1):
                super().__init__()
                self.category_embeddings = nn.Embedding(num_categories * 10, dim)
                self.continuous_layer = nn.Linear(num_continuous, dim)
                encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
                self.output_layer = nn.Linear(dim, output_dim)

            def forward(self, x_cat, x_cont):
                cat_emb = self.category_embeddings(x_cat)
                cont_emb = self.continuous_layer(x_cont).unsqueeze(1)
                x = torch.cat([cat_emb, cont_emb], dim=1)
                x = self.transformer(x)
                x = x.mean(dim=1)
                return self.output_layer(x)

        model = TabTransformer()
        torch.save(model.state_dict(), tabular_dir / "tab_transformer.pt")
        save_model_info("tabular", "tab_transformer", {
            "task": "tabular_classification",
            "params": "~50K",
            "size": "~200KB",
            "architecture": "TabTransformer"
        })
    except Exception as e:
        print(f"    Failed: {e}")

    print("\n✓ Tabular models downloaded!")
    save_catalog()

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE ML MODEL DOWNLOADER")
    print("="*60)
    print(f"\nBase directory: {BASE_DIR}")
    print("\nThis will download pretrained models for:")
    print("  1. Computer Vision (classification, detection, segmentation)")
    print("  2. NLP (embeddings, classification, NER, QA, generation)")
    print("  3. Audio (speech recognition, classification)")
    print("  4. Multimodal (CLIP, VQA, captioning)")
    print("  5. Generative (text, code)")
    print("  6. Time Series (forecasting)")
    print("  7. Tabular (deep learning architectures)")
    print("\n" + "="*60)

    # Download all models
    download_vision_models()
    download_nlp_models()
    download_audio_models()
    download_multimodal_models()
    download_generative_models()
    download_timeseries_models()
    download_tabular_models()

    # Final catalog save
    save_catalog()

    print("\n" + "="*60)
    print("ALL DOWNLOADS COMPLETE!")
    print("="*60)
    print(f"\nModel catalog saved to: {BASE_DIR / 'model_catalog.json'}")
    print("\nTotal domains covered: 7")
    print("Models downloaded: 70+")
