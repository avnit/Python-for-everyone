"""
OCR Engine Module - Image to Text Conversion
=============================================
Uses EasyOCR (deep-learning based) to extract text from financial documents,
charts, invoices, and any image containing text.

No external binary required - pure Python install.
"""

import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class OCREngine:
    """
    Optical Character Recognition engine that converts images to text.

    Uses EasyOCR under the hood, which is based on CRNN (Convolutional
    Recurrent Neural Network) architecture - a form of deep learning / ML.
    """

    def __init__(self, languages=None, use_gpu=False):
        """
        Initialize the OCR engine.

        Args:
            languages (list): Languages to detect. Default: ['en']
            use_gpu (bool): Use GPU acceleration if available. Default: False
        """
        if languages is None:
            languages = ['en']

        print("[OCR] Loading EasyOCR model (first run downloads ~100MB)...")
        try:
            import easyocr
            self.reader = easyocr.Reader(languages, gpu=use_gpu)
            self.available = True
            print("[OCR] Model loaded successfully.")
        except ImportError:
            print("[OCR] EasyOCR not installed. Run: pip install easyocr")
            self.available = False
            self.reader = None

    def preprocess_image(self, image_path):
        """
        Preprocess image to improve OCR accuracy.

        Applies contrast enhancement and sharpening - common ML pre-processing steps.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Preprocessed image as numpy array
        """
        img = Image.open(image_path).convert('RGB')

        # Resize if too small (OCR works better on larger images)
        width, height = img.size
        if width < 800:
            scale = 800 / width
            img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)

        # Sharpen
        img = img.filter(ImageFilter.SHARPEN)

        return np.array(img)

    def extract_text(self, image_path):
        """
        Extract all text from an image.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: All extracted text joined by newlines
        """
        if not self.available:
            return "OCR not available - please install easyocr"

        if not os.path.exists(image_path):
            return f"Image not found: {image_path}"

        print(f"[OCR] Analyzing image: {image_path}")
        preprocessed = self.preprocess_image(image_path)
        results = self.reader.readtext(preprocessed)

        # Sort results top-to-bottom, left-to-right (reading order)
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

        lines = [result[1] for result in results]
        return '\n'.join(lines)

    def extract_with_confidence(self, image_path):
        """
        Extract text with confidence scores for each detected region.

        Args:
            image_path (str): Path to the image file

        Returns:
            list: List of (text, confidence) tuples
        """
        if not self.available:
            return []

        if not os.path.exists(image_path):
            print(f"[OCR] Image not found: {image_path}")
            return []

        preprocessed = self.preprocess_image(image_path)
        results = self.reader.readtext(preprocessed)
        return [(result[1], round(result[2] * 100, 1)) for result in results]

    def extract_financial_data(self, image_path):
        """
        Extract text and attempt to identify financial values (numbers, percentages, $).

        Args:
            image_path (str): Path to image

        Returns:
            dict: {'raw_text': str, 'financial_items': list}
        """
        import re

        raw_text = self.extract_text(image_path)

        # Pattern match financial values: $1,234.56 or 12.5% or 1,000
        money_pattern = r'\$[\d,]+\.?\d*'
        percent_pattern = r'\d+\.?\d*\s*%'
        number_pattern = r'\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b'

        financial_items = []
        financial_items.extend(re.findall(money_pattern, raw_text))
        financial_items.extend(re.findall(percent_pattern, raw_text))
        financial_items.extend(re.findall(number_pattern, raw_text))

        return {
            'raw_text': raw_text,
            'financial_items': list(set(financial_items))
        }

    def display_results(self, image_path):
        """
        Display OCR results in a formatted, readable way.

        Args:
            image_path (str): Path to the image file
        """
        print("\n" + "=" * 60)
        print("OCR EXTRACTION RESULTS")
        print("=" * 60)

        results = self.extract_with_confidence(image_path)

        if not results:
            print("No text detected in image.")
            return ""

        print(f"{'TEXT':<40} {'CONFIDENCE':>10}")
        print("-" * 52)

        all_text = []
        for text, confidence in results:
            status = "HIGH" if confidence > 80 else ("MEDIUM" if confidence > 50 else "LOW")
            print(f"{text:<40} {confidence:>6.1f}% [{status}]")
            all_text.append(text)

        full_text = ' '.join(all_text)
        print("\n--- FULL EXTRACTED TEXT ---")
        print(full_text)
        print("=" * 60)

        return full_text
