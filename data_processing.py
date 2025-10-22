from pdf2image import convert_from_path
import pytesseract
import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import re
from opencc import OpenCC

# Initialize OpenCC converter for Traditional to Simplified Chinese
cc = OpenCC('t2s')  # t2s = Traditional to Simplified

def convert_traditional_to_simplified(text):
    """Convert Traditional Chinese characters to Simplified Chinese using OpenCC."""
    try:
        return cc.convert(text)
    except Exception as e:
        print(f"Warning: Could not convert text: {e}")
        return text

def preprocess_image_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy and reduce watermark interference.
    """
    # Convert to grayscale
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast to make text more prominent
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Apply slight blur to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

def filter_watermark_text(text):
    """
    Filter out common watermark patterns and repetitive text.
    """
    if not text or not text.strip():
        return ""
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip very short lines (likely noise)
        if len(line) < 2:
            continue
            
        # Skip lines that are mostly repeated characters (common in watermarks)
        if len(set(line)) < 3 and len(line) > 5:
            continue
            
        # Skip lines that appear to be watermarks (very repetitive patterns)
        if re.match(r'^(.{1,3})\1{3,}$', line):
            continue
            
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def process_pdf(pdf_path, output_path="output_notes.txt", language='chi_sim', max_pages=None):
    """
    Process a PDF file and extract text using OCR with watermark filtering.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_path (str): Path for the output text file
        language (str): Tesseract language code
        max_pages (int): Maximum number of pages to process (None for all pages)
    """
    try:
        print(f"Processing PDF: {pdf_path}")
        
        # Check if PDF exists
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file '{pdf_path}' not found!")
            return False
            
        # Convert PDF pages to images with optimized settings
        print("Converting PDF to images...")
        pages = convert_from_path(pdf_path, dpi=200, fmt='jpeg', jpegopt={'quality': 85})
        print(f"Found {len(pages)} pages")
        
        # Limit pages if specified
        if max_pages and max_pages < len(pages):
            pages = pages[:max_pages]
            print(f"Processing first {max_pages} pages only")
        
        full_text = ""
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)}...")
            try:
                # Preprocess the image
                processed_image = preprocess_image_for_ocr(page)
                
                # Extract text with Chinese language support
                text = pytesseract.image_to_string(processed_image, lang=language)
                
                # Filter out watermarks
                filtered_text = filter_watermark_text(text)
                
                # Convert Traditional Chinese to Simplified Chinese
                filtered_text = convert_traditional_to_simplified(filtered_text)
                
                if filtered_text.strip():
                    full_text += f"\n--- Page {i+1} ---\n{filtered_text}\n"
                else:
                    full_text += f"\n--- Page {i+1} ---\n[No readable text found]\n"
                    
            except Exception as e:
                print(f"Error processing page {i+1}: {e}")
                full_text += f"\n--- Page {i+1} ---\n[Error extracting text]\n"
        
        # Save the extracted text
        print(f"Saving extracted text to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        print(f"Successfully processed {len(pages)} pages!")
        return True
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return False

def process_with_multiple_approaches(pdf_path, output_path="output_notes.txt", max_pages=5):
    """
    Try multiple OCR approaches to get the best results.
    """
    approaches = [
        ("chi_sim", "Simplified Chinese"),
        ("chi_sim+eng", "Simplified Chinese + English"),
        ("chi_tra", "Traditional Chinese (will be converted to Simplified)"),
    ]
    
    best_result = ""
    best_approach = ""
    max_text_length = 0
    
    for lang, description in approaches:
        print(f"\nTrying approach: {description} ({lang})")
        temp_output = f"temp_output_{lang.replace('+', '_')}.txt"
        
        success = process_pdf(pdf_path, temp_output, lang, max_pages)
        
        if success and os.path.exists(temp_output):
            with open(temp_output, 'r', encoding='utf-8') as f:
                content = f.read()
                # Convert Traditional Chinese to Simplified if needed
                if lang == "chi_tra":
                    content = convert_traditional_to_simplified(content)
                if len(content) > max_text_length:
                    max_text_length = len(content)
                    best_result = content
                    best_approach = description
            
            # Clean up temp file
            os.remove(temp_output)
    
    if best_result:
        print(f"\nBest approach: {best_approach}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(best_result)
        return True
    
    return False

if __name__ == "__main__":
    # Process the lecture notes PDF with Chinese text and watermark filtering
    print("Starting PDF processing with watermark filtering...")
    
    # Try multiple approaches to get the best results
    success = process_with_multiple_approaches("input_notes.pdf", "output_notes.txt", max_pages=None)
    
    if success:
        print("PDF processing completed successfully!")
        print("Check 'output_notes.txt' for the extracted text.")
    else:
        print("PDF processing failed!")