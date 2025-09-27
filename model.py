import cv2 # openCV
import easyocr # OCR library
import numpy as np 
import matplotlib.pyplot as plt # Data Visualization

# Initialize EasyOCR reader (this will download models on first run)
reader = easyocr.Reader(['en'])  # Add other languages as needed: ['en', 'es', 'fr']

img = cv2.imread('training_data/algrebra.webp')

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Remove Noise
    denoised = cv2.medianBlur(gray, 3)
    
    # Enhance Contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Binarization (convert to black and white)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

def extract_text_regions(image):
    # Find Text Regions using contours
    processed = process_image(image)
    
    # Find Contours that likely contain text
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # filter by aspect ratio and size
        if w > 20 and h > 10 and w/h > 2:  # Typical text size
            text_regions.append((x, y, w, h))
    
    return text_regions

def extract_text_with_easyocr(image):
    """Extract text using EasyOCR"""
    try:
        # EasyOCR can work with original image or processed image
        # Using original for better results, but you can use processed if needed
        processed = process_image(image)
        
        # Extract text with EasyOCR
        results = reader.readtext(image)  # or use 'processed' for binary image
        
        # Combine all detected text
        extracted_text = ""
        text_regions = []
        
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Filter by confidence threshold
                extracted_text += text + " "
                
                # Get bounding box coordinates
                top_left = tuple([int(val) for val in bbox[0]])
                bottom_right = tuple([int(val) for val in bbox[2]])
                text_regions.append((top_left[0], top_left[1], 
                                   bottom_right[0] - top_left[0], 
                                   bottom_right[1] - top_left[1]))
        
        return extracted_text.strip(), text_regions, results
    
    except Exception as e:
        print(f"EasyOCR Error: {e}")
        return "OCR failed", [], []

def visualize_easyocr_results(image, results):
    """Visualize EasyOCR detected text regions"""
    result_img = image.copy()
    
    for (bbox, text, confidence) in results:
        if confidence > 0.5:  # Only show high-confidence detections
            # Convert bbox to integer coordinates
            points = np.array(bbox, dtype=np.int32)
            cv2.polylines(result_img, [points], True, (0, 255, 0), 2)
            
            # Add text label
            cv2.putText(result_img, f"{text[:15]}... ({confidence:.2f})", 
                       (int(bbox[0][0]), int(bbox[0][1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display using matplotlib
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('EasyOCR Detected Text')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_text_regions_traditional(image):
    """Traditional contour-based text region detection"""
    processed = process_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 10 and w/h > 2:
            text_regions.append((x, y, w, h))
    
    return text_regions

# Main execution
if img is not None:
    print("Processing image with EasyOCR...")
    
    # Extract text using EasyOCR
    extracted_text, easyocr_regions, raw_results = extract_text_with_easyocr(img)
    
    # Also get traditional contour-based regions for comparison
    traditional_regions = extract_text_regions_traditional(img)
    
    print(f"EasyOCR detected {len(easyocr_regions)} text regions")
    print(f"Traditional method detected {len(traditional_regions)} text regions")
    print("\nExtracted Text:")
    print(extracted_text)
    
    print("\nDetailed Results:")
    for i, (bbox, text, confidence) in enumerate(raw_results):
        if confidence > 0.5:
            print(f"Region {i+1}: '{text}' (confidence: {confidence:.3f})")
    
    # Visualize results
    visualize_easyocr_results(img, raw_results)
    
else:
    print("Error: Could not load image. Check the file path.")