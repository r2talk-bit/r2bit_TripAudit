# Standard library imports for file and path operations
import os  # Provides functions for interacting with the operating system and file paths

# Computer vision and image processing libraries
import cv2  # OpenCV library for image processing operations like thresholding and denoising
import numpy as np  # NumPy for numerical operations and array handling

# OCR (Optical Character Recognition) library
import pytesseract  # Python wrapper for Tesseract OCR engine to extract text from images

# PDF and image manipulation libraries
from pdf2image import convert_from_path  # Converts PDF pages to PIL Image objects
from PIL import Image, ImageDraw, ImageFont  # Python Imaging Library for image creation and editing

# Type hints for better code documentation and IDE support
from typing import List, Dict, Tuple, Optional  # Type annotations for function signatures


# Module-level docstring
"""
Funções para pré-processamento de relatórios de despesas em PDF.
Inclui funções para conversão de PDF para imagens, melhoria de qualidade
e visualização de resultados de OCR e extração.
"""
    
def pdf_to_images(pdf_path: str, output_dir: str = None, dpi: int = 300) -> List[str]:
    """
    Converte um PDF em imagens e salva em disco.
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        output_dir: Diretório para salvar as imagens (se None, usa diretório do PDF)
        dpi: Resolução para conversão
        
    Returns:
        Lista de caminhos para as imagens salvas
    """
    # If no output directory is specified, use the same directory as the PDF file
    # This is a common pattern for handling default output locations
    if output_dir is None:
        output_dir = os.path.dirname(pdf_path)
    
    # Create the output directory if it doesn't exist
    # exist_ok=True prevents errors if the directory already exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base filename without extension (e.g., 'expense_report' from 'expense_report.pdf')
    # This is used to create consistent filenames for the output images
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    
    # Convert the PDF to a list of PIL Image objects
    # The dpi parameter controls the resolution of the output images
    # Higher DPI values result in larger, more detailed images but require more memory
    print(f"Convertendo PDF para imagens: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Save each page as a separate JPEG image
    # We track the paths to all saved images to return them at the end
    image_paths = []
    for i, image in enumerate(images):
        # Create a filename for each page (e.g., 'expense_report_page_1.jpg')
        image_path = os.path.join(output_dir, f"{base_name}_page_{i+1}.jpg")
        # Save the PIL Image as a JPEG file
        image.save(image_path, "JPEG")
        # Add the path to our list of saved images
        image_paths.append(image_path)
        
    print(f"Convertidas {len(images)} páginas para {output_dir}")
    return image_paths
    
def enhance_image(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Melhora a qualidade da imagem para OCR.
    
    Args:
        image_path: Caminho para a imagem
        output_path: Caminho para salvar a imagem melhorada (se None, usa sufixo _enhanced)
        
    Returns:
        Caminho para a imagem melhorada
    """
    # Generate a default output path if none is provided
    # The convention is to add '_enhanced' to the original filename
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_enhanced.jpg"
        
    # Load the image using OpenCV
    # OpenCV reads images in BGR color format (not RGB)
    img = cv2.imread(image_path)
    
    # Convert the image to grayscale
    # This simplifies processing and improves OCR accuracy since color isn't needed
    # for text recognition - it reduces noise and computational complexity
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to create a binary image
    # This helps separate text (foreground) from background
    # ADAPTIVE_THRESH_GAUSSIAN_C uses a Gaussian-weighted sum of neighborhood values
    # The block size (11) determines the size of the neighborhood
    # The constant (2) is subtracted from the weighted mean to fine-tune the threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Apply non-local means denoising to reduce noise while preserving edges
    # This algorithm replaces each pixel with a weighted average of pixels
    # from the entire image, with weights based on patch similarity
    # Parameters: h=10 (filter strength), templateWindowSize=7, searchWindowSize=21
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Save the enhanced image to disk
    cv2.imwrite(output_path, denoised)
    
    print(f"Imagem melhorada salva em: {output_path}")
    return output_path
    
def ocr_image(image_path: str, tesseract_lang: str = "eng") -> List[Dict]:
    """
    Realiza OCR na imagem e retorna lista de palavras com bounding boxes.
    
    Args:
        image_path: Caminho para a imagem
        tesseract_lang: Idioma para o OCR do Tesseract (padrão: 'eng' para inglês)
        
    Returns:
        Lista de dicionários com 'text' e 'bbox' (x0, y0, x1, y1)
    """
    # Load the image using PIL (Python Imaging Library)
    # PIL is used here instead of OpenCV because pytesseract works well with PIL images
    image = Image.open(image_path)
    
    # Perform OCR (Optical Character Recognition) using Tesseract
    # image_to_data extracts text along with positioning information
    # lang parameter specifies the language model to use
    # output_type=DICT returns the results as a Python dictionary for easier processing
    data = pytesseract.image_to_data(
        image, lang=tesseract_lang, 
        output_type=pytesseract.Output.DICT
    )
    
    # Process OCR results to create a structured list of words with their properties
    # We filter out empty strings and collect detailed information about each word
    words = []
    for i, word in enumerate(data['text']):
        # Only process non-empty words (skip spaces and noise)
        if word.strip() != "":
            # Extract position information (left, top, width, height)
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Calculate the bounding box coordinates (x0, y0, x1, y1)
            bbox = [x, y, x + w, y + h]
            # Get the confidence score (0-100) for this word recognition
            conf = int(data['conf'][i])
            
            # Create a dictionary with all the word information
            # This includes the text, position, confidence, and Tesseract's
            # hierarchical structure information (page, block, paragraph, line)
            words.append({
                "text": word,
                "bbox": bbox,
                "confidence": conf,
                "page_num": data['page_num'][i],
                "block_num": data['block_num'][i],
                "par_num": data['par_num'][i],
                "line_num": data['line_num'][i],
            })
    
    return words
    
def visualize_ocr_results(image_path: str, words: List[Dict], output_path: Optional[str] = None) -> str:
    """
    Visualiza os resultados do OCR na imagem.
    
    Args:
        image_path: Caminho para a imagem
        words: Lista de palavras com bounding boxes (resultado do ocr_image)
        output_path: Caminho para salvar a imagem com visualização
        
    Returns:
        Caminho para a imagem com visualização
    """
    # Generate a default output path if none is provided
    # The convention is to add '_ocr_viz' to the original filename
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_ocr_viz.jpg"
    
    # Load the original image and convert to RGB mode to ensure color compatibility
    # We need RGB mode for drawing colored boxes and text
    image = Image.open(image_path).convert("RGB")
    # Create a drawing object to add visual elements to the image
    draw = ImageDraw.Draw(image)
    
    # Try to load a TrueType font for better text rendering
    # Fall back to the default font if the specified font isn't available
    try:
        # Try to use Arial font at 12pt size for text annotations
        font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        # If Arial isn't available, use the default system font
        font = ImageFont.load_default()
    
    # Draw bounding boxes and text labels for each recognized word
    # This creates a visual representation of what the OCR engine detected
    for word in words:
        # Extract the bounding box coordinates and text
        bbox = word["bbox"]
        text = word["text"]
        # Get the confidence score, defaulting to 0 if not available
        conf = word.get("confidence", 0)
        
        # Color-code the boxes based on confidence level
        # This provides a visual indication of OCR reliability
        if conf < 60:
            color = (255, 0, 0)  # Red for low confidence (<60%)
        elif conf < 80:
            color = (255, 165, 0)  # Orange for medium confidence (60-80%)
        else:
            color = (0, 255, 0)  # Green for high confidence (>80%)
        
        # Draw a rectangle around the detected word
        # width=2 makes the outline 2 pixels thick for better visibility
        draw.rectangle(bbox, outline=color, width=2)
        
        # Draw the recognized text above the bounding box
        # Positioning it 15 pixels above prevents overlap with the box
        draw.text((bbox[0], bbox[1] - 15), text, fill=color, font=font)
    
    # Save the annotated image to the output path
    image.save(output_path)
    
    print(f"Visualização OCR salva em: {output_path}")
    return output_path
    
def extract_tables(image_path: str, tesseract_lang: str = "eng") -> List[np.ndarray]:
    """
    Detecta e extrai tabelas da imagem usando img2table.
    
    Args:
        image_path: Caminho para a imagem
        tesseract_lang: Idioma para o OCR do Tesseract (padrão: 'eng' para inglês)
        
    Returns:
        Lista de arrays NumPy representando as tabelas detectadas
    """
    # Import img2table components here to avoid loading them if not needed
    # This is a form of lazy loading that improves startup performance
    from img2table.ocr import TesseractOCR
    from img2table.document import Image as Img2TableImage
    
    # Configure the OCR engine for table extraction
    # We use the same language setting as passed to the function
    # This ensures consistency between general OCR and table-specific OCR
    ocr = TesseractOCR(lang=tesseract_lang)
    
    # Load the image using img2table's document class
    # This class is specifically designed for document analysis and table detection
    img_document = Img2TableImage(image_path)
    
    # Extract tables from the image
    # img2table uses computer vision techniques to identify grid structures
    # that likely represent tables in the document
    tables_dict = img_document.extract_tables(ocr=ocr)
    
    # Initialize an empty list to store the extracted table images
    # We'll return these as NumPy arrays for further processing
    result_tables = []
    
    # Check if any tables were found
    # tables_dict will be a dictionary mapping table IDs to table objects if tables were found
    if isinstance(tables_dict, dict) and tables_dict:
        for table_id, table in tables_dict.items():
            # Get the bounding box coordinates of the detected table
            # This tells us where in the original image the table is located
            bbox = table.bbox
            x, y, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
            
            # Load the original image and crop out just the table region
            # We use OpenCV here because we want the result as a NumPy array
            img = cv2.imread(image_path)
            table_img = img[y:y+h, x:x+w]  # Crop using NumPy array slicing
            result_tables.append(table_img)
        
        print(f"Detectadas {len(result_tables)} tabelas na imagem usando img2table")
    else:
        print("Nenhuma tabela detectada na imagem usando img2table")
        
    return result_tables
    
def visualize_tables(image_path: str, output_dir: Optional[str] = None, tesseract_lang: str = "eng") -> List[str]:
    """
    Detecta tabelas e salva visualizações.
    
    Args:
        image_path: Caminho para a imagem
        output_dir: Diretório para salvar as imagens das tabelas
        tesseract_lang: Idioma para o OCR do Tesseract (padrão: 'eng' para inglês)
        
    Returns:
        Lista de caminhos para as imagens das tabelas
    """
    # Use the directory of the input image if no output directory is specified
    # This keeps all related outputs in the same location by default
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base filename without extension for consistent naming
    # This helps maintain the relationship between source and output files
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Extract tables from the image using the extract_tables function
    # This returns a list of NumPy arrays, each representing a detected table
    tables = extract_tables(image_path, tesseract_lang=tesseract_lang)
    
    # Save each detected table as a separate image file
    # We track the paths to return them to the caller
    table_paths = []
    for i, table in enumerate(tables):
        # Create a filename for each table (e.g., 'expense_report_page_1_table_1.jpg')
        table_path = os.path.join(output_dir, f"{base_name}_table_{i+1}.jpg")
        # Save the table image using OpenCV
        cv2.imwrite(table_path, table)
        # Add the path to our list of saved tables
        table_paths.append(table_path)
    
    return table_paths
    
def process_pdf(pdf_path: str, output_dir: Optional[str] = None, enhance: bool = True, tesseract_lang: str = "eng") -> Dict:
    """
    Processa um PDF completo: converte para imagens, melhora qualidade, 
    extrai texto e tabelas.
    
    Args:
        pdf_path: Caminho para o arquivo PDF
        output_dir: Diretório para salvar os resultados
        enhance: Se True, aplica melhoria de imagem
        tesseract_lang: Idioma para o OCR do Tesseract (padrão: 'eng' para inglês)
        
    Returns:
        Dicionário com resultados do processamento
    """
    # Set up the output directory with a default location if not specified
    # By default, creates a 'processed' subdirectory in the same location as the PDF
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(pdf_path), "processed")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # STEP 1: Convert the PDF to images
    # This is the first step in our processing pipeline
    # We need to convert the PDF to images to apply OCR and other image processing techniques
    image_paths = pdf_to_images(pdf_path, output_dir)
    
    # Initialize the results dictionary to store all processing outputs
    # This will be returned at the end with all the processing results
    results = {
        "pdf_path": pdf_path,  # Original PDF path for reference
        "output_dir": output_dir,  # Where all outputs are stored
        "pages": []  # Will contain results for each page
    }
    
    # STEP 2: Process each page of the PDF individually
    # For each page image, we'll perform enhancement, OCR, and table extraction
    for i, image_path in enumerate(image_paths):
        # Initialize a dictionary to store results for this specific page
        page_result = {"page_num": i + 1, "image_path": image_path}
        
        # STEP 2.1: Image enhancement (optional)
        # If enabled, apply image enhancement techniques to improve OCR accuracy
        if enhance:
            # Apply thresholding and denoising to improve image quality
            enhanced_path = enhance_image(image_path)
            page_result["enhanced_path"] = enhanced_path
            # Use the enhanced image for OCR
            ocr_image_path = enhanced_path
        else:
            # Use the original image if enhancement is disabled
            ocr_image_path = image_path
        
        # STEP 2.2: Extract text using OCR
        # Perform OCR to extract text and its position from the image
        words = ocr_image(ocr_image_path, tesseract_lang=tesseract_lang)
        page_result["words"] = words
        
        # STEP 2.3: Create a visualization of the OCR results
        # This helps in debugging and verifying the OCR accuracy
        viz_path = visualize_ocr_results(ocr_image_path, words)
        page_result["ocr_viz_path"] = viz_path
        
        # STEP 2.4: Extract tables from the image
        # Detect and extract tabular data, which is common in expense reports
        table_paths = visualize_tables(ocr_image_path, output_dir, tesseract_lang=tesseract_lang)
        page_result["table_paths"] = table_paths
        
        # Add this page's results to the overall results dictionary
        results["pages"].append(page_result)
    
    print(f"Processamento concluído. Resultados salvos em {output_dir}")
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Pré-processamento de relatórios de despesas em PDF")
    parser.add_argument("pdf_path", help="Caminho para o arquivo PDF")
    parser.add_argument("--output_dir", "-o", help="Diretório para salvar os resultados")
    parser.add_argument("--no-enhance", action="store_true", help="Desativar melhoria de imagem")
    parser.add_argument("--dpi", type=int, default=300, help="DPI para conversão do PDF")
    parser.add_argument("--lang", default="eng", help="Idioma para OCR (padrão: inglês)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Arquivo {args.pdf_path} não encontrado.")
        return 1
    
    # Process the PDF using our functional approach
    results = process_pdf(
        args.pdf_path, 
        output_dir=args.output_dir,
        enhance=not args.no_enhance,
        tesseract_lang=args.lang
    )
    
    print(f"\nResumo do processamento:")
    print(f"PDF: {results['pdf_path']}")
    print(f"Diretório de saída: {results['output_dir']}")
    print(f"Total de páginas processadas: {len(results['pages'])}")
    
    for page in results["pages"]:
        print(f"\nPágina {page['page_num']}:")
        print(f"  Imagem: {page['image_path']}")
        if "enhanced_path" in page:
            print(f"  Imagem melhorada: {page['enhanced_path']}")
        print(f"  Visualização OCR: {page['ocr_viz_path']}")
        print(f"  Palavras extraídas: {len(page['words'])}")
        print(f"  Tabelas detectadas: {len(page['table_paths'])}")
    
    return 0


if __name__ == "__main__":
    exit(main())
