"""
Expense report auditing logic for R2Bit TripAudit.
This module contains the core business logic for auditing expense reports,
separating it from the presentation layer in streamlit_app.py.

This module implements the business logic layer of the application,
following the separation of concerns design principle to keep the
UI code separate from the data processing logic.

This module uses a functional programming approach with pure functions
rather than classes, making the code more modular and easier to test.
"""

# Standard library imports for file operations and regex
import os
import re
from datetime import datetime

# Deep learning and OCR libraries
import torch  # PyTorch for deep learning operations
import pytesseract  # Interface to Tesseract OCR engine
import pandas as pd  # Data manipulation and analysis
from pdf2image import convert_from_path  # Convert PDF to images
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification  # Document understanding model
from PIL import Image  # Python Imaging Library for image processing
from typing import List, Dict, Tuple, Optional  # Type hints for better code documentation
from img2table.document import Image as Img2TableImage
from img2table.ocr import TesseractOCR

# Import global configurations from config.py
from config import MODEL_NAME, TESSERACT_LANG, TESSERACT_CONFIG, PDF_DPI, EXPENSE_CATEGORIES, REGEX_PATTERNS

# Set device for PyTorch operations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

# Module-level variables to store model and processor
_processor = None
_model = None

def initialize_model(model_name=MODEL_NAME, device=DEVICE, tesseract_lang=TESSERACT_LANG):
    """
    Initialize the LayoutLMv3 model and processor.
    
    Args:
        model_name: Name or path of the pre-trained model
        device: Device to run the model on (CPU or GPU)
        tesseract_lang: Language for OCR
        
    Returns:
        Tuple containing the processor and model
    """
    global _processor, _model
    
    # Initialize the LayoutLMv3 model and processor if not already initialized
    if _processor is None or _model is None:
        # Initialize the processor with apply_ocr=False because we'll handle OCR separately with Tesseract
        _processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        
        # Load the token classification model for entity extraction
        _model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        
        # Define detailed label configuration for expense reports if not already present
        # This is a comprehensive labeling system with 23 different entity types
        # Each entity type has a Beginning (B-) and Inside (I-) tag following the BIO tagging scheme
        if not hasattr(_model.config, 'id2label') or not _model.config.id2label:
            _model.config.id2label = {
                0: "O",                    # Outside (not a relevant entity)
                1: "B-VALOR_TOTAL",        # Beginning of a total value
                2: "I-VALOR_TOTAL",        # Continuation of a total value
                3: "B-VALOR_ITEM",         # Beginning of an individual item value
                4: "I-VALOR_ITEM",         # Continuation of an item value
                5: "B-DATA",               # Beginning of a date
                6: "I-DATA",               # Continuation of a date
                7: "B-CATEGORIA",          # Beginning of an expense category
                8: "I-CATEGORIA",          # Continuation of a category
                9: "B-FORNECEDOR",         # Beginning of vendor/supplier name
                10: "I-FORNECEDOR",        # Continuation of vendor name
                11: "B-DESCRICAO",         # Beginning of item description
                12: "I-DESCRICAO",         # Continuation of description
                13: "B-NUMERO_DOCUMENTO",  # Beginning of document/receipt number
                14: "I-NUMERO_DOCUMENTO",  # Continuation of document number
                15: "B-METODO_PAGAMENTO",  # Beginning of payment method
                16: "I-METODO_PAGAMENTO",  # Continuation of payment method
                17: "B-MOEDA",             # Beginning of currency code
                18: "I-MOEDA",             # Continuation of currency code
                19: "B-TAXA",              # Beginning of a tax or service fee
                20: "I-TAXA",              # Continuation of a tax
                21: "B-NOME_FUNCIONARIO",  # Beginning of employee name
                22: "I-NOME_FUNCIONARIO",  # Continuation of employee name
                23: "B-CIDADE",            # Beginning of city name
                24: "I-CIDADE",            # Continuation of city name
                25: "B-LOCAL_HOSPEDAGEM",  # Beginning of accommodation location
                26: "I-LOCAL_HOSPEDAGEM"   # Continuation of accommodation location
            }
            # Create reverse mapping from label to ID
            _model.config.label2id = {v: k for k, v in _model.config.id2label.items()}
        
        # Move model to the specified device (GPU/CPU) and set to evaluation mode
        _model.to(device)  # Transfer model to GPU if available
        _model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        
        print(f"Modelo carregado no dispositivo: {device}")  # Confirmation message
    
    return _processor, _model

def pdf_to_images(pdf_path: str, dpi: int = PDF_DPI) -> List[Image.Image]:
    """
    Converte PDF em lista de imagens PIL, uma por página.
    
    Args:
        pdf_path: Path to the PDF file
        dpi: Resolution for the converted images
        
    Returns:
        List of PIL Image objects, one per page
    """
    print(f"Convertendo PDF para imagens: {pdf_path}")
    # Convert PDF to images using pdf2image library
    # DPI (dots per inch) from config.py controls the resolution of the output images
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"Convertidas {len(images)} páginas")
    return images  # Returns a list of PIL Image objects, one per page

def ocr_image(image: Image.Image, tesseract_lang=TESSERACT_LANG) -> List[Dict]:
    """
    Realiza OCR na imagem e retorna lista de palavras com bounding boxes.
    Cada palavra é um dict com 'text' e 'bbox' (x0, y0, x1, y1).
    
    Args:
        image: PIL Image object to process
        tesseract_lang: Language for OCR (default from config)
        
    Returns:
        List of dictionaries with text and bounding box information
    """
    # Use Tesseract OCR to extract text and positioning data from the image
    # The configuration is imported from config.py
    # Note: Originally intended to use Portuguese ('por') but using English ('eng')
    # because Portuguese language data might not be installed in Tesseract
    data = pytesseract.image_to_data(
        image, 
        lang=tesseract_lang,
        config=TESSERACT_CONFIG,
        output_type=pytesseract.Output.DICT
    )
    
    words = []
    # Process each word detected by Tesseract
    for i, word in enumerate(data['text']):
        # Skip empty words
        if word.strip() != "":
            # Extract position and dimensions of the word
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            # Create bounding box coordinates [x0, y0, x1, y1]
            bbox = [x, y, x + w, y + h]
            # Add word and its bounding box to the results
            words.append({"text": word, "bbox": bbox})
    return words  # List of dictionaries with text and position information

def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """
    Normaliza bbox para escala 0-1000 (LayoutLM padrão).
    
    Args:
        bbox: Bounding box coordinates [x0, y0, x1, y1]
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        Normalized bounding box coordinates in 0-1000 scale
    """
    # LayoutLMv3 expects bounding boxes in a normalized format from 0-1000
    # This converts pixel coordinates to this normalized scale
    return [
        int(1000 * (bbox[0] / width)),   # Normalize x0 (left)
        int(1000 * (bbox[1] / height)),  # Normalize y0 (top)
        int(1000 * (bbox[2] / width)),   # Normalize x1 (right)
        int(1000 * (bbox[3] / height)),  # Normalize y1 (bottom)
    ]

def extract_tables_from_image(image: Image.Image, tesseract_lang=TESSERACT_LANG) -> List[Dict]:
    """
    Extrai tabelas de uma imagem usando a biblioteca img2table.
    
    Args:
        image: PIL Image object to process
        tesseract_lang: Language for OCR (default from config)
        
    Returns:
        List of dictionaries with table data
    """
    try:
        # Convert PIL Image to img2table Image format
        img2table_image = Img2TableImage(image)
        
        # Configure OCR engine
        ocr_engine = TesseractOCR(lang=tesseract_lang)
        
        # Extract tables from the image
        tables = img2table_image.extract_tables(
            ocr=ocr_engine,
            implicit_rows=True,  # Detect implicit rows
            borderless_tables=True,  # Detect tables without borders
            min_confidence=50  # Minimum confidence for table detection
        )
        
        # Process extracted tables into a more usable format
        processed_tables = []
        for i, table in enumerate(tables):
            # Convert table to pandas DataFrame
            df = table.df
            
            # Add table metadata
            processed_table = {
                "table_id": i,
                "bbox": table.bbox,  # Bounding box of the table
                "data": df.to_dict(orient="records"),  # Table data as list of dictionaries
                "shape": df.shape,  # Table dimensions (rows, columns)
                "headers": df.columns.tolist() if not df.empty else []  # Column headers
            }
            processed_tables.append(processed_table)
            
        print(f"Extraídas {len(processed_tables)} tabelas da imagem")
        return processed_tables
        
    except Exception as e:
        print(f"Erro ao extrair tabelas: {str(e)}")
        return []  # Return empty list in case of error

def extract_layoutlm_entities(image: Image.Image, words: List[Dict], device=DEVICE) -> List[Dict]:
    """
    Usa LayoutLMv3 para classificar tokens e extrair entidades.
    Retorna lista de entidades com texto e tipo.
    
    Args:
        image: PIL Image object to process
        words: List of dictionaries with text and bounding box information
        device: Device to run the model on (CPU or GPU)
        
    Returns:
        List of dictionaries with entity label and text
    """
    # Initialize the model if not already done
    processor, model = initialize_model(device=device)
    
    # Get image dimensions for bounding box normalization
    width, height = image.size

    # Extract text and bounding boxes from the words list
    texts = [w["text"] for w in words]  # List of all detected words
    boxes = [normalize_bbox(w["bbox"], width, height) for w in words]  # Normalized bounding boxes

    # Check if there are any words to process
    # If the image has no text or OCR failed, return empty list
    if not texts or not boxes:
        return []
        
    try:
        # Limit the number of tokens to avoid memory errors
        # LayoutLMv3 has a maximum sequence length limit (typically 512)
        max_tokens = 512
        if len(texts) > max_tokens:
            print(f"Aviso: Limitando de {len(texts)} para {max_tokens} tokens")
            texts = texts[:max_tokens]  # Truncate texts to maximum length
            boxes = boxes[:max_tokens]  # Truncate boxes to match texts
            
        # Process the image and tokens with the LayoutLMv3 processor
        # This prepares the inputs in the format expected by the model
        encoding = processor(image, texts, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
        # Move all tensors to the selected device (GPU/CPU)
        encoding = {k: v.to(device) for k, v in encoding.items()}

        # Run inference with the model
        # torch.no_grad() disables gradient calculation for inference, saving memory
        with torch.no_grad():
            outputs = model(**encoding)  # Forward pass through the model
        
        # Get the logits (raw prediction scores) from the output
        logits = outputs.logits
        # Get the predicted class for each token (argmax along the class dimension)
        # Convert to list for easier processing
        predictions = logits.argmax(-1).squeeze().tolist()
        
        # Ensure predictions is a list (it might be a single value if there's only one token)
        if not isinstance(predictions, list):
            predictions = [predictions]
            
        # Check if the model has an id2label mapping
        # This maps prediction indices to human-readable entity labels
        if not hasattr(model.config, 'id2label') or not model.config.id2label:
            print("Aviso: Modelo sem mapeamento id2label. Usando mapeamento padrão.")
            # Fallback to a basic label set if model doesn't have labels defined
            id2label = {0: "O", 1: "B-VALOR", 2: "I-VALOR", 3: "B-DATA", 4: "I-DATA"}
        else:
            # Use the model's predefined label mapping
            id2label = model.config.id2label
            
        # Map numeric predictions to text labels
        labels = []
        for p in predictions[:len(texts)]:  # Only process up to the number of input texts
            # Ensure the prediction index exists in the mapping
            if isinstance(p, int) and p in id2label:
                labels.append(id2label[p])  # Convert prediction ID to label
            else:
                labels.append("O")  # Default to 'Outside' label if not found

        # Group words by entity to form coherent entity spans
        # This combines consecutive tokens with the same entity label
        entities = []
        current_entity = None
        
        # Iterate through each word and its predicted label
        for word, label in zip(texts, labels):
            if label != "O":  # Skip tokens that aren't part of any entity
                # If we're continuing the same entity type
                if current_entity and current_entity["label"] == label:
                    current_entity["text"] += " " + word  # Append word to current entity
                else:
                    # If we have a previous entity, add it to results
                    if current_entity:
                        entities.append(current_entity)
                    # Start a new entity
                    current_entity = {"label": label, "text": word}
            else:  # When we hit a non-entity token
                # If we were tracking an entity, finish it
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Don't forget to add the last entity if we have one
        if current_entity:
            entities.append(current_entity)

        return entities  # Return the list of extracted entities
        
    except Exception as e:
        # Handle any errors that occur during processing
        print(f"Erro ao processar entidades com LayoutLMv3: {str(e)}")
        # Return an empty list in case of error to allow the program to continue
        return []

def extract_monetary_values(text: str) -> List[str]:
    """
    Extrai valores monetários no formato brasileiro (ex: R$ 1.234,56) do texto.
    
    Args:
        text: Text to search for monetary values
        
    Returns:
        List of monetary value strings found in the text
    """
    # Use the monetary value regex pattern from config.py
    # The pattern matches Brazilian currency format like R$ 1.234,56
    
    # Find all matches of the pattern in the text
    return re.findall(REGEX_PATTERNS["valor_monetario"], text)

def extract_dates(text: str) -> List[str]:
    """
    Extrai datas no formato DD/MM/YYYY ou DD-MM-YYYY.
    
    Args:
        text: Text to search for dates
        
    Returns:
        List of date strings found in the text
    """
    # Use the date regex pattern from config.py
    # The pattern matches date formats like DD/MM/YYYY or DD-MM-YYYY
    
    # Find all matches of the pattern in the text
    return re.findall(REGEX_PATTERNS["data"], text)

def extract_expense_categories(text: str) -> List[str]:
    """
    Identifica possíveis categorias de despesas baseado em palavras-chave.
    
    Args:
        text: Text to analyze for expense categories
        
    Returns:
        List of identified expense categories
    """
    # Use expense categories from config.py
    # The dictionary maps category names to lists of related keywords
    
    found_categories = []  # List to store identified categories
    text_lower = text.lower()  # Convert text to lowercase for case-insensitive matching
    
    # Iterate through each category and its keywords from config
    for category, keywords in EXPENSE_CATEGORIES.items():
        for keyword in keywords:
            # If any keyword is found in the text, add the category and move to next category
            if keyword in text_lower:
                found_categories.append(category)
                break  # Stop checking other keywords once a match is found
                
    # Return found categories or default to "não classificado" if none found
    return found_categories if found_categories else ["não classificado"]

def summarize_expense_report(pdf_path: str) -> Dict:
    """
    Processa um PDF de relatório de despesas e extrai informações estruturadas.
    Retorna um dicionário com os dados extraídos.
    
    Args:
        pdf_path: Path to the PDF file to process
        
    Returns:
        Dictionary with extracted information including text, entities, values, dates, categories, and tables
    """
    # Convert PDF to list of PIL images
    images = pdf_to_images(pdf_path)
    
    # Initialize empty lists to store extracted data
    all_text = []  # Combined text from all pages
    all_entities = []  # All entities extracted from all pages
    all_tables = []  # All tables extracted from all pages
    
    # Process each page of the PDF
    for i, image in enumerate(images):
        try:
            # Perform OCR on the image to get text and bounding boxes
            words = ocr_image(image)
            
            # Extract text from words list
            page_text = " ".join([w["text"] for w in words])
            all_text.append(page_text)  # Add page text to combined text
            
            # Extract entities using LayoutLM model
            entities = extract_layoutlm_entities(image, words)
            all_entities.extend(entities)  # Add page entities to all entities
            
            # Extract tables using img2table
            tables = extract_tables_from_image(image)
            
            # Add page number to each table for reference
            for table in tables:
                table["page_num"] = i + 1
            
            all_tables.extend(tables)  # Add page tables to all tables
            
            print(f"Processada página {i+1}/{len(images)} - {len(words)} palavras, {len(entities)} entidades, {len(tables)} tabelas")
        except Exception as e:
            print(f"Erro ao processar página {i+1}: {str(e)}")
    
    # Combine all text from all pages
    full_text = "\n".join(all_text)
    
    # Extract specific information using regex patterns
    monetary_values = extract_monetary_values(full_text)
    dates = extract_dates(full_text)
    categories = extract_expense_categories(full_text)
    
    # Create a structured summary of the extracted information
    summary = {
        "text": full_text,  # Full text from all pages
        "entities": all_entities,  # All extracted entities
        "monetary_values": monetary_values,  # Extracted monetary values
        "dates": dates,  # Extracted dates
        "categories": categories,  # Identified expense categories
        "tables": all_tables,  # Extracted tables
        "pdf_path": pdf_path,  # Original PDF path
        "num_pages": len(images),  # Number of pages in the PDF
        "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Timestamp
    }
    
    return summary

# Excel export functionality has been removed
        
def generate_text_report(summary: Dict, output_path: str) -> None:
    """
    Gera um relatório de texto com as conclusões do modelo LayoutLMv3, valores identificados por regex e tabelas extraídas.
    
    Args:
        summary: Dictionary with extracted information
        output_path: Path to save the text report
    """
    # Open a text file for writing with UTF-8 encoding to support Portuguese characters
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write the report header with a title banner
        f.write("=" * 80 + "\n")  # Separator line
        f.write("RELATÓRIO DE DESCOBERTA - R2BIT TRIPAUDIT\n")  # Title
        f.write("=" * 80 + "\n\n")  # Separator line and spacing
        
        # Add timestamp and basic statistics
        from datetime import datetime
        f.write(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")  # Current date/time
        f.write(f"Total de páginas analisadas: {summary.get('num_pages', 0)}\n\n")  # Page count
        
        # PART 1: Results from the LayoutLMv3 model
        f.write("PARTE 1: CONCLUSÕES DO MODELO LAYOUTLMV3\n")  # Section header
        f.write("-" * 50 + "\n\n")  # Section separator
        
        # Process entities
        entities = summary.get("entities", [])
        
        if entities:  # If we found any entities
            # Group entities by their label type for better organization
            entity_types = {}
            for entity in entities:
                label = entity.get("label", "unknown")
                text = entity.get("text", "")
                
                if label not in entity_types:
                    entity_types[label] = []  # Initialize empty list for this label type
                entity_types[label].append(text)  # Add this occurrence
            
            # Write each entity type and its occurrences
            for label, occurrences in entity_types.items():
                f.write(f"Entidade: {label}\n")  # Entity type header
                for text in occurrences:
                    # List each occurrence
                    f.write(f"  - {text}\n")
                f.write("\n")  # Add spacing between entity types
        else:
            # If no entities were found
            f.write("Nenhuma entidade foi extraída pelo modelo LayoutLMv3.\n\n")
        
        # PART 2: Values identified using regular expressions
        f.write("\nPARTE 2: VALORES IDENTIFICADOS POR EXPRESSÕES REGULARES\n")  # Section header
        f.write("-" * 50 + "\n\n")  # Section separator
        
        # Section for monetary values
        f.write("Valores monetários:\n")  # Subsection header
        monetary_values = summary.get("monetary_values", [])
        
        if monetary_values:  # If monetary values were found
            # Write each monetary value
            for value in monetary_values:
                f.write(f"  - {value}\n")
        else:
            # If no monetary values were found
            f.write("  Nenhum valor monetário encontrado.\n")
        
        # Section for dates
        f.write("\nDatas encontradas:\n")  # Subsection header
        dates = summary.get("dates", [])
        
        if dates:  # If dates were found
            # Write each date
            for date in dates:
                f.write(f"  - {date}\n")
        else:
            # If no dates were found
            f.write("  Nenhuma data encontrada.\n")
        
        # Section for categories
        f.write("\nCategorias identificadas:\n")  # Subsection header
        categories = summary.get("categories", [])
        
        if categories:
            # Write each category
            for category in categories:
                f.write(f"  - {category}\n")
        else:
            f.write("  Nenhuma categoria identificada.\n")
        
        # PART 3: Tables extracted using img2table
        f.write("\nPARTE 3: TABELAS EXTRAÍDAS COM IMG2TABLE\n")  # Section header
        f.write("-" * 50 + "\n\n")  # Section separator
        
        # Section for tables
        tables = summary.get("tables", [])
        
        if tables:  # If tables were found
            f.write(f"Total de tabelas encontradas: {len(tables)}\n\n")
            
            # Write summary for each table
            for i, table in enumerate(tables):
                page_num = table.get("page_num", 0)
                table_id = table.get("table_id", i)
                rows, cols = table.get("shape", (0, 0))
                headers = table.get("headers", [])
                
                f.write(f"Tabela {i+1}:\n")
                f.write(f"  - Página: {page_num}\n")
                f.write(f"  - Dimensões: {rows} linhas x {cols} colunas\n")
                
                if headers:
                    f.write(f"  - Cabeçalhos: {', '.join(headers)}\n")
                
                f.write("\n")
        else:
            # If no tables were found
            f.write("  Nenhuma tabela encontrada.\n")
        
        # Final summary section
        f.write("\n" + "=" * 80 + "\n")  # Separator line
        f.write(f"RELATÓRIO COMPLETO DISPONÍVEL NO ARQUIVO EXCEL\n")  # Reference to Excel file
        f.write("=" * 80 + "\n")  # Separator line
    
    # Confirmation message
    print(f"Relatório de texto exportado para: {output_path}")


def main():
    """
    Função principal que processa argumentos de linha de comando e executa o fluxo de extração.
    """
    # Import required modules for command-line argument parsing and file path operations
    import argparse
    import os.path
    
    try:
        # Set up command-line argument parser with descriptions
        parser = argparse.ArgumentParser(description="Extrai informações de relatórios de despesas em PDF")
        
        # Define command-line arguments
        parser.add_argument("pdf_path", 
                          help="Caminho para o arquivo PDF do relatório de despesas")  # Required positional argument
        parser.add_argument("--output", "-o", 
                          help="Caminho para salvar o relatório Excel (opcional)")  # Optional Excel output path
        parser.add_argument("--dpi", type=int, default=PDF_DPI, 
                          help=f"DPI para conversão do PDF (padrão: {PDF_DPI})")  # DPI for PDF conversion
        parser.add_argument("--text-report", "-t",
                          help="Caminho para salvar o relatório de texto (opcional)")  # Optional text report path
        
        # Parse the command-line arguments
        args = parser.parse_args()
        
        # Validate that the input PDF file exists
        if not os.path.exists(args.pdf_path):
            print(f"Arquivo {args.pdf_path} não encontrado.")
            return 1  # Return error code
    
        # Set default output paths if not specified
        if not args.output:
            # Create default Excel output filename based on input filename
            base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
            args.output = f"{base_name}_relatorio.xlsx"
        
        if not args.text_report:
            # Create default text report filename
            base_name = os.path.splitext(os.path.basename(args.pdf_path))[0]
            args.text_report = f"{base_name}_discovery_summary.txt"
        
        # Process the PDF and extract all information
        print(f"Processando o arquivo PDF: {args.pdf_path}")
        summary = summarize_expense_report(args.pdf_path)
    
        # Display a summary of the findings in the console
        print(f"\nResumo do relatório de despesas:")
        print(f"Total de páginas: {summary.get('num_pages', 0)}")  # Number of pages processed
        print(f"Total de entidades: {len(summary.get('entities', []))}")  # Number of entities
        print(f"Valores monetários: {len(summary.get('monetary_values', []))}")  # Number of monetary values
        print(f"Datas encontradas: {len(summary.get('dates', []))}")  # Number of dates
        print(f"Categorias identificadas: {', '.join(summary.get('categories', []))}")  # All categories
    
        # Display some extracted entities if available
        entities = summary.get('entities', [])
        if entities:
            print("\nExemplos de entidades extraídas:")
            for i, entity in enumerate(entities[:5]):  # Show first 5 entities
                print(f"  - {entity.get('label', '')}: {entity.get('text', '')}")
            if len(entities) > 5:
                print(f"  ... e mais {len(entities) - 5} entidades")
    
        # Excel export functionality has been removed
        print(f"A exportação para Excel foi removida do código")
        
        # Generate a text report with the findings
        print(f"Gerando relatório de texto: {args.text_report}")
        generate_text_report(summary, args.text_report)
    
        print("\nProcessamento concluído com sucesso!")
        
        return 0  # Success exit code
        
    except Exception as e:
        # Handle any exceptions that occur during processing
        print(f"Erro durante o processamento: {str(e)}")
        # Print the full stack trace for debugging
        import traceback
        traceback.print_exc()
        return 1  # Error exit code


class ExpenseReportExtractor:
    """
    Class for extracting information from expense reports using LayoutLMv3.
    This class wraps the functional API to provide an object-oriented interface.
    """
    
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, tesseract_lang=TESSERACT_LANG):
        """
        Initialize the expense report extractor with model and configuration.
        
        Args:
            model_name: Name or path of the pre-trained model
            device: Device to run the model on (CPU or GPU)
            tesseract_lang: Language for OCR
        """
        self.model_name = model_name
        self.device = device
        self.tesseract_lang = tesseract_lang
        # Initialize the model when the extractor is created
        self.processor, self.model = initialize_model(model_name, device, tesseract_lang)
    
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """
        Process a PDF expense report and extract information.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Dictionary with extracted information
        """
        return summarize_expense_report(pdf_path)
    
    def generate_text_report(self, summary: Dict, output_path: str) -> None:
        """
        Generate a text report from the extracted information.
        
        Args:
            summary: Dictionary with extracted information
            output_path: Path to save the text report
        """
        generate_text_report(summary, output_path)


if __name__ == "__main__":
    # This block executes when the script is run directly (not imported)
    # Call the main function and use its return value as the exit code
    exit(main())
