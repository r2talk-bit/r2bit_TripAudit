# Standard library imports for file operations and regex
import os
import re

# Deep learning and OCR libraries
import torch  # PyTorch for deep learning operations
import pytesseract  # Interface to Tesseract OCR engine
import pandas as pd  # Data manipulation and analysis
from pdf2image import convert_from_path  # Convert PDF to images
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification  # Document understanding model
from PIL import Image  # Python Imaging Library for image processing
from typing import List, Dict  # Type hints for better code documentation

# Import global configurations from config.py
from config import MODEL_NAME, TESSERACT_LANG, TESSERACT_CONFIG, PDF_DPI, EXPENSE_CATEGORIES, REGEX_PATTERNS

# Set device for PyTorch operations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

class ExpenseReportExtractor:
    def __init__(self, model_name=MODEL_NAME, device=DEVICE, tesseract_lang=TESSERACT_LANG):
        """Inicializa o extrator com o modelo LayoutLMv3 e configurações."""
        # Store configuration parameters
        self.device = device  # CPU or GPU for model inference
        self.tesseract_lang = tesseract_lang  # Language for OCR (default is English)
        
        # Initialize the LayoutLMv3 model and processor
        # apply_ocr=False because we'll handle OCR separately with Tesseract
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        # Load the token classification model for entity extraction
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        
        # Define detailed label configuration for expense reports if not already present
        # This is a comprehensive labeling system with 23 different entity types
        # Each entity type has a Beginning (B-) and Inside (I-) tag following the BIO tagging scheme
        if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
            self.model.config.id2label = {
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
            self.model.config.label2id = {v: k for k, v in self.model.config.id2label.items()}
        
        # Move model to the specified device (GPU/CPU) and set to evaluation mode
        self.model.to(self.device)  # Transfer model to GPU if available
        self.model.eval()  # Set model to evaluation mode (disables dropout, etc.)
        
        print(f"Modelo carregado no dispositivo: {self.device}")  # Confirmation message

    def pdf_to_images(self, pdf_path: str, dpi: int = PDF_DPI) -> List[Image.Image]:
        """Converte PDF em lista de imagens PIL, uma por página."""
        print(f"Convertendo PDF para imagens: {pdf_path}")
        # Convert PDF to images using pdf2image library
        # DPI (dots per inch) from config.py controls the resolution of the output images
        images = convert_from_path(pdf_path, dpi=dpi)
        print(f"Convertidas {len(images)} páginas")
        return images  # Returns a list of PIL Image objects, one per page

    def ocr_image(self, image: Image.Image) -> List[Dict]:
        """
        Realiza OCR na imagem e retorna lista de palavras com bounding boxes.
        Cada palavra é um dict com 'text' e 'bbox' (x0, y0, x1, y1).
        """
        # Use Tesseract OCR to extract text and positioning data from the image
        # The configuration is imported from config.py
        # Note: Originally intended to use Portuguese ('por') but using English ('eng')
        # because Portuguese language data might not be installed in Tesseract
        data = pytesseract.image_to_data(
            image, 
            lang=self.tesseract_lang,
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

    def normalize_bbox(self, bbox: List[int], width: int, height: int) -> List[int]:
        """Normaliza bbox para escala 0-1000 (LayoutLM padrão)."""
        # LayoutLMv3 expects bounding boxes in a normalized format from 0-1000
        # This converts pixel coordinates to this normalized scale
        return [
            int(1000 * (bbox[0] / width)),   # Normalize x0 (left)
            int(1000 * (bbox[1] / height)),  # Normalize y0 (top)
            int(1000 * (bbox[2] / width)),   # Normalize x1 (right)
            int(1000 * (bbox[3] / height)),  # Normalize y1 (bottom)
        ]

    def extract_layoutlm_entities(self, image: Image.Image, words: List[Dict]) -> List[Dict]:
        """
        Usa LayoutLMv3 para classificar tokens e extrair entidades.
        Retorna lista de entidades com texto e tipo.
        """
        # Get image dimensions for bounding box normalization
        width, height = image.size

        # Extract text and bounding boxes from the words list
        texts = [w["text"] for w in words]  # List of all detected words
        boxes = [self.normalize_bbox(w["bbox"], width, height) for w in words]  # Normalized bounding boxes

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
            encoding = self.processor(image, texts, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
            # Move all tensors to the selected device (GPU/CPU)
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Run inference with the model
            # torch.no_grad() disables gradient calculation for inference, saving memory
            with torch.no_grad():
                outputs = self.model(**encoding)  # Forward pass through the model
            
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
            if not hasattr(self.model.config, 'id2label') or not self.model.config.id2label:
                print("Aviso: Modelo sem mapeamento id2label. Usando mapeamento padrão.")
                # Fallback to a basic label set if model doesn't have labels defined
                id2label = {0: "O", 1: "B-VALOR", 2: "I-VALOR", 3: "B-DATA", 4: "I-DATA"}
            else:
                # Use the model's predefined label mapping
                id2label = self.model.config.id2label
                
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

    def extract_monetary_values(self, text: str) -> List[str]:
        """Extrai valores monetários no formato brasileiro (ex: R$ 1.234,56) do texto."""
        # Use the monetary value regex pattern from config.py
        # The pattern matches Brazilian currency format like R$ 1.234,56
        
        # Find all matches of the pattern in the text
        return re.findall(REGEX_PATTERNS["valor_monetario"], text)

    def extract_dates(self, text: str) -> List[str]:
        """Extrai datas no formato DD/MM/YYYY ou DD-MM-YYYY."""
        # Use the date regex pattern from config.py
        # The pattern matches date formats like DD/MM/YYYY or DD-MM-YYYY
        
        # Find all matches of the pattern in the text
        return re.findall(REGEX_PATTERNS["data"], text)

    def extract_expense_categories(self, text: str) -> List[str]:
        """
        Identifica possíveis categorias de despesas baseado em palavras-chave.
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

    def summarize_expense_report(self, pdf_path: str) -> Dict:
        """
        Função principal que recebe o caminho do PDF, processa e retorna resumo.
        """
        # Convert the PDF to a list of images (one per page)
        images = self.pdf_to_images(pdf_path)
        
        # Initialize the summary dictionary to store all extracted information
        summary = {
            "pages": len(images),           # Total number of pages in the PDF
            "expenses": [],                # List to store per-page expense details
            "total_value": 0.0,           # Running sum of all monetary values
            "dates": set(),               # Set to store unique dates (prevents duplicates)
            "categories": {},             # Dictionary to count occurrences of each category
        }

        # Process each page of the PDF individually
        for page_num, image in enumerate(images, start=1):
            print(f"Processando página {page_num}/{len(images)}")
            
            # Step 1: Perform OCR to extract text and positions from the image
            words = self.ocr_image(image)
            
            # Step 2: Use LayoutLMv3 to identify and extract structured entities
            entities = self.extract_layoutlm_entities(image, words)

            # Step 3: Create a full text representation of the page for regex-based extraction
            full_text = " ".join([w["text"] for w in words])
            
            # Step 4: Extract specific information using regular expressions and keyword matching
            values = self.extract_monetary_values(full_text)      # Extract monetary values
            dates = self.extract_dates(full_text)                # Extract dates
            categories = self.extract_expense_categories(full_text)  # Identify expense categories

            # Convert extracted monetary values from string format to float for calculations
            values_float = []
            for v in values:
                # Clean and convert Brazilian currency format to standard float format:
                # 1. Remove currency symbol (R$)
                # 2. Remove spaces
                # 3. Remove dots (thousand separators)
                # 4. Replace comma with dot (decimal separator)
                clean_v = v.replace("R$", "").replace(" ", "").replace(".", "").replace(",", ".")
                try:
                    values_float.append(float(clean_v))  # Convert to float
                except:
                    pass  # Skip values that can't be converted to float

            # Update the category counts in the summary
            for category in categories:
                if category in summary["categories"]:
                    summary["categories"][category] += 1  # Increment existing category count
                else:
                    summary["categories"][category] = 1   # Initialize new category count

            # Add any found dates to the set of dates (automatically handles duplicates)
            for date in dates:
                summary["dates"].add(date)  # Using a set ensures each date is only counted once

            # Create a dictionary with all the information extracted from this page
            page_expenses = {
                "page": page_num,                 # Page number
                "entities": entities,             # Structured entities from LayoutLMv3
                "values_found": values,           # Raw monetary value strings
                "values_float": values_float,     # Converted float values
                "sum_values": sum(values_float),  # Sum of all values on this page
                "dates": dates,                   # Dates found on this page
                "categories": categories          # Categories identified on this page
            }
            
            # Add this page's data to the summary
            summary["expenses"].append(page_expenses)
            
            # Update the running total of all monetary values
            summary["total_value"] += sum(values_float)

        # Convert the set of dates to a list for easier serialization (JSON doesn't support sets)
        summary["dates"] = list(summary["dates"])
        
        # Return the complete summary of the expense report
        return summary

    def export_to_excel(self, summary: Dict, output_path: str) -> None:
        """
        Exporta o resumo para um arquivo Excel.
        """
        # Create a DataFrame for the general summary information
        # This will be the first sheet in the Excel file with high-level metrics
        general_data = {
            "Total de Páginas": [summary["pages"]],                    # Number of pages processed
            "Valor Total": [f"R$ {summary['total_value']:.2f}"],        # Total monetary value with currency formatting
            "Datas Encontradas": [", ".join(summary["dates"])]         # All unique dates joined as a comma-separated string
        }
        
        # Convert general data to a pandas DataFrame
        general_df = pd.DataFrame(general_data)
        
        # Create a DataFrame for category information
        # This will show the distribution of expense categories
        categories_data = {
            "Categoria": list(summary["categories"].keys()),    # Category names
            "Contagem": list(summary["categories"].values())   # Count of occurrences for each category
        }
        
        # Convert categories data to a pandas DataFrame
        categories_df = pd.DataFrame(categories_data)
        
        # Create a DataFrame for detailed information about each value found
        # This provides a line-by-line breakdown of all monetary values
        details_data = []
        for page in summary["expenses"]:
            # For each monetary value found on the page
            for i, value in enumerate(page["values_found"]):
                # Get the corresponding float value if available
                float_value = page["values_float"][i] if i < len(page["values_float"]) else None
                
                # Create a record for this value with contextual information
                details_data.append({
                    "Página": page["page"],                        # Page number where value was found
                    "Valor Encontrado": value,                    # Original text of the monetary value
                    "Valor Numérico": float_value,               # Converted numeric value
                    "Datas": ", ".join(page["dates"]),           # Dates found on the same page
                    "Categorias": ", ".join(page["categories"])  # Categories identified on the same page
                })
        
        # Convert details data to a DataFrame, or create an empty one if no data
        details_df = pd.DataFrame(details_data) if details_data else pd.DataFrame()
        
        # Write all DataFrames to a single Excel file with multiple sheets
        with pd.ExcelWriter(output_path) as writer:
            # First sheet: General summary
            general_df.to_excel(writer, sheet_name="Resumo Geral", index=False)
            # Second sheet: Categories breakdown
            categories_df.to_excel(writer, sheet_name="Categorias", index=False)
            # Third sheet: Detailed values (only if we have data)
            if not details_df.empty:
                details_df.to_excel(writer, sheet_name="Detalhes", index=False)
        
        # Confirmation message
        print(f"Relatório exportado para: {output_path}")
        
    def generate_text_report(self, summary: Dict, output_path: str) -> None:
        """
        Gera um relatório de texto com as conclusões do modelo LayoutLMv3 e valores identificados por regex.
        O relatório é dividido em duas partes:
        1. Conclusões do modelo LayoutLMv3
        2. Valores identificados por expressões regulares
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
            f.write(f"Total de páginas analisadas: {summary['pages']}\n\n")  # Page count
            
            # PART 1: Results from the LayoutLMv3 model
            f.write("PARTE 1: CONCLUSÕES DO MODELO LAYOUTLMV3\n")  # Section header
            f.write("-" * 50 + "\n\n")  # Section separator
            
            # Collect all entities extracted by the model from all pages
            all_entities = []
            for page in summary["expenses"]:
                for entity in page["entities"]:
                    # Store as tuples of (page_number, entity_label, entity_text)
                    all_entities.append((page["page"], entity["label"], entity["text"]))
            
            if all_entities:  # If we found any entities
                # Group entities by their label type for better organization
                entity_types = {}
                for page, label, text in all_entities:
                    if label not in entity_types:
                        entity_types[label] = []  # Initialize empty list for this label type
                    entity_types[label].append((page, text))  # Add this occurrence
                
                # Write each entity type and its occurrences
                for label, occurrences in entity_types.items():
                    f.write(f"Entidade: {label}\n")  # Entity type header
                    for page, text in occurrences:
                        # List each occurrence with page number
                        f.write(f"  - Página {page}: {text}\n")
                    f.write("\n")  # Add spacing between entity types
            else:
                # If no entities were found
                f.write("Nenhuma entidade foi extraída pelo modelo LayoutLMv3.\n\n")
            
            # PART 2: Values identified using regular expressions
            f.write("\nPARTE 2: VALORES IDENTIFICADOS POR EXPRESSÕES REGULARES\n")  # Section header
            f.write("-" * 50 + "\n\n")  # Section separator
            
            # Section for monetary values
            f.write("Valores monetários:\n")  # Subsection header
            
            # Collect all monetary values from all pages
            all_values = []
            for page in summary["expenses"]:
                for i, value in enumerate(page["values_found"]):
                    # Get the corresponding float value if available
                    float_value = page["values_float"][i] if i < len(page["values_float"]) else None
                    # Store as tuples of (page_number, original_value, numeric_value)
                    all_values.append((page["page"], value, float_value))
            
            if all_values:  # If monetary values were found
                # Write each monetary value with its page number and numeric conversion
                for page, value, float_value in all_values:
                    f.write(f"  - Página {page}: {value} (valor numérico: {float_value})\n")
            else:
                # If no monetary values were found
                f.write("  Nenhum valor monetário encontrado.\n")
            
            # Section for dates
            f.write("\nDatas encontradas:\n")  # Subsection header
            
            # Collect all dates from all pages
            all_dates = []
            for page in summary["expenses"]:
                for date in page["dates"]:
                    # Store as tuples of (page_number, date)
                    all_dates.append((page["page"], date))
            
            if all_dates:  # If dates were found
                # Write each date with its page number
                for page, date in all_dates:
                    f.write(f"  - Página {page}: {date}\n")
            else:
                # If no dates were found
                f.write("  Nenhuma data encontrada.\n")
            
            # Section for categories
            f.write("\nCategorias identificadas:\n")  # Subsection header
            
            # Write each category with its occurrence count
            for category, count in summary["categories"].items():
                f.write(f"  - {category}: {count} ocorrência(s)\n")
            
            # Final summary section with total value
            f.write("\n" + "=" * 80 + "\n")  # Separator line
            f.write(f"VALOR TOTAL EXTRAÍDO: R$ {summary['total_value']:.2f}\n")  # Total value with currency formatting
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
        
        # Initialize the expense report extractor
        extractor = ExpenseReportExtractor()
        
        # Process the PDF and extract all information
        print(f"Processando o arquivo PDF: {args.pdf_path}")
        resumo = extractor.summarize_expense_report(args.pdf_path)
    
        # Display a summary of the findings in the console
        print(f"\nResumo do relatório de despesas:")
        print(f"Total de páginas: {resumo['pages']}")  # Number of pages processed
        print(f"Valor total extraído: R$ {resumo['total_value']:.2f}")  # Total monetary value
        print(f"Datas encontradas: {', '.join(resumo['dates'])}")  # All unique dates
        print(f"Categorias identificadas: {', '.join(resumo['categories'].keys())}")  # All categories
    
        # Display detailed information for each page
        print("\nDetalhes por página:")
        for page in resumo["expenses"]:
            print(f"Página {page['page']}:")  # Page number
            print(f"  Valores encontrados: {page['values_found']}")  # Raw monetary values
            print(f"  Soma dos valores: R$ {page['sum_values']:.2f}")  # Sum of values on this page
            print(f"  Datas: {page['dates']}")  # Dates found on this page
            print(f"  Categorias: {page['categories']}")  # Categories identified on this page
            print(f"  Entidades extraídas:")  # Entities extracted by LayoutLMv3
            for ent in page["entities"]:
                # Display each entity with its label and text
                print(f"    - {ent['label']}: {ent['text']}")
    
        # Export the results to Excel format
        print(f"Exportando relatório para Excel: {args.output}")
        extractor.export_to_excel(resumo, args.output)
        
        # Generate a text report with the findings
        print(f"Gerando relatório de texto: {args.text_report}")
        extractor.generate_text_report(resumo, args.text_report)
    
        print("\nProcessamento concluído com sucesso!")
        
        return 0  # Success exit code
        
    except Exception as e:
        # Handle any exceptions that occur during processing
        print(f"Erro durante o processamento: {str(e)}")
        # Print the full stack trace for debugging
        import traceback
        traceback.print_exc()
        return 1  # Error exit code


if __name__ == "__main__":
    # This block executes when the script is run directly (not imported)
    # Call the main function and use its return value as the exit code
    exit(main())
