import pdfplumber

def extract_text_from_pdf(file_path):
    """
    Extracts clean text from all pages of a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    full_text = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    full_text.append(f"\n\n---Page{i + 1} ---\n{text.strip()}")
                else:
                    print(f"Warning Page {i + 1} is empty!")
    
    except Exception as e:
        print(f"[Error] failed to parse: {e}")
        return " "
    
    return "\n".join(full_text)
