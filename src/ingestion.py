import pymupdf
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# create data ingestion class
# defining a class for extracting text from documents
class PdfExtractors:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text_from_pdf(self) -> str:
        """Extract text directly from PDF pages."""
        try:
            doc = pymupdf.open(self.pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "".join(text_parts)
        except FileNotFoundError:
            return f"Error: PDF file '{self.pdf_path}' not found."
        except Exception as e:
            return f"Error extracting text from PDF: {str(e)}"

    def extract_text_from_images(self) -> str:
        """Extract text from images in a PDF using OCR."""
        try:
            images = convert_from_path(self.pdf_path)
            text_from_images = []
            for img in images:
                text = pytesseract.image_to_string(img)
                text_from_images.append(text)
            return "\n".join(text_from_images)
        except FileNotFoundError:
            return f"Error: PDF file '{self.pdf_path}' not found."
        except Exception as e:
            return f"Error extracting text from images: {str(e)}"

    def extract_from_jpg(self, image_path: str) -> str:
        """Extract text from a single JPG image using OCR."""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            img.close()
            return text
        except FileNotFoundError:
            return f"Error: Image file '{image_path}' not found."
        except Exception as e:
            return f"Error extracting text from image: {str(e)}"