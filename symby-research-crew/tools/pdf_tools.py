"""
PDF Paper Analysis Tool for CrewAI
Parses and extracts information from uploaded research papers.
"""

from typing import Optional, List, Dict, Any
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from pathlib import Path
import re
import logging

logger = logging.getLogger(__name__)


class PDFAnalysisInput(BaseModel):
    """Input schema for PDF analysis."""
    file_path: str = Field(..., description="Path to the PDF file to analyze")


class PDFSectionInput(BaseModel):
    """Input schema for extracting a specific section."""
    file_path: str = Field(..., description="Path to the PDF file")
    section_name: str = Field(..., description="Name of the section to extract (e.g., 'Abstract', 'Methods', 'Results')")


class PDFParserTool(BaseTool):
    """Tool to parse and analyze research paper PDFs."""
    
    name: str = "parse_paper_pdf"
    description: str = """
    Parse a research paper PDF and extract its key components.
    Use this to understand the structure and content of an uploaded paper.
    Returns: Title, abstract, sections, references, and key information.
    """
    args_schema: type[BaseModel] = PDFAnalysisInput
    
    def _run(self, file_path: str) -> str:
        """Parse a PDF research paper."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"
            
            if not path.suffix.lower() == '.pdf':
                return f"Not a PDF file: {file_path}"
            
            # Try PyMuPDF first (better quality)
            try:
                import fitz  # PyMuPDF
                return self._parse_with_pymupdf(path)
            except ImportError:
                pass
            
            # Fallback to PyPDF2
            try:
                from PyPDF2 import PdfReader
                return self._parse_with_pypdf2(path)
            except ImportError:
                return "Error: Neither PyMuPDF nor PyPDF2 is installed. Please install one of them."
            
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            return f"Error parsing PDF: {str(e)}"
    
    def _parse_with_pymupdf(self, path: Path) -> str:
        """Parse PDF using PyMuPDF."""
        import fitz
        
        doc = fitz.open(path)
        
        output = f"# Paper Analysis: {path.name}\n\n"
        output += f"**Pages**: {len(doc)}\n\n"
        
        full_text = ""
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text += text + "\n"
            
            # Only process first few pages in detail
            if page_num == 0:
                output += f"## First Page Content\n{text[:2000]}\n\n"
        
        # Extract common sections
        sections = self._extract_sections(full_text)
        
        output += "## Detected Sections\n"
        for section, content in sections.items():
            if content:
                output += f"### {section}\n{content[:1000]}"
                if len(content) > 1000:
                    output += "... [truncated]"
                output += "\n\n"
        
        # Try to extract references
        refs = self._extract_references(full_text)
        if refs:
            output += f"## References (first 10)\n"
            for ref in refs[:10]:
                output += f"- {ref}\n"
        
        doc.close()
        return output
    
    def _parse_with_pypdf2(self, path: Path) -> str:
        """Parse PDF using PyPDF2."""
        from PyPDF2 import PdfReader
        
        reader = PdfReader(path)
        
        output = f"# Paper Analysis: {path.name}\n\n"
        output += f"**Pages**: {len(reader.pages)}\n\n"
        
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            full_text += text + "\n"
            
            if page_num == 0:
                output += f"## First Page Content\n{text[:2000]}\n\n"
        
        sections = self._extract_sections(full_text)
        
        output += "## Detected Sections\n"
        for section, content in sections.items():
            if content:
                output += f"### {section}\n{content[:1000]}"
                if len(content) > 1000:
                    output += "... [truncated]"
                output += "\n\n"
        
        return output
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common paper sections."""
        sections = {}
        
        # Common section patterns
        section_patterns = [
            (r'abstract[:\s]*\n(.*?)(?=\n[1I][\.\s]|\nintroduction|\nkeywords)', 'Abstract'),
            (r'introduction[:\s]*\n(.*?)(?=\n[2II][\.\s]|\nrelated|\nbackground|\nmethod)', 'Introduction'),
            (r'(?:related work|background)[:\s]*\n(.*?)(?=\n[3III][\.\s]|\nmethod|\napproach)', 'Background'),
            (r'(?:method|methodology|approach)[:\s]*\n(.*?)(?=\n[4IV][\.\s]|\nresult|\nexperiment)', 'Methods'),
            (r'(?:result|experiment)[:\s]*\n(.*?)(?=\n[5V][\.\s]|\ndiscussion|\nconclusion)', 'Results'),
            (r'(?:conclusion|summary)[:\s]*\n(.*?)(?=\nreference|\nacknowledg|\nappendix|$)', 'Conclusion'),
        ]
        
        text_lower = text.lower()
        
        for pattern, name in section_patterns:
            match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
            if match:
                # Get corresponding text from original (preserve case)
                start, end = match.start(1), match.end(1)
                sections[name] = text[start:end].strip()
        
        return sections
    
    def _extract_references(self, text: str) -> List[str]:
        """Extract references from the paper."""
        refs = []
        
        # Find references section
        ref_match = re.search(r'references?\s*\n(.*?)(?=\nappendix|$)', text, re.DOTALL | re.IGNORECASE)
        if ref_match:
            ref_text = ref_match.group(1)
            
            # Try to split into individual references
            # Pattern: [1], [2], etc. or 1., 2., etc.
            ref_patterns = re.split(r'\n\s*(?:\[\d+\]|\d+\.)\s*', ref_text)
            refs = [r.strip()[:200] for r in ref_patterns if r.strip() and len(r.strip()) > 20]
        
        return refs


class PDFExtractSectionTool(BaseTool):
    """Tool to extract a specific section from a PDF paper."""
    
    name: str = "extract_paper_section"
    description: str = """
    Extract a specific section from a research paper PDF.
    Use this when you need detailed content from a particular section.
    Sections: Abstract, Introduction, Methods, Results, Discussion, Conclusion, References.
    """
    args_schema: type[BaseModel] = PDFSectionInput
    
    def _run(self, file_path: str, section_name: str) -> str:
        """Extract a specific section from the PDF."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"
            
            try:
                import fitz
                doc = fitz.open(path)
                full_text = ""
                for page in doc:
                    full_text += page.get_text() + "\n"
                doc.close()
            except ImportError:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                full_text = ""
                for page in reader.pages:
                    full_text += (page.extract_text() or "") + "\n"
            
            # Find the section
            section_lower = section_name.lower()
            text_lower = full_text.lower()
            
            # Find section start
            start_patterns = [
                f'\n{section_lower}',
                f'\n{section_lower}:',
                f'\n{section_lower}\n',
            ]
            
            start_idx = -1
            for pattern in start_patterns:
                idx = text_lower.find(pattern)
                if idx != -1:
                    start_idx = idx
                    break
            
            if start_idx == -1:
                return f"Section '{section_name}' not found in the paper."
            
            # Find section end (next major section)
            next_sections = ['introduction', 'background', 'related', 'method', 'approach', 
                           'result', 'experiment', 'discussion', 'conclusion', 'reference', 'acknowledgment']
            
            end_idx = len(full_text)
            for ns in next_sections:
                if ns != section_lower:
                    ns_idx = text_lower.find(f'\n{ns}', start_idx + len(section_name))
                    if ns_idx != -1 and ns_idx < end_idx:
                        end_idx = ns_idx
            
            section_content = full_text[start_idx:end_idx].strip()
            
            # Truncate if too long
            if len(section_content) > 4000:
                section_content = section_content[:4000] + "\n\n... [Section truncated, showing first 4000 chars]"
            
            return f"# {section_name}\n\n{section_content}"
            
        except Exception as e:
            logger.error(f"Section extraction error: {e}")
            return f"Error extracting section: {str(e)}"


class PDFMetadataTool(BaseTool):
    """Tool to extract metadata from a PDF."""
    
    name: str = "get_paper_metadata"
    description: str = """
    Extract metadata from a research paper PDF including title, authors, and dates.
    Use this to quickly identify key information about a paper.
    """
    args_schema: type[BaseModel] = PDFAnalysisInput
    
    def _run(self, file_path: str) -> str:
        """Extract PDF metadata."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"File not found: {file_path}"
            
            try:
                import fitz
                doc = fitz.open(path)
                metadata = doc.metadata
                doc.close()
            except ImportError:
                from PyPDF2 import PdfReader
                reader = PdfReader(path)
                metadata = reader.metadata or {}
                # Convert PyPDF2 metadata format
                metadata = {k.lstrip('/'): v for k, v in metadata.items()}
            
            output = f"# Paper Metadata: {path.name}\n\n"
            
            fields = [
                ('title', 'Title'),
                ('author', 'Author'),
                ('subject', 'Subject'),
                ('creator', 'Creator'),
                ('producer', 'Producer'),
                ('creationDate', 'Created'),
                ('modDate', 'Modified'),
            ]
            
            for key, label in fields:
                value = metadata.get(key) or metadata.get(key.lower())
                if value:
                    output += f"**{label}**: {value}\n"
            
            if not any(metadata.get(k) for k, _ in fields):
                output += "No metadata found in PDF.\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return f"Error extracting metadata: {str(e)}"
