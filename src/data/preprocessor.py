"""
Data Preprocessor for ML Impact Analysis
Extracts ML techniques, builds citation proxies, computes metrics

Enhanced for Full Hybrid Semantic-Graph Search System:
- TextPreprocessor: Clean and prepare text for BGE-M3 embeddings
- Smart truncation for 8K token limit
- Section extraction (title, abstract, introduction, methods, results)
"""

import re
from typing import List, Dict, Tuple, Optional, Generator
from collections import defaultdict
from pathlib import Path
import hashlib

try:
    import cudf as pd
    GPU_AVAILABLE = True
except ImportError:
    import pandas as pd
    GPU_AVAILABLE = False


class TextPreprocessor:
    """
    Preprocess paper text for BGE-M3 embedding generation.

    Handles:
    - Boilerplate removal (acknowledgments, references, author info)
    - Section extraction with priority ordering
    - Smart truncation to fit 8K token limit (~32K chars)
    - Unicode normalization
    """

    # Maximum characters (BGE-M3 supports 8K tokens, ~4 chars/token avg)
    MAX_CHARS = 28000  # Conservative to stay under 8K tokens

    # Section headers to identify paper structure
    SECTION_PATTERNS = {
        'abstract': r'(?:^|\n)\s*(?:abstract|summary)\s*[:\n]',
        'introduction': r'(?:^|\n)\s*(?:1\.?\s*)?introduction\s*[:\n]',
        'methods': r'(?:^|\n)\s*(?:\d\.?\s*)?(?:method(?:s|ology)?|materials?\s*(?:and|&)\s*methods?|experimental)\s*[:\n]',
        'results': r'(?:^|\n)\s*(?:\d\.?\s*)?results?\s*(?:and\s+discussion)?\s*[:\n]',
        'discussion': r'(?:^|\n)\s*(?:\d\.?\s*)?discussion\s*[:\n]',
        'conclusion': r'(?:^|\n)\s*(?:\d\.?\s*)?(?:conclusion|concluding\s+remarks?|summary)\s*[:\n]',
        'references': r'(?:^|\n)\s*(?:references?|bibliography|works?\s+cited)\s*[:\n]',
        'acknowledgments': r'(?:^|\n)\s*(?:acknowledgm?ents?|funding|disclosure)\s*[:\n]',
    }

    # Boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r'©\s*\d{4}.*?(?:reserved|license)',  # Copyright notices
        r'doi:\s*10\.\d+/\S+',  # DOI
        r'https?://\S+',  # URLs
        r'\b(?:email|e-mail):\s*\S+@\S+',  # Email addresses
        r'(?:corresponding\s+)?author(?:s)?[:\s]+.*?(?:\n|$)',  # Author info
        r'keywords?:\s*.*?(?:\n\n|\n[A-Z])',  # Keywords section
        r'received:?\s*\d+.*?(?:accepted|published):?\s*\d+.*?(?:\n|$)',  # Dates
    ]

    # Section priority for truncation (higher = keep more)
    SECTION_PRIORITY = {
        'abstract': 10,
        'introduction': 8,
        'methods': 7,
        'results': 9,
        'discussion': 6,
        'conclusion': 7,
        'references': 1,  # Low priority - mostly noise for embeddings
        'acknowledgments': 1,  # Low priority
    }

    def __init__(self, max_chars: int = None):
        self.max_chars = max_chars or self.MAX_CHARS
        self._compile_patterns()

    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.section_regex = {
            name: re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for name, pattern in self.SECTION_PATTERNS.items()
        }
        self.boilerplate_regex = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for pattern in self.BOILERPLATE_PATTERNS
        ]

    def clean(self, text: str) -> str:
        """
        Clean text by removing boilerplate and normalizing whitespace.

        Args:
            text: Raw paper text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove boilerplate
        for pattern in self.boilerplate_regex:
            text = pattern.sub(' ', text)

        # Normalize unicode
        text = text.encode('utf-8', errors='ignore').decode('utf-8')

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove very short lines (likely headers/page numbers)
        lines = text.split('\n')
        lines = [l for l in lines if len(l.strip()) > 10 or not l.strip()]
        text = '\n'.join(lines)

        return text.strip()

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract paper sections based on common headers.

        Args:
            text: Paper text

        Returns:
            Dict mapping section names to content
        """
        sections = {}
        text_lower = text.lower()

        # Find all section positions
        positions = []
        for name, pattern in self.section_regex.items():
            match = pattern.search(text)
            if match:
                positions.append((match.start(), name, match.end()))

        # Sort by position
        positions.sort(key=lambda x: x[0])

        # Extract sections
        for i, (start, name, content_start) in enumerate(positions):
            if i + 1 < len(positions):
                end = positions[i + 1][0]
            else:
                end = len(text)

            section_text = text[content_start:end].strip()
            sections[name] = section_text

        # If no sections found, treat entire text as content
        if not sections:
            # Try to extract title from first line
            lines = text.split('\n')
            if lines:
                sections['title'] = lines[0].strip()[:500]
                sections['content'] = text

        return sections

    def truncate_smart(self, text: str, sections: Dict[str, str] = None) -> str:
        """
        Smart truncation that preserves high-priority sections.

        Uses section priorities to keep the most important content
        when text exceeds max_chars limit.

        Args:
            text: Full text (used if sections not provided)
            sections: Pre-extracted sections dict

        Returns:
            Truncated text fitting within max_chars
        """
        if len(text) <= self.max_chars:
            return text

        if not sections:
            sections = self.extract_sections(text)

        # Sort sections by priority
        sorted_sections = sorted(
            sections.items(),
            key=lambda x: self.SECTION_PRIORITY.get(x[0], 5),
            reverse=True
        )

        # Build truncated text
        result_parts = []
        remaining_chars = self.max_chars

        for section_name, section_text in sorted_sections:
            if remaining_chars <= 0:
                break

            # Skip low-priority sections if running low on space
            if remaining_chars < 2000 and self.SECTION_PRIORITY.get(section_name, 5) < 5:
                continue

            # Allocate space based on priority
            priority = self.SECTION_PRIORITY.get(section_name, 5)
            max_section_chars = min(
                len(section_text),
                remaining_chars,
                int(self.max_chars * (priority / 50))  # High priority gets more space
            )

            if max_section_chars > 100:  # Don't add tiny snippets
                truncated = section_text[:max_section_chars]
                # Try to end at sentence boundary
                last_period = truncated.rfind('.')
                if last_period > max_section_chars * 0.7:
                    truncated = truncated[:last_period + 1]

                result_parts.append(f"[{section_name.upper()}]\n{truncated}")
                remaining_chars -= len(truncated) + len(section_name) + 5

        return '\n\n'.join(result_parts)

    def process(self, text: str, paper_id: str = None) -> Dict:
        """
        Full preprocessing pipeline for a paper.

        Args:
            text: Raw paper text
            paper_id: Optional paper ID for tracking

        Returns:
            Dict with processed text and metadata
        """
        # Clean
        cleaned = self.clean(text)

        # Extract sections
        sections = self.extract_sections(cleaned)

        # Get title (first line or from sections)
        title = sections.get('title', '')
        if not title and cleaned:
            title = cleaned.split('\n')[0][:300]

        # Smart truncate
        truncated = self.truncate_smart(cleaned, sections)

        # Create embedding-ready text (title + truncated content)
        embedding_text = f"{title}\n\n{truncated}" if title else truncated

        # Compute text hash for deduplication
        text_hash = hashlib.md5(embedding_text.encode()).hexdigest()

        return {
            'paper_id': paper_id,
            'title': title,
            'embedding_text': embedding_text[:self.max_chars],
            'text_length': len(text),
            'embedding_length': len(embedding_text),
            'sections_found': list(sections.keys()),
            'text_hash': text_hash,
            'has_abstract': 'abstract' in sections,
            'has_methods': 'methods' in sections,
            'has_results': 'results' in sections,
        }

    def process_batch(self, papers: List[Dict]) -> Generator[Dict, None, None]:
        """
        Process a batch of papers.

        Args:
            papers: List of paper dicts with 'text' and 'id' keys

        Yields:
            Processed paper dicts
        """
        for paper in papers:
            text = paper.get('text', '')
            paper_id = paper.get('id', '')

            processed = self.process(text, paper_id)

            # Merge with original paper metadata
            processed.update({
                k: v for k, v in paper.items()
                if k not in ['text']  # Don't duplicate raw text
            })

            yield processed


class Preprocessor:
    """Preprocess papers for ML impact analysis"""

    # ML Technique Categories with keywords
    ML_TECHNIQUES = {
        'deep_learning': ['deep learning', 'neural network', 'deep neural'],
        'transformer': ['transformer', 'attention mechanism', 'self-attention', 'bert', 'gpt'],
        'cnn': ['convolutional', 'cnn', 'convnet', 'resnet', 'vgg', 'inception'],
        'rnn': ['recurrent', 'rnn', 'lstm', 'gru', 'sequence model'],
        'gan': ['generative adversarial', 'gan', 'discriminator', 'generator network'],
        'reinforcement_learning': ['reinforcement learning', 'q-learning', 'policy gradient', 'actor-critic'],
        'graph_neural': ['graph neural', 'gnn', 'graph convolutional', 'message passing'],
        'diffusion': ['diffusion model', 'denoising', 'score matching', 'stable diffusion'],
        'llm': ['large language model', 'llm', 'gpt-3', 'gpt-4', 'chatgpt', 'llama', 'claude'],
        'autoencoder': ['autoencoder', 'vae', 'variational autoencoder'],
        'transfer_learning': ['transfer learning', 'fine-tuning', 'pre-training', 'pretrained'],
        'few_shot': ['few-shot', 'zero-shot', 'meta-learning', 'prompt learning'],
    }

    # Domain categories
    DOMAINS = {
        'drug_discovery': ['drug discovery', 'pharmaceutical', 'molecule', 'compound screening', 'drug design'],
        'materials_science': ['materials science', 'crystal structure', 'alloy', 'polymer', 'semiconductor'],
        'climate_science': ['climate', 'weather prediction', 'atmospheric', 'carbon', 'emissions'],
        'physics': ['quantum', 'particle physics', 'cosmology', 'astrophysics', 'condensed matter'],
        'neuroscience': ['neuroscience', 'brain', 'neural activity', 'cognitive', 'fmri'],
        'biology': ['genomics', 'proteomics', 'cell biology', 'molecular biology', 'genetics'],
        'medicine': ['clinical', 'diagnosis', 'treatment', 'patient', 'medical imaging'],
    }

    # Breakthrough indicators
    BREAKTHROUGH_KEYWORDS = [
        'state-of-the-art', 'sota', 'breakthrough', 'novel', 'first',
        'outperforms', 'significant improvement', 'remarkable', 'unprecedented'
    ]

    def __init__(self):
        self.technique_patterns = self._compile_patterns(self.ML_TECHNIQUES)
        self.domain_patterns = self._compile_patterns(self.DOMAINS)

    def _compile_patterns(self, keyword_dict: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
        """Compile regex patterns for efficient matching"""
        patterns = {}
        for category, keywords in keyword_dict.items():
            pattern = '|'.join(re.escape(kw) for kw in keywords)
            patterns[category] = re.compile(pattern, re.IGNORECASE)
        return patterns

    def extract_ml_techniques(self, text: str) -> List[str]:
        """Extract ML techniques mentioned in paper"""
        techniques = []
        text_lower = text.lower()
        for technique, pattern in self.technique_patterns.items():
            if pattern.search(text_lower):
                techniques.append(technique)
        return techniques

    def extract_domains(self, text: str) -> List[str]:
        """Extract scientific domains from paper"""
        domains = []
        text_lower = text.lower()
        for domain, pattern in self.domain_patterns.items():
            if pattern.search(text_lower):
                domains.append(domain)
        return domains

    def compute_breakthrough_score(self, text: str) -> float:
        """Compute a breakthrough/novelty score based on language"""
        text_lower = text.lower()
        score = sum(1 for kw in self.BREAKTHROUGH_KEYWORDS if kw in text_lower)
        return min(score / len(self.BREAKTHROUGH_KEYWORDS), 1.0)

    def extract_references_count(self, text: str) -> int:
        """Estimate number of references from text patterns"""
        # Look for citation patterns like [1], [2], etc.
        bracket_refs = len(re.findall(r'\[\d+\]', text))
        # Look for author-year patterns
        author_year = len(re.findall(r'\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', text))
        return bracket_refs + author_year

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extracted features to dataframe"""
        print("Extracting ML techniques...")

        # Convert to pandas for text processing if using cuDF
        if GPU_AVAILABLE:
            df_cpu = df.to_pandas()
        else:
            df_cpu = df

        # Extract techniques
        df_cpu['ml_techniques'] = df_cpu['title'].apply(
            lambda x: self.extract_ml_techniques(str(x))
        )

        # Extract domains
        df_cpu['domains'] = df_cpu['title'].apply(
            lambda x: self.extract_domains(str(x))
        )

        # Breakthrough score
        df_cpu['breakthrough_score'] = df_cpu['title'].apply(
            lambda x: self.compute_breakthrough_score(str(x))
        )

        # Flatten techniques for analysis
        df_cpu['technique_count'] = df_cpu['ml_techniques'].apply(len)
        df_cpu['domain_count'] = df_cpu['domains'].apply(len)

        # Primary technique (first one found)
        df_cpu['primary_technique'] = df_cpu['ml_techniques'].apply(
            lambda x: x[0] if x else 'none'
        )

        # Primary domain
        df_cpu['primary_domain'] = df_cpu['domains'].apply(
            lambda x: x[0] if x else 'other'
        )

        # Convert back to cuDF if needed
        if GPU_AVAILABLE:
            # Note: cuDF doesn't support list columns well, keep as pandas for now
            return df_cpu
        return df_cpu

    def compute_adoption_metrics(self, df: pd.DataFrame) -> Dict:
        """Compute ML adoption metrics by field and year"""
        metrics = {
            'by_year': {},
            'by_field': {},
            'by_technique': {},
            'technique_by_year': {},
        }

        # Adoption by year
        year_groups = df.groupby('year')
        for year, group in year_groups:
            total = len(group)
            ml_papers = group['is_ml_paper'].sum()
            metrics['by_year'][year] = {
                'total': total,
                'ml_papers': ml_papers,
                'adoption_rate': ml_papers / total if total > 0 else 0,
                'avg_ml_score': group['ml_score'].mean(),
            }

        # Adoption by field
        field_groups = df.groupby('source_field')
        for field, group in field_groups:
            total = len(group)
            ml_papers = group['is_ml_paper'].sum()
            metrics['by_field'][field] = {
                'total': total,
                'ml_papers': ml_papers,
                'adoption_rate': ml_papers / total if total > 0 else 0,
                'avg_repro_score': group['repro_score'].mean(),
            }

        # Technique popularity
        technique_counts = defaultdict(int)
        for techniques in df['ml_techniques']:
            if isinstance(techniques, list):
                for tech in techniques:
                    technique_counts[tech] += 1

        metrics['by_technique'] = dict(technique_counts)

        return metrics


if __name__ == "__main__":
    from loader import DataLoader

    # Test preprocessing
    loader = DataLoader("/home/abheekp/ai_research/s2orc_data")
    df = loader.load_sample(200)

    preprocessor = Preprocessor()
    df_processed = preprocessor.process_dataframe(df)

    print(f"\nProcessed columns: {df_processed.columns.tolist()}")
    print(f"\nML techniques found: {df_processed['primary_technique'].value_counts()}")
    print(f"\nDomains found: {df_processed['primary_domain'].value_counts()}")
