"""
Paper Iterator for Full Hybrid Semantic-Graph Search System
Memory-efficient streaming through 3.2M+ papers for embedding generation

Supports:
- Batch iteration with configurable batch size
- Checkpoint/resume for long-running jobs
- Field filtering
- Progress tracking
- Parallel preprocessing
"""

import json
import gzip
from pathlib import Path
from typing import Generator, Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pickle
import time

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    def tqdm(x, **kwargs):
        return x

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

from .preprocessor import TextPreprocessor


class PaperIterator:
    """
    Memory-efficient iterator for streaming through the S2ORC dataset.

    Yields batches of preprocessed papers ready for embedding generation.
    Supports checkpointing for resuming interrupted jobs.
    """

    def __init__(
        self,
        data_dir: str = "s2orc_data",
        batch_size: int = 64,
        checkpoint_dir: str = "checkpoints",
        preprocess: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize the paper iterator.

        Args:
            data_dir: Path to s2orc_data directory
            batch_size: Number of papers per batch (tune for GPU memory)
            checkpoint_dir: Directory for checkpoint files
            preprocess: Whether to run TextPreprocessor
            max_workers: Number of threads for parallel preprocessing
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.preprocess = preprocess
        self.max_workers = max_workers

        if preprocess:
            self.text_preprocessor = TextPreprocessor()
        else:
            self.text_preprocessor = None

        # Track progress
        self.total_papers = 0
        self.processed_papers = 0
        self.current_file_idx = 0
        self.current_paper_idx = 0

    def discover_files(
        self,
        fields_filter: Optional[List[str]] = None,
        splits: Optional[List[str]] = None
    ) -> List[Tuple[Path, str, str]]:
        """
        Discover all data files in the dataset.

        Args:
            fields_filter: Only include files from these fields
            splits: Only include these splits ('train', 'val', 'test')

        Returns:
            List of (file_path, field, split) tuples
        """
        if splits is None:
            splits = ['train', 'val', 'test']

        files = []

        for item in self.data_dir.iterdir():
            if not item.is_dir() or ',' not in item.name:
                continue

            parts = item.name.split(',')
            field = parts[0]

            # Apply field filter
            if fields_filter and field not in fields_filter:
                continue

            for split in splits:
                split_dir = item / split
                if not split_dir.exists():
                    continue

                # Find all data files
                for pattern in ['*.json', '*.jsonl', '*.json.gz', '*.jsonl.gz']:
                    for data_file in split_dir.glob(pattern):
                        files.append((data_file, field, split))

        return sorted(files, key=lambda x: str(x[0]))

    def count_papers(self, files: List[Tuple[Path, str, str]]) -> int:
        """
        Estimate total paper count from files.

        Args:
            files: List of (file_path, field, split) tuples

        Returns:
            Estimated total paper count
        """
        total = 0
        print("Counting papers...")

        for file_path, _, _ in tqdm(files[:10]):  # Sample first 10 files
            count = sum(1 for _ in self._read_file(file_path))
            total += count

        # Estimate for remaining files
        avg_per_file = total / min(len(files), 10)
        estimated_total = int(avg_per_file * len(files))

        print(f"Estimated total papers: {estimated_total:,}")
        return estimated_total

    def _read_file(self, filepath: Path) -> Generator[Dict, None, None]:
        """Read papers from a single file."""
        is_gzipped = str(filepath).endswith('.gz')

        try:
            if is_gzipped:
                with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if USE_ORJSON:
                                yield orjson.loads(line)
                            else:
                                yield json.loads(line)
                        except (json.JSONDecodeError, Exception):
                            continue
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            if USE_ORJSON:
                                yield orjson.loads(line)
                            else:
                                yield json.loads(line)
                        except (json.JSONDecodeError, Exception):
                            continue
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    def _preprocess_paper(self, paper: Dict, field: str, split: str) -> Dict:
        """Preprocess a single paper."""
        text = paper.get('text', '')
        paper_id = str(paper.get('id', ''))
        metadata = paper.get('metadata', {})

        # Extract year
        year = metadata.get('year', 0)
        if not year:
            created = paper.get('created', '')
            if created and len(created) >= 4:
                try:
                    year = int(created[:4])
                except:
                    year = 0

        # Get title from first line
        title = text.split('\n')[0][:300] if text else ""

        # Preprocess text for embeddings
        if self.text_preprocessor:
            processed = self.text_preprocessor.process(text, paper_id)
            embedding_text = processed['embedding_text']
            text_hash = processed['text_hash']
            sections_found = processed['sections_found']
        else:
            embedding_text = text[:28000]  # Simple truncation
            text_hash = ''
            sections_found = []

        # Get fields of study
        s2_fields = metadata.get('s2fieldsofstudy', []) or []
        ext_fields = metadata.get('extfieldsofstudy', []) or []

        return {
            'paper_id': paper_id,
            'title': title,
            'embedding_text': embedding_text,
            'text_length': len(text),
            'year': year,
            'source_field': field,
            'split': split,
            's2_fields': s2_fields,
            'ext_fields': ext_fields,
            'text_hash': text_hash,
            'sections_found': sections_found,
            'created': paper.get('created', ''),
        }

    def save_checkpoint(self, checkpoint_name: str = "iterator"):
        """Save current iteration state."""
        state = {
            'file_idx': self.current_file_idx,
            'paper_idx': self.current_paper_idx,
            'processed_papers': self.processed_papers,
            'timestamp': time.time(),
        }

        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(state, f)

        print(f"Checkpoint saved: {checkpoint_path} (processed {self.processed_papers:,} papers)")

    def load_checkpoint(self, checkpoint_name: str = "iterator") -> bool:
        """Load iteration state from checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"

        if not checkpoint_path.exists():
            return False

        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)

        self.current_file_idx = state['file_idx']
        self.current_paper_idx = state['paper_idx']
        self.processed_papers = state['processed_papers']

        print(f"Resumed from checkpoint: {self.processed_papers:,} papers processed")
        return True

    def iterate(
        self,
        fields_filter: Optional[List[str]] = None,
        splits: Optional[List[str]] = None,
        max_papers: Optional[int] = None,
        resume: bool = True,
        checkpoint_every: int = 10000,
        show_progress: bool = True
    ) -> Generator[List[Dict], None, None]:
        """
        Iterate through papers in batches.

        Args:
            fields_filter: Only process these fields
            splits: Only process these splits
            max_papers: Maximum papers to process (None = all)
            resume: Whether to resume from checkpoint
            checkpoint_every: Save checkpoint every N papers
            show_progress: Show progress bar

        Yields:
            List of preprocessed paper dicts (batch_size papers)
        """
        # Discover files
        files = self.discover_files(fields_filter, splits)
        print(f"Found {len(files)} data files")

        if not files:
            return

        # Try to resume from checkpoint
        if resume:
            self.load_checkpoint()

        # Skip already processed files
        files = files[self.current_file_idx:]

        batch = []
        papers_in_current_file = 0
        last_checkpoint = self.processed_papers

        # Create progress bar
        if show_progress and TQDM_AVAILABLE:
            pbar = tqdm(
                total=max_papers or self.count_papers(files),
                initial=self.processed_papers,
                desc="Processing papers",
                unit="papers"
            )
        else:
            pbar = None

        try:
            for file_idx, (file_path, field, split) in enumerate(files):
                self.current_file_idx = file_idx + (len(files) - len(files))
                papers_in_current_file = 0

                for paper in self._read_file(file_path):
                    # Skip papers if resuming mid-file
                    if papers_in_current_file < self.current_paper_idx:
                        papers_in_current_file += 1
                        continue

                    # Preprocess
                    processed = self._preprocess_paper(paper, field, split)
                    batch.append(processed)
                    self.processed_papers += 1
                    papers_in_current_file += 1

                    # Update progress
                    if pbar:
                        pbar.update(1)

                    # Yield batch when full
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []

                    # Save checkpoint periodically
                    if self.processed_papers - last_checkpoint >= checkpoint_every:
                        self.current_paper_idx = papers_in_current_file
                        self.save_checkpoint()
                        last_checkpoint = self.processed_papers

                    # Check max papers
                    if max_papers and self.processed_papers >= max_papers:
                        if batch:
                            yield batch
                        return

                # Reset paper index after completing file
                self.current_paper_idx = 0

            # Yield remaining papers
            if batch:
                yield batch

        finally:
            if pbar:
                pbar.close()
            # Save final checkpoint
            self.save_checkpoint()

    def get_stats(self) -> Dict:
        """Get statistics about the dataset."""
        files = self.discover_files()

        stats = {
            'total_files': len(files),
            'fields': defaultdict(int),
            'splits': defaultdict(int),
        }

        for _, field, split in files:
            stats['fields'][field] += 1
            stats['splits'][split] += 1

        stats['fields'] = dict(stats['fields'])
        stats['splits'] = dict(stats['splits'])

        return stats


class EmbeddingBatcher:
    """
    Batches preprocessed papers for efficient embedding generation.

    Optimized for GPU memory management and throughput.
    """

    def __init__(
        self,
        target_tokens: int = 8000,
        max_batch_size: int = 32,
        min_batch_size: int = 4
    ):
        """
        Initialize the batcher.

        Args:
            target_tokens: Target tokens per batch (for GPU memory)
            max_batch_size: Maximum papers per batch
            min_batch_size: Minimum papers per batch
        """
        self.target_tokens = target_tokens
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size

        # Approximate chars per token
        self.chars_per_token = 4

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.chars_per_token

    def create_batches(
        self,
        papers: List[Dict]
    ) -> Generator[Tuple[List[str], List[str]], None, None]:
        """
        Create optimally-sized batches for embedding generation.

        Args:
            papers: List of preprocessed paper dicts

        Yields:
            Tuple of (paper_ids, texts) for each batch
        """
        current_batch_ids = []
        current_batch_texts = []
        current_tokens = 0

        for paper in papers:
            text = paper.get('embedding_text', '')
            paper_id = paper.get('paper_id', '')
            tokens = self.estimate_tokens(text)

            # Check if adding this paper would exceed limits
            would_exceed_tokens = current_tokens + tokens > self.target_tokens * self.max_batch_size
            would_exceed_size = len(current_batch_ids) >= self.max_batch_size

            if (would_exceed_tokens or would_exceed_size) and len(current_batch_ids) >= self.min_batch_size:
                yield (current_batch_ids, current_batch_texts)
                current_batch_ids = []
                current_batch_texts = []
                current_tokens = 0

            current_batch_ids.append(paper_id)
            current_batch_texts.append(text)
            current_tokens += tokens

        # Yield remaining
        if current_batch_ids:
            yield (current_batch_ids, current_batch_texts)


if __name__ == "__main__":
    # Test the iterator
    iterator = PaperIterator("/home/abheekp/ai_research/s2orc_data", batch_size=10)

    print("\n=== Dataset Stats ===")
    stats = iterator.get_stats()
    print(f"Total files: {stats['total_files']}")
    print(f"Fields: {stats['fields']}")
    print(f"Splits: {stats['splits']}")

    print("\n=== Testing Iterator ===")
    batch_count = 0
    paper_count = 0

    for batch in iterator.iterate(max_papers=100, resume=False):
        batch_count += 1
        paper_count += len(batch)

        if batch_count == 1:
            print(f"\nFirst batch sample:")
            print(f"  Paper ID: {batch[0]['paper_id']}")
            print(f"  Title: {batch[0]['title'][:80]}...")
            print(f"  Year: {batch[0]['year']}")
            print(f"  Field: {batch[0]['source_field']}")
            print(f"  Embedding text length: {len(batch[0]['embedding_text'])}")
            print(f"  Sections found: {batch[0]['sections_found']}")

    print(f"\nTotal: {batch_count} batches, {paper_count} papers")
