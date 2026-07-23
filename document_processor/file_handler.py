# Standard Library
import os
import hashlib
import pickle
from datetime import datetime, timedelta
from pathlib import Path

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from config import constants
from config.settings import settings
from utils.logging import logger


class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_path(self, file) -> str:
        """Helper to safely extract the file path from strings or Gradio file objects."""
        return file if isinstance(file, str) else getattr(file, 'name', str(file))

    def validate_files(self, files: list) -> None:
        """
        Validate the total size of the uploaded files.

        Args:
            files (list): A list of file objects (must have 'name' and be readable).
        """
        total_size = sum(os.path.getsize(self._get_path(f)) for f in files)
        if total_size > constants.MAX_FILE_SIZE:
            raise ValueError(f"Total size exceeded {constants.MAX_FILE_SIZE // 1024 // 1024}MB limit.")
        
    def process(self, files: list) -> list:
        """
        Processes a batch of files into text chunks, utilizing a local cache to skip re-parsing unchanged files.

        Args:
            files (list): A list of file objects (must have 'name' and be readable).

        Returns:
            list: A flattened list of unique document chunks from all successfully 
                processed files.

        Raises:
            ValidationError: (Assuming `validate_files` raises this) If the input files list is invalid.
        """
        self.validate_files(files)

        all_chunks = []
        seen_hashes = set() 

        # Generate content-based hash for caching
        for file in files:
            path = self._get_path(file)
            try:
                with open(path, "rb") as f:
                    # Hash the raw binary to detect if the file has been modified
                    file_hash = self._generate_hash(f.read())
                    cache_path = self.cache_dir / f"{file_hash}.pkl"

                    if self._is_cache_valid(cache_path):
                        chunks = self._load_from_cache(cache_path)
                    else:
                        logger.info(f"Processing and caching: {file.name}")
                        chunks = self._process_file(file)
                        self._save_to_cache(chunks, cache_path)

                    # De-duplicate chunks across different files to prevent redundant context being fed to the LLM
                    for chunk in chunks:
                        chunk_hash = self._generate_hash(chunk.page_content.encode())
                        if chunk_hash not in seen_hashes:
                            all_chunks.append(chunk)
                            seen_hashes.add(chunk_hash)

            except Exception as e:
                # Log the error but continue so one corrupted file doesn't crash the entire batch processing
                logger.error(f"Failed to process {file.name}: {str(e)}")
                continue

            # Note: logs a running total after each file.
            logger.info(f"Total unique chunks: {len(all_chunks)}")

        return all_chunks

    def _process_file(self, file_path: str) -> list:
        """
        Converts a single supported document into semantically split Markdown chunks.

        This method uses Docling to extract structural elements (tables, text), 
        standardizes them into Markdown, and chunks the content based on header hierarchy for optimal LLM retrieval.

        Args:
            file: A file objects (must have 'name' and be readable).

        Returns:
            list: A list of Document/chunk objects split by headers.
        """
        # Failcheck to prevent the DocumentConverter from crashing on unsupported binary files
        if not file_path.endswith(('.pdf', '.docx', '.txt', '.md')):
            logger.warning(f"Skipping unsupported file type: {file_path}")
            return []
        
        # Initialize the Docling converter to parse the raw document
        converter = DocumentConverter()

        # Convert to Markdown
        markdown = converter.convert(file_path).document.export_to_markdown()   
        splitter = MarkdownHeaderTextSplitter(self.headers) # Split the document contextually at header boundaries
        
        return splitter.split_text(markdown)
    
    def _generate_hash(self, content: bytes) -> str:
        """Generates a deterministic SHA-256 hex string for file/chunk tracking."""
        return hashlib.sha256(content).hexdigest()
    
    def _save_to_cache(self, chunks: list, cache_path: Path):
        """Serializes document chunks to a local file along with a processing timestamp."""
        with open(cache_path, "wb")as f:
            pickle.dump({
                "timestamp": datetime.now().timestamp(),
                "chunks": chunks,
            }, f)

    def _load_from_cache(self, cache_path: Path) -> list:
        """Deserializes and returns the list of cached document chunks from a file."""
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            return data['chunks']
        
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Checks if a valid, unexpired cache file exists.
        
        Returns:
            bool: True if cache exists and is newer than the expiration threshold,
                  False otherwise.
        """
        # Failsafe
        if not cache_path.exists():
            return False
        # Calculate time since last modification
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)

        # Return True only if the cache age is younger than the expiration setting
        return cache_age < timedelta(days=settings.CACHE_EXPIRE_DAYS)
