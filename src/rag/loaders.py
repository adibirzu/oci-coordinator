"""
Document Loaders for RAG.

Provides loaders for various document sources including
OCI documentation, markdown files, and structured data.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import structlog
import yaml

from src.rag.vector_store import Document

logger = structlog.get_logger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for document loading."""

    include_metadata: bool = True
    max_file_size: int = 1024 * 1024  # 1MB
    supported_extensions: tuple[str, ...] = (".md", ".txt", ".yaml", ".yml", ".json")


class DirectoryLoader:
    """
    Load documents from a directory.

    Recursively loads all supported files from a directory.
    """

    def __init__(
        self,
        directory: str,
        config: LoaderConfig | None = None,
        glob_pattern: str = "**/*",
    ):
        self.directory = directory
        self.config = config or LoaderConfig()
        self.glob_pattern = glob_pattern

    def load(self) -> list[Document]:
        """Load all documents from the directory."""
        import glob

        documents = []
        pattern = os.path.join(self.directory, self.glob_pattern)

        for file_path in glob.glob(pattern, recursive=True):
            if not os.path.isfile(file_path):
                continue

            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.config.supported_extensions:
                continue

            # Check file size
            if os.path.getsize(file_path) > self.config.max_file_size:
                logger.warning("Skipping large file", path=file_path)
                continue

            try:
                doc = self._load_file(file_path)
                if doc:
                    documents.append(doc)
            except Exception as e:
                logger.warning("Failed to load file", path=file_path, error=str(e))

        logger.info("Loaded documents from directory", count=len(documents), directory=self.directory)
        return documents

    def _load_file(self, file_path: str) -> Document | None:
        """Load a single file."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            return None

        # Extract metadata based on file type
        ext = os.path.splitext(file_path)[1].lower()
        metadata = {
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "file_type": ext[1:],
        }

        if ext in (".md",):
            # Extract markdown frontmatter
            frontmatter, body = self._parse_markdown(content)
            metadata.update(frontmatter)
            content = body

        elif ext in (".yaml", ".yml"):
            # Parse YAML
            try:
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    # Use as structured content
                    content = yaml.dump(data, default_flow_style=False)
            except Exception:
                pass

        elif ext in (".json",):
            # Format JSON nicely
            try:
                data = json.loads(content)
                content = json.dumps(data, indent=2)
            except Exception:
                pass

        # Generate ID from path
        doc_id = file_path.replace("/", "_").replace("\\", "_").replace(".", "_")

        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
        )

    def _parse_markdown(self, content: str) -> tuple[dict[str, Any], str]:
        """Parse markdown frontmatter and body."""
        frontmatter = {}
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                except Exception:
                    pass

        return frontmatter, body


class MarkdownLoader:
    """
    Load and parse markdown documents.

    Extracts headers, sections, and metadata from markdown files.
    """

    def __init__(self, split_by_headers: bool = True, min_section_size: int = 100):
        self.split_by_headers = split_by_headers
        self.min_section_size = min_section_size

    def load_file(self, file_path: str) -> list[Document]:
        """Load a markdown file, optionally splitting by headers."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        return self.load_text(content, source=file_path)

    def load_text(self, content: str, source: str = "markdown") -> list[Document]:
        """Load markdown text."""
        # Parse frontmatter
        frontmatter = {}
        body = content

        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2].strip()
                except Exception:
                    pass

        if not self.split_by_headers:
            return [
                Document(
                    id=f"{source}_full",
                    content=body,
                    metadata={"source": source, **frontmatter},
                )
            ]

        # Split by headers
        documents = []
        sections = self._split_by_headers(body)

        for i, (header, text) in enumerate(sections):
            if len(text) < self.min_section_size:
                continue

            section_id = f"{source}_{i}_{self._slugify(header)}"
            documents.append(
                Document(
                    id=section_id,
                    content=f"# {header}\n\n{text}" if header else text,
                    metadata={
                        "source": source,
                        "section": header,
                        "section_index": i,
                        **frontmatter,
                    },
                )
            )

        return documents

    def _split_by_headers(self, content: str) -> list[tuple[str, str]]:
        """Split content by markdown headers."""
        # Match headers (# to ######)
        header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

        sections = []
        last_end = 0
        last_header = ""

        for match in header_pattern.finditer(content):
            # Save previous section
            if last_end > 0 or match.start() > 0:
                text = content[last_end : match.start()].strip()
                if text or last_header:
                    sections.append((last_header, text))

            last_header = match.group(2).strip()
            last_end = match.end()

        # Add final section
        final_text = content[last_end:].strip()
        if final_text:
            sections.append((last_header, final_text))

        return sections

    def _slugify(self, text: str) -> str:
        """Convert text to slug for ID."""
        text = text.lower()
        text = re.sub(r"[^\w\s-]", "", text)
        text = re.sub(r"[\s_-]+", "_", text)
        return text[:50]


class OCIDocumentationLoader:
    """
    Load OCI-specific documentation.

    Handles OCI documentation formats and extracts relevant sections
    for agent context.
    """

    def __init__(self, docs_directory: str | None = None):
        self.docs_directory = docs_directory or os.path.join(
            os.path.dirname(__file__), "..", "..", "docs"
        )
        self.markdown_loader = MarkdownLoader(split_by_headers=True)

    def load_all(self) -> list[Document]:
        """Load all OCI documentation."""
        documents = []

        # Load markdown docs
        if os.path.exists(self.docs_directory):
            loader = DirectoryLoader(self.docs_directory)
            documents.extend(loader.load())

        # Load agent prompts (useful for understanding capabilities)
        prompts_dir = os.path.join(self.docs_directory, "..", "prompts")
        if os.path.exists(prompts_dir):
            loader = DirectoryLoader(prompts_dir)
            for doc in loader.load():
                doc.metadata["type"] = "agent_prompt"
                documents.append(doc)

        logger.info("Loaded OCI documentation", count=len(documents))
        return documents

    def load_agent_knowledge(self, agent_id: str) -> list[Document]:
        """Load knowledge specific to an agent."""
        documents = []

        # Load agent-specific docs
        agent_docs_dir = os.path.join(self.docs_directory, "agents", agent_id)
        if os.path.exists(agent_docs_dir):
            loader = DirectoryLoader(agent_docs_dir)
            documents.extend(loader.load())

        # Load shared knowledge
        shared_dir = os.path.join(self.docs_directory, "shared")
        if os.path.exists(shared_dir):
            loader = DirectoryLoader(shared_dir)
            documents.extend(loader.load())

        return documents


class StructuredDataLoader:
    """
    Load structured data (JSON, YAML) as documents.

    Useful for loading configuration, schemas, and API definitions.
    """

    def load_json(
        self,
        file_path: str,
        text_field: str | None = None,
        id_field: str = "id",
    ) -> list[Document]:
        """
        Load documents from a JSON file.

        Args:
            file_path: Path to JSON file
            text_field: Field to use as document content (None = full document)
            id_field: Field to use as document ID
        """
        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return self._load_json_array(data, text_field, id_field, file_path)
        elif isinstance(data, dict):
            return [self._create_document(data, text_field, id_field, file_path)]
        else:
            return []

    def _load_json_array(
        self,
        data: list,
        text_field: str | None,
        id_field: str,
        source: str,
    ) -> list[Document]:
        """Load documents from JSON array."""
        documents = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                doc = self._create_document(item, text_field, id_field, source, i)
                documents.append(doc)
        return documents

    def _create_document(
        self,
        data: dict,
        text_field: str | None,
        id_field: str,
        source: str,
        index: int = 0,
    ) -> Document:
        """Create a document from dict data."""
        if text_field and text_field in data:
            content = str(data[text_field])
        else:
            content = json.dumps(data, indent=2)

        doc_id = data.get(id_field, f"{source}_{index}")

        return Document(
            id=str(doc_id),
            content=content,
            metadata={"source": source, "data": data},
        )
