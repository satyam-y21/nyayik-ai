import os
import re
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Regex for Part (e.g. "PART III", "PART III — FUNDAMENTAL RIGHTS")
        self.part_pattern = re.compile(
            r"^\s*PART\s+([IVXLCDM\d]+[A-Z]*)\b(?:\s*[\:\—\-–]\s*(.*))?",
            re.IGNORECASE,
        )
        # Regex for Chapter (e.g. "CHAPTER XVII", "CHAPTER XVII — OF OFFENCES AGAINST PROPERTY")
        self.chapter_pattern = re.compile(
            r"^\s*CHAPTER\s+([IVXLCDM\d]+[A-Z]*)\b(?:\s*[\:\—\-–]\s*(.*))?",
            re.IGNORECASE,
        )
        # Regex for Schedule headers (e.g. "SEVENTH SCHEDULE", "FIRST SCHEDULE")
        self.schedule_pattern = re.compile(
            r"^\s*(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|"
            r"TENTH|ELEVENTH|TWELFTH|[A-Z]+|\d+)?\s*SCHEDULE\b",
            re.IGNORECASE,
        )

        # Footnote pattern to prevent false article matches from page bottom notes
        self.footnote_pattern = re.compile(
            r"^\s*(?:\d+[\(\[])?\s*\d+\.\s*(?:Ins\.|Subs\.|Sub\.|Added|Omitted|"
            r"Re-numbered|Amended|See|Entry|Entries|Art|Cl|Paragraph|Published|"
            r"The\s+words|Chhit|Nazirganja|Garati|Renumbered|Proviso|Arts|Cls|"
            r"Sub-clause)",
            re.IGNORECASE,
        )

        # Section/Article pattern
        self.section_pattern = re.compile(
            r"^\s*(Section|Article)?\s*(\d+[A-Z]*(?:\(\d+\))?)\s*[\.:]?\s+([A-Z].*)",
            re.IGNORECASE,
        )

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load a single PDF document using PyPDFLoader."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at {file_path}")

        loader = PyPDFLoader(file_path)
        return loader.load()

    def _is_toc_page(self, text: str) -> bool:
        """Heuristic: returns True if this page is a pure Table of Contents page.

        A pure TOC page has ZERO long body lines and many short section-entry lines.
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return False
        toc_entry_lines = sum(
            1 for line in lines if self.section_pattern.match(line) and len(line) < 90
        )
        long_body_lines = sum(1 for line in lines if len(line) > 90)
        return long_body_lines == 0 and toc_entry_lines >= 5

    def process_document(self, file_path: str) -> List[Document]:
        """Load a PDF, split into structural chunks with legal context, format metadata."""
        raw_docs = self.load_pdf(file_path)
        filename = os.path.basename(file_path)
        is_constitution = "constitution" in filename.lower()
        doc_label = (
            "THE CONSTITUTION OF INDIA"
            if is_constitution
            else filename.replace("_", " ").replace(".pdf", "").upper()
        )

        processed_chunks = []
        current_part = "General Context"
        current_section = "General Section"
        current_section_title = ""
        last_section_page = 0
        in_schedule = False

        for doc in raw_docs:
            page_num = doc.metadata.get("page", 0) + 1  # 1-indexed
            text = doc.page_content
            lines = text.split("\n")

            # Skip section/chapter tracking on TOC pages
            if self._is_toc_page(text):
                body = text.strip()
                if body:
                    processed_chunks.extend(
                        self._create_chunks(
                            body,
                            filename,
                            doc_label,
                            page_num,
                            current_part,
                            current_section,
                            current_section_title,
                        )
                    )
                continue

            # Reset section carry-over if more than 1 page away
            if page_num - last_section_page > 1:
                current_section = "General Section"
                current_section_title = ""

            paragraph_buffer = []

            for i, line in enumerate(lines):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                # 1. Check for Schedule Headers
                sched_match = self.schedule_pattern.match(stripped_line)
                if sched_match and len(stripped_line) < 60:
                    if paragraph_buffer:
                        processed_chunks.extend(
                            self._create_chunks(
                                "\n".join(paragraph_buffer),
                                filename,
                                doc_label,
                                page_num,
                                current_part,
                                current_section,
                                current_section_title,
                            )
                        )
                        paragraph_buffer = []
                    in_schedule = True
                    current_part = f"SCHEDULE: {stripped_line}"
                    current_section = "General Section"
                    current_section_title = ""
                    continue

                # 2. Check for Part / Chapter Header changes
                part_match = self.part_pattern.match(
                    stripped_line
                ) or self.chapter_pattern.match(stripped_line)
                if part_match:
                    if paragraph_buffer:
                        processed_chunks.extend(
                            self._create_chunks(
                                "\n".join(paragraph_buffer),
                                filename,
                                doc_label,
                                page_num,
                                current_part,
                                current_section,
                                current_section_title,
                            )
                        )
                        paragraph_buffer = []

                    in_schedule = False
                    p_prefix = (
                        "PART" if "PART" in stripped_line.upper() else "CHAPTER"
                    )
                    p_num = part_match.group(1)
                    p_title = part_match.group(2) or ""

                    # If title is on the next line, grab it
                    if not p_title and i + 1 < len(lines):
                        next_l = lines[i + 1].strip()
                        if len(next_l) < 70 and not self.section_pattern.match(
                            next_l
                        ):
                            p_title = next_l

                    current_part = f"{p_prefix} {p_num}" + (
                        f" — {p_title}" if p_title else ""
                    )
                    continue

                # 3. Ignore footnote lines
                if self.footnote_pattern.match(stripped_line):
                    continue

                # 4. Check for Section/Article changes (main body only)
                if not in_schedule:
                    section_match = self.section_pattern.match(stripped_line)
                    if section_match:
                        if paragraph_buffer:
                            processed_chunks.extend(
                                self._create_chunks(
                                    "\n".join(paragraph_buffer),
                                    filename,
                                    doc_label,
                                    page_num,
                                    current_part,
                                    current_section,
                                    current_section_title,
                                )
                            )
                            paragraph_buffer = []

                        label_type = "article" if is_constitution else "section"
                        sec_num = section_match.group(2)
                        sec_raw_title = (
                            section_match.group(3)
                            .split("—")[0]
                            .split("--")[0]
                            .split(".")[0]
                            .strip()
                        )

                        current_section = f"{label_type} {sec_num}"
                        current_section_title = (
                            f"{label_type.capitalize()} {sec_num} — {sec_raw_title}"
                            if sec_raw_title
                            else f"{label_type.capitalize()} {sec_num}"
                        )
                        last_section_page = page_num

                paragraph_buffer.append(line)

            # Flush any remaining text in this page
            if paragraph_buffer:
                processed_chunks.extend(
                    self._create_chunks(
                        "\n".join(paragraph_buffer),
                        filename,
                        doc_label,
                        page_num,
                        current_part,
                        current_section,
                        current_section_title,
                    )
                )

        return processed_chunks

    def _create_chunks(
        self,
        text: str,
        filename: str,
        doc_label: str,
        page_num: int,
        part: str,
        section: str,
        section_title: str,
    ) -> List[Document]:
        """Create split chunks, prepending structural hierarchy for rich embedding context."""
        sec_context = section_title if section_title else section
        header_context = (
            f"[Document: {doc_label} | Hierarchy: {part} > {sec_context}]\n\n"
        )

        meta = {
            "source_file": filename,
            "doc_title": doc_label,
            "page": page_num,
            "chapter": part,
            "part": part,
            "section": section,
            "section_title": section_title,
        }

        if len(text) <= self.chunk_size:
            doc = Document(
                page_content=header_context + text,
                metadata=meta,
            )
            return [doc]

        sub_splits = self.text_splitter.split_text(text)
        docs = []
        for i, split in enumerate(sub_splits):
            chunk_meta = dict(meta)
            chunk_meta["chunk_idx"] = i
            docs.append(
                Document(
                    page_content=header_context + split,
                    metadata=chunk_meta,
                )
            )
        return docs
