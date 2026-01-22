import hashlib
import json
import logging
import re
from typing import List

import ollama

from ..models import Document, Section

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class SyntheticSource:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    def _generate_doc_id(self, topic: str) -> str:
        return "syn_" + hashlib.md5(topic.lower().encode()).hexdigest()[:8]

    def _parse_generated_sections(self, content: str) -> List[Section]:
        sections = []
        current_section = None
        current_content_lines = []
        section_index = 1

        lines = content.strip().split("\n")
        for line in lines:
            heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
            if heading_match:
                if current_section is not None:
                    current_section.content = "\n".join(current_content_lines).strip()
                    sections.append(current_section)
                    current_content_lines = []

                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                current_section = Section(
                    id=f"sec_{section_index}",
                    heading=heading,
                    level=level,
                    content="",
                    subsections=[],
                )
                section_index += 1
            elif current_section is not None:
                current_content_lines.append(line)
            elif line.strip():
                current_section = Section(
                    id=f"sec_{section_index}",
                    heading="Introduction",
                    level=1,
                    content="",
                    subsections=[],
                )
                current_content_lines.append(line)
                section_index += 1

        if current_section is not None:
            current_section.content = "\n".join(current_content_lines).strip()
            sections.append(current_section)

        return sections

    def generate(self, topic: str, num_sections: int = 4) -> Document:
        logging.info(f"Generating synthetic document for topic: {topic}")

        prompt = f"""Write an informative article about "{topic}" with exactly {num_sections} main sections.

Requirements:
- Use markdown headings (## for main sections)
- Each section should have 2-3 paragraphs of factual, detailed content
- Include specific facts, names, dates, and technical details that could be used for Q&A
- Write in an encyclopedic style similar to Wikipedia
- Do not include a title heading, start directly with the first section

Begin the article:"""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": 2000,
                    "temperature": 0.7,
                },
            )
            generated_content = response["response"].strip()
        except Exception as e:
            logging.error(f"Ollama generation failed: {e}")
            generated_content = f"## Overview\n\nContent generation failed for topic: {topic}"

        sections = self._parse_generated_sections(generated_content)

        if not sections:
            sections = [
                Section(
                    id="sec_1",
                    heading="Overview",
                    level=1,
                    content=generated_content,
                    subsections=[],
                )
            ]

        summary = sections[0].content.split("\n")[0] if sections else ""

        return Document(
            id=self._generate_doc_id(topic),
            title=topic,
            source="synthetic",
            summary=summary,
            sections=sections,
        )

    def generate_multiple(self, topics: List[str], num_sections: int = 4) -> List[Document]:
        documents = []
        for topic in topics:
            doc = self.generate(topic, num_sections)
            documents.append(doc)
        return documents
