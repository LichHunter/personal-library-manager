import hashlib
import logging
from typing import List

import wikipediaapi

from ..models import Document, Section

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class WikipediaSource:
    def __init__(self, language: str = "en", user_agent: str = "RAGTestDataGenerator/1.0"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent=user_agent,
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

    def _generate_doc_id(self, title: str) -> str:
        return "wiki_" + hashlib.md5(title.lower().encode()).hexdigest()[:8]

    def _parse_sections(
        self,
        wiki_sections: list,
        parent_id: str,
        start_index: int = 1,
    ) -> tuple[List[Section], int]:
        sections = []
        current_index = start_index

        for wiki_section in wiki_sections:
            section_id = f"{parent_id}_{current_index}"
            content = wiki_section.text.strip()

            subsections, current_index = self._parse_sections(
                wiki_section.sections,
                section_id,
                1,
            )

            section = Section(
                id=section_id,
                heading=wiki_section.title,
                level=wiki_section.level,
                content=content,
                subsections=subsections,
            )
            sections.append(section)
            current_index = start_index + len(sections)

        return sections, current_index

    def fetch(self, title: str) -> Document | None:
        logging.info(f"Fetching Wikipedia article: {title}")
        page = self.wiki.page(title)

        if not page.exists():
            logging.warning(f"Article not found: {title}")
            return None

        doc_id = self._generate_doc_id(title)
        summary = page.summary.split("\n")[0] if page.summary else ""

        sections, _ = self._parse_sections(page.sections, f"sec")

        return Document(
            id=doc_id,
            title=page.title,
            source="wikipedia",
            summary=summary,
            sections=sections,
        )

    def fetch_multiple(self, titles: List[str]) -> List[Document]:
        documents = []
        for title in titles:
            doc = self.fetch(title)
            if doc is not None:
                documents.append(doc)
        return documents
