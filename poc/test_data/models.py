from dataclasses import dataclass, field
from typing import List, Protocol


@dataclass
class Section:
    id: str
    heading: str
    level: int
    content: str
    subsections: List['Section'] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "heading": self.heading,
            "level": self.level,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Section':
        return cls(
            id=data["id"],
            heading=data["heading"],
            level=data["level"],
            content=data["content"],
            subsections=[cls.from_dict(s) for s in data.get("subsections", [])],
        )


@dataclass
class Document:
    id: str
    title: str
    source: str
    summary: str
    sections: List[Section]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Document':
        return cls(
            id=data["id"],
            title=data["title"],
            source=data["source"],
            summary=data["summary"],
            sections=[Section.from_dict(s) for s in data.get("sections", [])],
        )

    def get_all_sections(self) -> List[Section]:
        result = []
        def collect(sections: List[Section]):
            for s in sections:
                result.append(s)
                collect(s.subsections)
        collect(self.sections)
        return result

    def get_section_by_id(self, section_id: str) -> Section | None:
        for section in self.get_all_sections():
            if section.id == section_id:
                return section
        return None


@dataclass
class GroundTruth:
    id: str
    question: str
    answer: str
    document_id: str
    section_ids: List[str]
    evidence: List[str]
    difficulty: str
    query_type: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "document_id": self.document_id,
            "section_ids": self.section_ids,
            "evidence": self.evidence,
            "difficulty": self.difficulty,
            "query_type": self.query_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'GroundTruth':
        return cls(
            id=data["id"],
            question=data["question"],
            answer=data["answer"],
            document_id=data["document_id"],
            section_ids=data["section_ids"],
            evidence=data["evidence"],
            difficulty=data["difficulty"],
            query_type=data["query_type"],
        )


@dataclass
class RetrievalResult:
    test_id: str
    question: str
    expected_doc: str
    retrieved_docs: List[str]
    doc_found: bool
    doc_rank: int
    expected_sections: List[str]
    retrieved_sections: List[str]
    section_found: bool
    evidence_overlap: float

    def to_dict(self) -> dict:
        return {
            "test_id": self.test_id,
            "question": self.question,
            "expected_doc": self.expected_doc,
            "retrieved_docs": self.retrieved_docs,
            "doc_found": self.doc_found,
            "doc_rank": self.doc_rank,
            "expected_sections": self.expected_sections,
            "retrieved_sections": self.retrieved_sections,
            "section_found": self.section_found,
            "evidence_overlap": self.evidence_overlap,
        }


@dataclass
class EvaluationReport:
    total_tests: int
    doc_recall_at_1: float
    doc_recall_at_3: float
    doc_recall_at_5: float
    section_recall_at_1: float
    section_recall_at_3: float
    section_recall_at_5: float
    mean_evidence_overlap: float
    failures: List[RetrievalResult]

    def to_dict(self) -> dict:
        return {
            "total_tests": self.total_tests,
            "doc_recall_at_1": self.doc_recall_at_1,
            "doc_recall_at_3": self.doc_recall_at_3,
            "doc_recall_at_5": self.doc_recall_at_5,
            "section_recall_at_1": self.section_recall_at_1,
            "section_recall_at_3": self.section_recall_at_3,
            "section_recall_at_5": self.section_recall_at_5,
            "mean_evidence_overlap": self.mean_evidence_overlap,
            "failures": [f.to_dict() for f in self.failures],
        }


@dataclass
class SearchResult:
    document_id: str
    section_id: str
    content: str
    score: float


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]: ...
