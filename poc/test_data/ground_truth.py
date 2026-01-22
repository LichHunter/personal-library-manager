import json
import logging
import re
import uuid
from typing import List

import ollama

from .models import Document, GroundTruth, Section

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class GroundTruthGenerator:
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    def _extract_json_from_response(self, response: str) -> list | None:
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return None

    def _classify_difficulty(self, question: str, section_ids: List[str], evidence: List[str]) -> str:
        if len(section_ids) > 1:
            return "hard"
        total_evidence_length = sum(len(e) for e in evidence)
        if total_evidence_length > 500 or len(evidence) > 2:
            return "medium"
        return "easy"

    def _classify_query_type(self, question: str) -> str:
        question_lower = question.lower()
        comparison_keywords = ["compare", "difference", "versus", "vs", "unlike", "similar"]
        if any(kw in question_lower for kw in comparison_keywords):
            return "comparison"
        synthesis_keywords = ["how does", "why does", "explain", "describe the process", "relationship"]
        if any(kw in question_lower for kw in synthesis_keywords):
            return "synthesis"
        return "factual"

    def _build_context(self, doc: Document) -> str:
        lines = [f"Document: {doc.title}", f"Summary: {doc.summary}", ""]
        for section in doc.get_all_sections():
            lines.append(f"[Section ID: {section.id}] {section.heading}")
            lines.append(section.content)
            lines.append("")
        return "\n".join(lines)

    # Common stopwords to filter out
    STOPWORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'in', 'to',
        'for', 'with', 'on', 'at', 'by', 'from', 'as', 'into', 'through',
        'and', 'or', 'but', 'if', 'then', 'than', 'so', 'that', 'this', 'it',
        'its', 'also', 'such', 'which', 'who', 'what', 'when', 'where', 'how',
        'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
        'any', 'no', 'not', 'only', 'own', 'same', 'just', 'now', 'new', 'used'
    })

    def _get_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text."""
        words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        return words - self.STOPWORDS

    def _find_best_evidence(self, doc: Document, answer: str) -> tuple[str | None, str | None]:
        """
        Find the best evidence sentence across all sections.
        Returns (section_id, evidence_sentence) or (None, None) if not found.
        """
        answer_keywords = self._get_keywords(answer)
        if len(answer_keywords) < 2:
            return None, None

        best_match = None  # (score, section_id, sentence)

        for section in doc.get_all_sections():
            if not section.content:
                continue

            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', section.content)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30:  # Skip very short sentences
                    continue

                sentence_keywords = self._get_keywords(sentence)
                overlap = answer_keywords & sentence_keywords

                # Score: number of matching keywords
                score = len(overlap)

                # Bonus for having multiple key answer terms
                if score >= 3 and (best_match is None or score > best_match[0]):
                    best_match = (score, section.id, sentence)

        if best_match and best_match[0] >= 3:
            return best_match[1], best_match[2]

        return None, None

    def generate(self, doc: Document, num_questions: int = 5) -> List[GroundTruth]:
        logging.info(f"Generating {num_questions} Q&A pairs for document: {doc.title}")

        context = self._build_context(doc)
        section_ids = [s.id for s in doc.get_all_sections()]

        prompt = f"""Generate exactly {num_questions} question-answer pairs from the document below.

For each question:
1. The answer must be directly supported by the text
2. Include the section ID(s) where the answer is found
3. The evidence field can be a brief description - it will be validated later

Document:
{context}

Available section IDs: {section_ids}

Return a JSON array:
[
  {{
    "question": "The question text",
    "answer": "A complete answer based on the document",
    "section_ids": ["sec_1"],
    "evidence": ["brief description of supporting evidence"]
  }}
]

Generate {num_questions} diverse questions covering different sections.
Return ONLY the JSON array."""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": 2000,
                    "temperature": 0.3,
                },
            )
            raw_response = response["response"].strip()
        except Exception as e:
            logging.error(f"Ollama generation failed: {e}")
            return []

        parsed = self._extract_json_from_response(raw_response)
        if not parsed:
            logging.warning("Failed to parse LLM response as JSON")
            return []

        ground_truths = []
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue

            question = item.get("question", "")
            answer = item.get("answer", "")
            raw_section_ids = item.get("section_ids", [])

            if not question or not answer:
                continue

            # Find the best evidence from the document
            found_section_id, evidence_sentence = self._find_best_evidence(doc, answer)

            # Skip if we couldn't find any real evidence
            if not found_section_id or not evidence_sentence:
                logging.warning(f"Could not extract evidence for question: {question[:50]}...")
                continue

            # Use the section where evidence was actually found
            valid_section_ids = [found_section_id]
            evidence = [evidence_sentence]

            difficulty = self._classify_difficulty(question, valid_section_ids, evidence)
            query_type = self._classify_query_type(question)

            gt = GroundTruth(
                id=f"gt_{doc.id}_{i+1}",
                question=question,
                answer=answer,
                document_id=doc.id,
                section_ids=valid_section_ids,
                evidence=evidence,
                difficulty=difficulty,
                query_type=query_type,
            )
            ground_truths.append(gt)

        logging.info(f"Generated {len(ground_truths)} ground truth entries")
        return ground_truths

    def generate_for_documents(
        self,
        documents: List[Document],
        questions_per_doc: int = 5,
    ) -> List[GroundTruth]:
        all_ground_truths = []
        for doc in documents:
            gts = self.generate(doc, questions_per_doc)
            all_ground_truths.extend(gts)
        return all_ground_truths
