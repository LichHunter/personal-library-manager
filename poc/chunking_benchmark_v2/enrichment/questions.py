"""Question generation enricher."""

import re
from typing import Optional

from . import Enricher, EnrichmentResult, call_llm

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
MIN_WORDS_FOR_LLM = 30

QUESTION_PROMPT = """Read this documentation text and generate 3 questions that it answers. Keep questions concise and natural, like how a user would ask.

Text:
{content}

Questions (one per line):"""


def _calculate_code_ratio(text: str) -> float:
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _word_count_without_code(text: str) -> int:
    text_without_code = CODE_BLOCK_PATTERN.sub(" ", text)
    return len(text_without_code.split())


class QuestionEnricher(Enricher):
    def __init__(
        self,
        model: str = "llama3.2:3b",
        questions_per_chunk: int = 3,
        code_ratio_threshold: float = 0.5,
    ):
        super().__init__(model)
        self.questions_per_chunk = questions_per_chunk
        self.code_ratio_threshold = code_ratio_threshold

    @property
    def enrichment_type(self) -> str:
        return "questions"

    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        if not content.strip():
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
                questions=[],
            )

        code_ratio = _calculate_code_ratio(content)
        word_count = _word_count_without_code(content)
        use_llm = (
            code_ratio < self.code_ratio_threshold and word_count >= MIN_WORDS_FOR_LLM
        )

        if use_llm:
            questions = self._generate_questions(content)
            questions = questions[: self.questions_per_chunk]
        else:
            questions = []

        if questions:
            questions_text = "\n".join(f"Q: {q}" for q in questions)
            enhanced = f"{questions_text}\n\n{content}"
        else:
            enhanced = content

        return EnrichmentResult(
            original_content=content,
            enhanced_content=enhanced,
            enrichment_type=self.enrichment_type,
            questions=questions,
            metadata={
                "code_ratio": round(code_ratio, 2),
                "used_llm": use_llm,
                "word_count": word_count,
            },
        )

    def _generate_questions(self, content: str) -> list[str]:
        prompt = QUESTION_PROMPT.format(content=content[:1500])
        response = call_llm(prompt, self.model)

        questions = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                parts = line.split(". ", 1)
                if len(parts) > 1:
                    questions.append(parts[1])
            elif "?" in line:
                questions.append(line)

        return questions
