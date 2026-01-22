import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from .ground_truth import GroundTruthGenerator
from .models import Document, GroundTruth
from .sources.synthetic import SyntheticSource
from .sources.wikipedia import WikipediaSource

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class TestDataGenerator:
    def __init__(self, ollama_model: str = "llama3.2:3b"):
        self.wikipedia_source = WikipediaSource()
        self.synthetic_source = SyntheticSource(model=ollama_model)
        self.ground_truth_generator = GroundTruthGenerator(model=ollama_model)
        self.documents: List[Document] = []
        self.ground_truths: List[GroundTruth] = []

    def add_wikipedia(self, titles: List[str]) -> List[Document]:
        logging.info(f"Adding {len(titles)} Wikipedia articles")
        docs = self.wikipedia_source.fetch_multiple(titles)
        self.documents.extend(docs)
        logging.info(f"Added {len(docs)} documents (total: {len(self.documents)})")
        return docs

    def add_synthetic(self, topics: List[str], num_sections: int = 4) -> List[Document]:
        logging.info(f"Adding {len(topics)} synthetic documents")
        docs = self.synthetic_source.generate_multiple(topics, num_sections)
        self.documents.extend(docs)
        logging.info(f"Added {len(docs)} documents (total: {len(self.documents)})")
        return docs

    def generate_ground_truth(self, questions_per_doc: int = 5) -> List[GroundTruth]:
        logging.info(f"Generating ground truth for {len(self.documents)} documents")
        self.ground_truths = self.ground_truth_generator.generate_for_documents(
            self.documents,
            questions_per_doc,
        )
        logging.info(f"Generated {len(self.ground_truths)} ground truth entries")
        return self.ground_truths

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        docs_dir = output_dir / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)

        for doc in self.documents:
            doc_path = docs_dir / f"{doc.id}.json"
            with open(doc_path, "w") as f:
                json.dump(doc.to_dict(), f, indent=2)
            logging.info(f"Saved document: {doc_path}")

        gt_path = output_dir / "ground_truth.json"
        with open(gt_path, "w") as f:
            json.dump([gt.to_dict() for gt in self.ground_truths], f, indent=2)
        logging.info(f"Saved ground truth: {gt_path}")

        manifest = {
            "created_at": datetime.now().isoformat(),
            "num_documents": len(self.documents),
            "num_ground_truths": len(self.ground_truths),
            "documents": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "source": doc.source,
                    "num_sections": len(doc.get_all_sections()),
                }
                for doc in self.documents
            ],
            "ground_truth_stats": {
                "total": len(self.ground_truths),
                "by_difficulty": {
                    "easy": sum(1 for gt in self.ground_truths if gt.difficulty == "easy"),
                    "medium": sum(1 for gt in self.ground_truths if gt.difficulty == "medium"),
                    "hard": sum(1 for gt in self.ground_truths if gt.difficulty == "hard"),
                },
                "by_query_type": {
                    "factual": sum(1 for gt in self.ground_truths if gt.query_type == "factual"),
                    "synthesis": sum(1 for gt in self.ground_truths if gt.query_type == "synthesis"),
                    "comparison": sum(1 for gt in self.ground_truths if gt.query_type == "comparison"),
                },
            },
        }

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logging.info(f"Saved manifest: {manifest_path}")

    @classmethod
    def load(cls, input_dir: Path, ollama_model: str = "llama3.2:3b") -> "TestDataGenerator":
        input_dir = Path(input_dir)
        generator = cls(ollama_model=ollama_model)

        docs_dir = input_dir / "documents"
        if docs_dir.exists():
            for doc_path in docs_dir.glob("*.json"):
                with open(doc_path) as f:
                    data = json.load(f)
                    generator.documents.append(Document.from_dict(data))

        gt_path = input_dir / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path) as f:
                data = json.load(f)
                generator.ground_truths = [GroundTruth.from_dict(gt) for gt in data]

        logging.info(
            f"Loaded {len(generator.documents)} documents and "
            f"{len(generator.ground_truths)} ground truths"
        )
        return generator
