# External Labeled Datasets for Term Extraction Validation

> **Status**: Research complete, integration in progress
> **Date**: 2026-02-09
> **Related**: poc/poc-1b-llm-extraction-improvements/
> **Purpose**: Find human-annotated datasets to validate our extraction pipeline independent of our K8s GT quality issues

## Why External Datasets

Our POC-1b scale testing revealed that **ground truth quality is the bottleneck**:
- 100-chunk test shows 84.1% precision / 94.1% recall / 15.9% hallucination
- But ~60-66% of "false positives" are valid K8s terms the GT missed
- True precision is ~90-93%, true hallucination ~8-10%
- We can't trust our metrics because we can't trust our GT

**Solution**: Validate against professionally human-annotated datasets to prove pipeline quality.

## Confirmed Non-Existent

- No Kubernetes-specific NER dataset exists publicly
- No cloud computing / DevOps NER dataset exists publicly
- No RAG-specific term extraction benchmark exists
- No API/reference documentation keyphrase dataset exists

---

## TIER 1: HIGHLY RELEVANT (Closest to our use case)

### 1. StackOverflow NER (ACL 2020) — BEST MATCH

| Attribute | Value |
|-----------|-------|
| What | 15,372 sentences from StackOverflow, annotated with 20 fine-grained software entity types |
| Entity Types | Library, Framework, Application, Programming Language, Algorithm, Data Structure, API, Tool, Platform, Standard, Protocol, Error Name, Version, License, File Type, Device, OS, Website, Language, UI Element |
| Format | BIO-tagged token-level annotations (CoNLL format) |
| Domain | Software/programming — technical documentation-like text |
| Size | 15,372 sentences, human-annotated |
| Source | github.com/jeniyat/StackOverflowNER |
| Paper | Tabassum et al., ACL 2020 |
| License | MIT |

**Why it's great for us**: The entity types (Library, API, Framework, Tool, Platform, Version, Error Name) are extremely close to what we're extracting from Kubernetes docs. We're looking for things like CrashLoopBackOff (Error Name), kube-apiserver (Tool/Application), v1.28 (Version), Pod (Data Structure/concept). The annotation style is professional human labeling — not LLM-generated.

**Adaptation needed**: Convert from BIO-tagged NER to our term-list format. Filter to relevant entity types. The domain is general software rather than specifically Kubernetes, but the overlap is significant.

---

### 2. SemEval-2017 Task 10: ScienceIE

| Attribute | Value |
|-----------|-------|
| What | Keyphrases extracted from 500 scientific paragraphs, classified into 3 types |
| Entity Types | Process, Task, Material (+ relations: Hyponym-of, Synonym-of) |
| Format | Stand-off annotations (text spans with offsets) |
| Domain | Computer Science, Material Science, Physics |
| Size | 500 paragraphs, 5,738 annotations |
| Source | scienceie.github.io |
| Paper | Augenstein et al., SemEval 2017 |
| License | Free for research |

**Why it's great for us**: Keyphrase extraction from technical paragraphs with typed entities — very close to extracting Kubernetes terms from documentation sections. The Process/Task/Material taxonomy maps partially to our needs (K8s processes like "rolling update", tasks like "scale", materials like "ConfigMap"). Human-annotated by domain experts.

**Adaptation needed**: Domain is scientific papers, not Kubernetes. But the annotation methodology and format are directly applicable.

---

### 3. Inspec Dataset (Hulth, 2003)

| Attribute | Value |
|-----------|-------|
| What | 2,000 abstracts from CS journal papers with controlled and uncontrolled keyphrases |
| Two Keyphrase Types | Controlled (from a thesaurus, like our Tier 1/2 terms) + Uncontrolled (author-assigned, like our Tier 3 terms) |
| Format | Text + keyword lists (exact match) |
| Domain | Computer Science (information retrieval, AI, databases, networks) |
| Size | 2,000 documents, ~29,230 keyphrases total |
| Source | github.com/boudinfl/hulth-2003-pre (preprocessed) |
| License | Research use |

**Why it's great for us**: This is exactly our format — text + keyword list pairs. The controlled/uncontrolled split is very similar to our Tier 1-2 (domain-specific) vs Tier 3 (contextual) distinction. The CS domain overlaps heavily with technical documentation.

**Adaptation needed**: Abstracts are shorter than our K8s doc chunks but similar length. Domain is academic CS, not infrastructure/K8s specifically. But this can validate our extraction pipeline immediately.

---

## TIER 2: VERY RELEVANT (Requires some adaptation)

### 4. OpenKP (Microsoft, EMNLP 2019)

| Attribute | Value |
|-----------|-------|
| What | ~148,000 real web pages with 1-3 expert-annotated keyphrases per document |
| Format | Document text + keyphrase list |
| Domain | Open domain (diverse web pages, including technical docs) |
| Size | 148,124 documents |
| Source | github.com/microsoft/OpenKP, HuggingFace: midas/openkp |
| Paper | Xiong et al., EMNLP 2019 |

**Useful because**: Massive scale, diverse domains (including tech), real web pages. Format is text → keyphrases, exactly what we need. Can filter for technical/documentation-like pages.

**Limitation**: Only 1-3 keyphrases per document (we extract 15-50 per chunk). Web pages are varied quality. But great for validating our pipeline against a gold standard at scale.

---

### 5. KP20k (Meng et al., 2017)

| Attribute | Value |
|-----------|-------|
| What | 567,830 CS papers (title + abstract) with author-assigned keyphrases |
| Format | Text + keyphrase list |
| Domain | Computer Science |
| Size | 567,830 documents |
| Source | HuggingFace: midas/kp20k |

**Useful because**: Absolutely massive. Author-assigned keyphrases are high quality. CS domain overlaps with our K8s use case.

**Limitation**: Author keyphrases are typically 3-8 per paper (less dense than our needs). Papers are not infrastructure docs.

---

### 6. boudinfl/ake-datasets (Meta-Collection)

| Attribute | Value |
|-----------|-------|
| What | Curated collection of 20+ keyphrase extraction benchmark datasets, all preprocessed to common format |
| Datasets Included | Inspec, SemEval-2010, NUS, KDD, WWW, DUC-2001, 500N-KPCrowd, and more |
| Format | XML with tokenized text + keyphrases |
| Source | github.com/boudinfl/ake-datasets |
| License | Apache-2.0 |

**Useful because**: One-stop-shop for many datasets. Preprocessed and standardized. Can benchmark our extraction against established baselines.

---

### 7. LIAAD/KeywordExtractor-Datasets

| Attribute | Value |
|-----------|-------|
| What | Collection of 20 datasets for keyword extraction |
| Datasets | Includes: wiki20, fao30 (agricultural domain), theses100, Pak2018, SemEval, Inspec |
| Format | Text + keyword files |
| Source | github.com/LIAAD/KeywordExtractor-Datasets |

**Useful because**: Multiple domains including Wikipedia articles and theses. wiki20 is Wikipedia-based which shares some structural similarity with Kubernetes documentation.

---

## TIER 3: PARTIALLY RELEVANT (Different domain but useful methodology)

### 8. NIST Keyphrase Extraction for Technical Language Processing

| Attribute | Value |
|-----------|-------|
| What | Training/test data for keyphrase extraction from technical/scientific registries |
| Domain | NIST technical standards and scientific literature |
| Source | data.nist.gov/pdr/lps/ark:/88434/mds2-2161 |

**Useful because**: Technical standards documentation is structurally very similar to Kubernetes API docs (formal, structured, reference-style).

---

### 9. LDKP - Long Document Keyphrases (MIDAS, 2022)

| Attribute | Value |
|-----------|-------|
| What | Keyphrases from full-length CS papers (not just abstracts) |
| Size | Multiple datasets of varying sizes |
| Source | github.com/midas-research/ldkp, HuggingFace: midas/ldkp3k |

**Useful because**: Tests keyphrase extraction on long documents, closer to our use case of extracting from full documentation pages.

---

### 10. PubMedAKE - Author-Assigned Keywords from PubMed

| Attribute | Value |
|-----------|-------|
| What | 843K PubMed articles with author-assigned keywords |
| Size | 843,000+ documents |
| Source | PMC |

**Useful because**: Massive scale, high-quality author-assigned labels. Medical domain but annotation methodology is transferable.

---

## TIER 4: DIFFERENT TASK BUT TRANSFERABLE

### 11. CoNLL-2003 NER

Classic NER dataset (Person, Organization, Location, Misc). Not directly relevant to our domain but useful as a baseline for our evaluation methodology.

### 12. BEIR Benchmark

18 IR datasets for retrieval evaluation. Not keyphrase extraction, but useful if you want to evaluate our term extraction's downstream impact on retrieval quality.

---

## Selected for Validation: StackOverflow NER

**Rationale**: Best domain overlap (software entities), professional human annotations, MIT license, well-documented format, 20 entity types that map directly to our K8s term types.

**Plan**: 
1. Download dataset from GitHub
2. Convert BIO tags → term lists
3. Run our extraction pipeline on 10 sample documents
4. Compare extracted terms vs human annotations
5. Use results to calibrate our pipeline and validate metrics

---

**Research Date**: 2026-02-09
**Status**: COMPLETE — Moving to integration
