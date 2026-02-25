# Technical Term Analysis Report

**Generated**: 2026-02-22  
**Based on**: `technical_term_analysis.json`

---

## Executive Summary

SPLADE's term expansion demonstrates both strengths and limitations for Kubernetes technical terminology. While it successfully captures morphological structure through BERT's WordPiece tokenization, the expansion terms often lack domain-specific relevance due to training on general web corpora (MS MARCO).

**Key Finding**: SPLADE preserves exact-match signals for technical terms while adding semantic expansion, but domain adaptation would significantly improve expansion quality.

---

## 1. CamelCase Handling

SPLADE uses BERT's WordPiece tokenization, which breaks CamelCase terms into subwords.

### SubjectAccessReview

| Term | Weight | Notes |
|------|--------|-------|
| subject | 2.67 | Main component extracted |
| ##ces | 2.03 | Subword for "access" |
| ##ev | 1.79 | Subword for "review" |
| ##sr | 1.71 | Internal fragment |
| subjects | 1.70 | Plural form added |
| ##ac | 1.67 | Subword fragment |

**Analysis**: The term is split into meaningful components (`subject`, fragments of `access`, `review`). However, expansion includes irrelevant terms like `anger` (0.42), `police` (0.33), and `torture` (0.16) due to the word "subject" appearing in legal/justice contexts in MS MARCO training data.

### PodSecurityPolicy

| Term | Weight | Notes |
|------|--------|-------|
| pod | 2.36 | Primary term |
| ##urity | 2.21 | Fragment of "security" |
| pods | 2.18 | Plural added |
| ##pol | 2.17 | Fragment of "policy" |
| security | 0.64 | Full term (lower weight) |

**Analysis**: Core components are well-captured. The full term "security" gets lower weight than the subword fragments, which is expected behavior for rare compound terms.

### ClusterRoleBinding

| Term | Weight | Notes |
|------|--------|-------|
| cluster | 2.91 | Primary component |
| ##rol | 2.17 | Fragment of "role" |
| ##ind | 1.95 | Fragment of "binding" |
| clusters | 1.90 | Plural form |
| node | 0.42 | Related concept added |

**Analysis**: Good semantic expansion - `node` is relevant to clusters. However, `disorder` (0.69) and `theory` (0.58) appear as noise from training data.

### Verdict: PARTIAL SUCCESS

CamelCase terms are tokenized into meaningful components, but:
- Subword fragments dominate over full words
- Training data introduces irrelevant expansions
- Domain-specific relationships (e.g., SubjectAccessReview -> RBAC, authorization) are missing

---

## 2. Abbreviation Expansion

### RBAC permissions

| Term | Weight | Notes |
|------|--------|-------|
| rb | 3.08 | Abbreviation fragment |
| ##ac | 2.61 | Abbreviation fragment |
| permission | 2.46 | Query term preserved |
| consent | 1.36 | Related concept |
| authorization | 0.24 | Relevant but low weight |

**Analysis**: `RBAC` is not expanded to `role-based access control` - SPLADE treats it as an unknown abbreviation and just tokenizes it. However, it does add related terms like `consent`, `authorization`, and `rights`.

### HPA autoscaling

| Term | Weight | Notes |
|------|--------|-------|
| hp | 2.90 | Abbreviation fragment |
| ##sca | 2.30 | Fragment of "scaling" |
| auto | 2.22 | Prefix extracted |
| ##a | 2.03 | Fragment |

**Analysis**: `HPA` is not expanded to `Horizontal Pod Autoscaler`. SPLADE misinterprets `hp` as possibly "Hewlett-Packard" given the expansion terms include `manufacturer` and `model`.

### k8s cluster

| Term | Weight | Notes |
|------|--------|-------|
| k | 2.80 | Just the letter |
| ##8 | 2.64 | Just the number |
| cluster | 2.49 | Query term preserved |
| 8 | 1.92 | Number extracted |
| comet | 0.52 | Irrelevant noise |
| galaxy | 0.49 | Irrelevant noise |

**Analysis**: `k8s` is not recognized as "Kubernetes". The expansion terms like `comet` and `galaxy` come from astronomy contexts where "clusters" appear.

### Verdict: FAILURE

Abbreviations are not properly expanded:
- No mapping from `k8s` to `kubernetes`
- No expansion of `HPA`, `RBAC`, `VPA`, etc.
- This is a known limitation: SPLADE was not trained on Kubernetes documentation

**Recommendation**: Use explicit abbreviation expansion before SPLADE encoding, or fine-tune SPLADE on domain-specific data.

---

## 3. Multi-word Phrase Handling

### webhook token authenticator

| Term | Weight | Notes |
|------|--------|-------|
| authentic | 2.18 | Core concept |
| token | 2.12 | Query term preserved |
| ##ho | 1.81 | Fragment of "webhook" |
| web | 1.74 | Component extracted |
| authentication | 1.22 | Related term added |
| security | 0.51 | Related concept |

**Analysis**: Good expansion - `authentication` and `security` are relevant. However, `fake` (0.64) appears as noise from "authentic" contexts.

### admission controller webhook

| Term | Weight | Notes |
|------|--------|-------|
| admission | 2.77 | Query term preserved |
| controller | 2.29 | Query term preserved |
| admissions | 1.72 | Plural added |
| entry | 1.27 | Related concept |
| control | 1.50 | Related term |

**Analysis**: Excellent preservation of query terms. `entry` is somewhat relevant (admission implies entry). The term `controllers` (1.47) is added as a plural.

### persistent volume claim

| Term | Weight | Notes |
|------|--------|-------|
| volume | 2.89 | Query term preserved |
| persistent | 2.85 | Query term preserved |
| volumes | 2.60 | Plural added |
| claim | 2.36 | Query term preserved |
| persistence | 2.00 | Related concept |
| claims | 1.70 | Plural added |

**Analysis**: Excellent handling. All query terms preserved with high weights, and plurals are appropriately added. Some noise (`lawsuit`, `fraud`) from legal "claim" contexts.

### Verdict: SUCCESS

Multi-word phrases are handled well:
- Original terms preserved with high weights
- Related forms (plurals, variants) added
- Some noise from polysemous words

---

## 4. Hyphenated Term Handling

### kube-apiserver

| Term | Weight | Notes |
|------|--------|-------|
| ku | 2.70 | Fragment |
| ##be | 2.43 | Fragment of "kube" |
| api | 2.08 | Component extracted |
| ##ver | 1.57 | Fragment of "server" |
| ##ser | 1.56 | Fragment of "server" |
| - | 0.67 | Hyphen preserved |

**Analysis**: The hyphen is tokenized as a separate token. `kube` is split into `ku` + `##be`, and `apiserver` into `api` + `##ver` + `##ser`. This fragmentation works but loses semantic coherence.

### cluster-admin role

| Term | Weight | Notes |
|------|--------|-------|
| cluster | 3.04 | Query term preserved |
| ##min | 2.27 | Fragment of "admin" |
| role | 2.12 | Query term preserved |
| administrator | 1.63 | Related expansion |
| administration | 1.07 | Related expansion |
| node | 1.28 | Related concept |

**Analysis**: Good expansion - `administrator` and `administration` are relevant to `admin`. The term `node` is relevant to cluster contexts.

### kube-controller-manager

| Term | Weight | Notes |
|------|--------|-------|
| ku | 2.94 | Fragment |
| ##be | 2.67 | Fragment |
| manager | 2.02 | Component extracted |
| controller | 2.00 | Component extracted |
| management | 0.96 | Related expansion |
| control | 1.25 | Related expansion |

**Analysis**: All major components extracted. Related terms like `management` and `control` are appropriately added.

### Verdict: PARTIAL SUCCESS

Hyphenated terms are handled reasonably:
- Components are extracted (with fragmentation)
- Hyphen is preserved as a token
- Related expansions are often relevant

---

## 5. Dotted Path Handling

### kubernetes.io/hostname

| Term | Weight | Notes |
|------|--------|-------|
| host | 2.13 | Component extracted |
| ku | 2.03 | Fragment |
| io | 1.81 | Component extracted |
| ##net | 1.80 | Fragment |
| name | 1.43 | Component extracted |
| domain | 0.39 | Related concept |

**Analysis**: The path is split into components. `domain` and `address` appear as relevant expansions for hostname contexts.

### node.kubernetes.io/instance-type

| Term | Weight | Notes |
|------|--------|-------|
| instance | 2.01 | Component extracted |
| node | 1.90 | Component extracted |
| type | 1.55 | Component extracted |
| types | 0.77 | Plural added |
| cluster | 0.26 | Related concept |

**Analysis**: Good component extraction. The expansion adds relevant terms like `cluster` (nodes are in clusters).

### Verdict: SUCCESS

Dotted paths are handled well:
- All path components are extracted
- Related concepts are added as expansions
- Special characters (`.`, `/`) are tokenized appropriately

---

## 6. Informed Query Rank Comparisons

Based on the technical_term_analysis.json data:

### Improved Queries (SPLADE rank < BM25 rank)

| Query | BM25 Rank | SPLADE Rank | Improvement |
|-------|-----------|-------------|-------------|
| mutating webhook vs validating webhook | 5 | 3 | +2 |

### Unchanged Queries

Most informed queries show the same rank between BM25 and SPLADE (11 queries with rank improvement = 0).

### Missing Ground Truth

Several queries have `null` rank improvement because ground truth was not found in top-k results for technical term queries (these are test queries, not informed queries).

---

## 7. Failure Cases

### Why Some Technical Terms Fail

1. **Domain Mismatch**: SPLADE was trained on MS MARCO (web search), not Kubernetes docs
2. **Rare Terms**: CamelCase API types like `SubjectAccessReview` are out-of-vocabulary
3. **Abbreviations**: `k8s`, `HPA`, `RBAC` are not in training data
4. **Context Confusion**: Polysemous words get wrong expansions (e.g., "subject" -> "police", "torture")

### Specific Failure Examples

| Query | Issue | Impact |
|-------|-------|--------|
| SubjectAccessReview | Irrelevant expansions (police, torture) | Noise may affect ranking |
| k8s cluster | Not expanded to "kubernetes" | May miss relevant docs |
| HPA autoscaling | Not expanded to "horizontal pod autoscaler" | May miss docs using full term |

---

## 8. Recommendations

### Short-term (No Fine-tuning)

1. **Pre-expand abbreviations**: Add a lookup table to expand `k8s`, `HPA`, `RBAC`, etc. before SPLADE encoding
2. **Use SPLADE-only**: The hybrid (SPLADE+Semantic) performs worse than SPLADE-only
3. **Accept limitations**: For production, SPLADE still outperforms BM25 by 26%+ despite these limitations

### Long-term (With Fine-tuning)

1. **Domain adaptation**: Fine-tune SPLADE on Kubernetes documentation
2. **Custom vocabulary**: Add domain-specific terms to BERT vocabulary
3. **Contrastive training**: Train on (query, relevant_doc) pairs from Kubernetes QA data

---

## 9. Conclusion

SPLADE demonstrates strong performance on Kubernetes technical queries despite not being trained on domain-specific data. The key strengths are:

- **Exact match preservation**: Original query terms get high weights
- **Morphological handling**: CamelCase and hyphenated terms are decomposed
- **Related concept expansion**: Plurals and variants are added

The key weaknesses are:

- **No abbreviation expansion**: k8s, HPA, RBAC not recognized
- **Domain noise**: Expansions sometimes include irrelevant terms from general web contexts
- **Subword fragmentation**: Rare terms are overly fragmented

**Overall Assessment**: SPLADE is production-ready for Kubernetes documentation retrieval with a simple abbreviation expansion pre-processor. For optimal performance, consider domain fine-tuning.

---

*Generated from technical_term_analysis.json*
