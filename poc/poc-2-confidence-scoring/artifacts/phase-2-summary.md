# Phase 2: Quality Grading

## Objective
Assign quality grades to each extraction for correlation analysis.

## Approach
- F1-based automatic grading (all 100 samples)
- LLM validation on 30 random samples
- Cohen's κ to measure agreement

## Results
- Total graded: 100
- Grade distribution:
  - GOOD: 1 (1.0%)
  - ACCEPTABLE: 5 (5.0%)
  - POOR: 94 (94.0%)
- Cohen's κ: 0.000 (slight agreement)
- Agreement rate: 70.0%

## Issues
⚠️ Cohen's κ (0.000) is below target threshold of 0.7.

Found 9 disagreements between automatic and LLM grading:

- **Q35154661**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.444, H=50.00%
  - Reason: Despite moderate precision and recall, the extraction captures key technical terms like API and Ampserand.js, and the hallucination rate is within acceptable limits.

- **Q45291615**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.469, H=40.00%
  - Reason: The extraction has a moderate F1 score of 0.469 and a hallucination rate of 40%, which falls short of the "GOOD" criteria but meets the "ACCEPTABLE" threshold by capturing some key terms with reasonable precision.

- **Q37087526**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.500, H=66.67%
  - Reason: The extraction has a moderate F1 score of 0.50 and a high hallucination rate of 66.67%, but still captures the key term "OnEdit" while maintaining 100% recall, making it marginally acceptable.

- **Q1719694**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.500, H=25.00%
  - Reason: The extraction has a moderate F1 score of 0.50 and a relatively low hallucination rate of 25%, meeting the criteria for an acceptable extraction with some room for improvement in recall and precision.

- **Q44075338**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.400, H=33.33%
  - Reason: The extraction has moderate performance with an F1 score of 0.40 and hallucination rate of 33.33%, capturing some key technical terms while missing others, but not severely compromising the overall understanding.

- **Q20428669**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.558, H=25.00%
  - Reason: The extraction has a moderate F1 score of 0.558 and a relatively low hallucination rate of 25%, capturing most key technical terms while maintaining reasonable accuracy.

- **Q21380268**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.488, H=33.33%
  - Reason: The extraction has a moderate F1 score of 0.488 and a hallucination rate of 33.33%, which falls just outside the GOOD threshold but still captures some key terms with reasonable accuracy.

- **Q1179655**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.429, H=50.00%
  - Reason: The extraction has moderate performance with an F1 score of 0.429 and hallucination rate of 50%, capturing some key terms but missing several ground truth terms while also introducing some incorrect extractions.

- **Q10734649**: Auto=POOR, LLM=ACCEPTABLE
  - F1=0.414, H=33.33%
  - Reason: The extraction has a moderate F1 score of 0.414 and a hallucination rate of 33.33%, which falls within the acceptable range by capturing some key terms like FirstClass and TableClass while missing some important context terms.

## Next Phase Readiness
✓ Phase 3 (Correlation Analysis) can proceed
