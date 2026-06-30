# TTC-LLM-Unit-Test-Evaluation

A research framework for evaluating testing technique awareness in LLM-generated C++ unit tests using the proposed Test Technique Coverage (TTC) metric. The framework evaluates generated tests based on Boundary Value Analysis (BVA) and Equivalence Class Partitioning (ECP) rather than relying solely on traditional structural metrics such as code coverage and compilation success.

---

## Overview

This repository accompanies the Master's thesis:

**"A Metric-Based Evaluation of Testing Technique Awareness in LLM-Generated C++ Unit Tests"**

The objective of this research is to investigate whether GPT-5.5-generated C++ unit tests demonstrate awareness of established software testing techniques.

The proposed **Test Technique Coverage (TTC)** metric measures how well generated unit tests capture:

- Boundary Value Analysis (BVA)
- Equivalence Class Partitioning (ECP)

Unlike traditional evaluation methods that focus on structural metrics, TTC provides a methodology-aware assessment of AI-generated unit tests.

---

## Repository Structure

```
.
├── dataset/                              # Selected CPP-UT-Bench functions
├── prompts/                              # Prompt templates used for GPT-5.5
├── generated_tests/                      # Generated unit tests
├── scripts/                              # Evaluation and preprocessing scripts
├── experimental_results/                 # Experimental results
├── experimental_results/figures/         # Thesis figures
└── README.md
```

*(Update the structure according to your repository.)*

---

## Research Workflow

1. Select representative C++ functions.
2. Create manual BVA and ECP ground-truth annotations.
3. Generate unit tests using GPT-5.5.
4. Evaluate generated tests using the TTC framework.
5. Perform statistical analysis of the results.

---

## Evaluation Metrics

The framework evaluates generated tests using:

- Compilation Success
- Boundary Value Analysis (BVA)
- Equivalence Class Partitioning (ECP)
- Test Technique Coverage (TTC)

---

## Technologies Used

- Python
- C++
- Google Test
- GPT-5.5
- Pandas
- NumPy

---

## Thesis

This repository accompanies the Master's thesis submitted in partial fulfilment of the requirements for the Executive Post Graduate Programme in Machine Learning and Artificial Intelligence.

---

## Citation

If you use this repository in academic work, please cite:

Rohit Chhabra, *A Metric-Based Evaluation of Testing Technique Awareness in LLM-Generated C++ Unit Tests*, Master's Thesis, 2026.

---

## Author

**Rohit Chhabra**

2026