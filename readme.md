# Penalized Regression for High-Dimensional Molecular Toxicity Classification

**Final Year Project | Mathematics and Statistics**

## Project Overview

This repository contains the code, data, and thesis materials for my undergraduate Final Year Project (FYP). The study critically re-evaluates the predictive performance of penalized regression models (Lasso, Ridge, Elastic Net) on the **UCI Toxicity-2 dataset** (CRY1 molecules).

While prior literature claimed high classification accuracy (~79%), this project applies rigorous statistical validation techniques—specifically **Nested Cross-Validation** to mitigate overfitting in this high-dimensional setting ($p \gg n$).

**Key Objectives:**

- Replicate and stress-test findings from the original _Scientific Reports_ paper.
- Address the "Curse of Dimensionality" using regularization.
- Evaluate the impact of class imbalance and multicollinearity on model stability.

## Repository Structure

The project files are organized as follows:

```text
├── data/
│   ├── data.csv                    # Raw dataset (171 molecules, 1203 descriptors)
│   └── Toxicity-13F.csv            # Subset/Processed features
│
├── literature/
│   ├── s41598-021-97962-5.pdf      # Original reference paper
│   └── original_paper.md           # Markdown file of the original paper
│
├── notebooks/                      # Analysis and experimentation
│   ├── toxicity_15.ipynb           # MAIN ANALYSIS: Final model evaluation & Nested CV
│   ├── toxicity_13_colab.ipynb     # Deceptively simple notebook replicating original results
│   └── archive/                    # Exploratory/older notebooks
│
└── thesis/
    ├── 22203133.pdf                # Final thesis document (PDF)
    ├── thesis.md                   # Full thesis text (Markdown source)
    ├── FYP Oral Presentation.pdf   # Final defense slides
    └── drafts/                     # Previous iterations
```

## Methodology

This project utilizes the following statistical frameworks:

1.  **Data Preprocessing:** Standardization and handling of the UCI Toxicity-2 dataset (171 observations, 1203 features).
2.  **Regularization:** Implementation of **Lasso (L1)**, **Ridge (L2)**, and **Elastic Net** to handle multicollinearity.
3.  **Validation:** A **5-fold Nested Cross-Validation** scheme to provide an unbiased estimate of generalization error, avoiding the bias common in simple split-sample validation.

## Key Findings

- **Dimensionality Issues:** The dataset exhibits extreme multicollinearity, which severely affects standard logistic regression models.
- **Performance Re-evaluation:** Under strict Nested CV, the penalized models struggled to significantly outperform random classifiers, contradicting the optimistic results (79% accuracy) reported in previous literature.
- **Conclusion:** The high dimensionality relative to the sample size suggests that feature selection stability is low, and previously reported accuracies may have been inflated by overfitting or data leakage.

## References

- **Dataset:** [UCI Machine Learning Repository: Toxicity-2](https://archive.ics.uci.edu/dataset/728/toxicity-2)
- **Original Paper:** Gul, S., Rahim, F., Isin, S. et al. Structure-based design and classifications of small molecules regulating the circadian rhythm period. Sci Rep 11, 18510 (2021).
  https://doi.org/10.1038/s41598-021-97962-5

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
