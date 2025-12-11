A Framework for Enhanced High-Dimensional QSAR Modeling: Stabilizing Penalized Regression and Robust Classification for the CRY1 Toxicity Dataset
Phase I: Re-establishing Robust Performance and Validation
1. The Problem of p≫n Toxicity Modeling: Data Context and Challenges
The analysis of the Toxicity Dataset for the CRY1 core clock protein represents a classic, yet challenging, high-dimensional quantitative structure–activity relationship (QSAR) problem, requiring specialized statistical handling to ensure generalizability and interpretability.
1.1 Data Structure and QSAR Context
The dataset is characterized by a small number of observations (N=171 molecules) relative to a significantly larger number of predictors (P=1203 molecular descriptors), defining a high-dimensional regime where P is substantially greater than N (p≫n).[1, 2] This structure inherently makes classical statistical inference difficult, as parameter estimation becomes unstable and non-unique, leading to a high risk of overfitting the training data.
The core objective of QSAR is to establish a functional relationship (A 
pred
​
 =f(D 
1
​
 ,D 
2
​
 ,…,D 
n
​
 )) between the biological activity (toxicity endpoint) and the chemical or structural properties represented by the molecular descriptors.[3] The 1,203 descriptors in this dataset, ranging from topological and constitutional descriptors to electronic and geometric features [4], constitute a complete set that requires feature selection because they are often highly redundant and correlated.[5] This high degree of multicollinearity is not an artifact but an intrinsic property of large descriptor sets, and it severely complicates the interpretation of linear models.
1.2 Penalized Regression as an Embedded Feature Selection Strategy
Penalized regression models (shrinkage or regularization) address the overfitting issue typical in high-dimensional settings by adding a penalty term to the least squares error function. The use of the Least Absolute Shrinkage and Selection Operator (Lasso), Ridge regression, and Elastic Net is particularly appropriate here. Lasso, utilizing the L 
1
​
  penalty, is uniquely suited for performing embedded variable selection by driving the coefficients of irrelevant or redundant predictors exactly to zero, resulting in a sparse model.[6]
The primary goal of this research project extends beyond mere predictive accuracy; it aims to identify an interpretable model structure.[7, 8] In the p≫n setting, the instability of parameter estimates means that the variable selection process itself is susceptible to the slightest noise or changes in data subsets.[9, 10] The necessity of imposing constraints through regularization parameter λ is a statistical necessity to achieve a sparse solution, but this approach demands subsequent validation regarding the stability of the selected feature set, which is addressed in Phase II.
1.3 The Challenge of Class Imbalance
The dataset exhibits a significant class imbalance: 115 NonToxic molecules (67.25%) and 56 Toxic molecules (32.75%) [User Query]. This distribution is typical for high-throughput screening (HTS) QSAR assays, where active (toxic) compounds are rare compared to inactive ones.[11] Handling this imbalance is critical, as standard classification models tend to favor the larger majority class, resulting in poor prediction performance for the minority (Toxic) class.[12]
2. Transitioning from Accuracy to Robust Classification Metrics
The initial observation that the Lasso model achieved an accuracy (68.57%) matching that of a complex non-linear model (SVM) is statistically suspect given the inherent data imbalance. Reporting classification performance solely through accuracy in this context renders the model quality assessment misleading.
2.1 The Bias of Accuracy in Imbalanced Data
With a majority class prevalence of 67.25%, a trivial model that simply predicts all molecules are NonToxic would achieve 67.25% accuracy. The reported 68.57% accuracy for the Lasso model, while seemingly satisfactory, offers only a marginal improvement over this baseline, indicating that the model may not be adequately learning the patterns specific to the minority (Toxic) class.[13] This common statistical pitfall necessitates the adoption of metrics that are independent of, or robust to, the prevalence of the classes in the test set.
2.2 Mandate for Prevalence-Independent Metrics (PIMs)
For published QSAR models handling binary classification with imbalance, moving beyond simple accuracy to robust PIMs is mandatory.
• Matthews Correlation Coefficient (MCC): The MCC is highly recommended as the primary comparative metric.[14] Unlike accuracy, MCC is derived from all four entries of the confusion matrix (True Positives, True Negatives, False Positives, and False Negatives). A high MCC value signifies that the classifier performs well across all classes, correctly predicting both positive (Toxic) and negative (NonToxic) instances. It is specifically designed to mitigate the effects of imbalanced test sets on the model's perceived performance.[13]
• Precision-Recall Area Under the Curve (PR-AUC): The PR-AUC should serve as the secondary metric, focusing specifically on the model's ability to identify the rare positive cases (Toxic molecules).[15, 16] Research demonstrates that the commonly used Receiver Operating Characteristic Area Under the Curve (ROC-AUC) tends to strongly overestimate classification performance on highly imbalanced datasets, potentially masking poor performance on the minority class. PR-AUC provides a more realistic and informative measure of retrieval performance.[15, 16]
• Balanced Accuracy (BACC): Defined as the average of sensitivity (recall of the positive class) and specificity (recall of the negative class), BACC is another valuable metric used to assess global performance, specifically calibrated to assess performance as if the prevalence were 50%.[13]
2.3 Re-evaluating Class Weighting
The initial finding that training weighted models (where weights are inversely proportional to class size) resulted in "No significant improvement to accuracy" [User Query] is expected if accuracy remains the sole evaluation metric. Class weighting attempts to improve the recall of the minority class, often at the expense of overall accuracy. The PIMs—specifically MCC and PR-AUC—are the necessary tools to reveal if this strategic trade-off resulted in a genuinely better model structure or improved capture of Toxic compounds, which are the critical outcome in QSAR toxicity studies.
3. Advanced Hyperparameter Tuning: Mitigating Overfitting via Nested Cross-Validation
The current model training, which allowed a linear model (Lasso) to match a complex non-linear model (SVM) in accuracy, suggests that the performance metrics may suffer from selection bias—a critical vulnerability in high-dimensional modeling.
3.1 The Danger of Optimization Bias in p≫n Data
In small-sample, high-dimensional settings, determining the optimal regularization parameter (λ in Lasso/Ridge or λ and ρ in Elastic Net) is typically done via cross-validation (CV). If the performance reported is derived from the same CV procedure used to select the optimal parameters, the resulting generalization error estimate will be overly optimistic.[1] This optimism bias is particularly severe in QSAR studies where the sample size (N=171) is low, and the search space for predictive features is enormous.
The statistical necessity is to ensure that the reported performance metrics (MCC and PR-AUC) genuinely reflect the model's performance on unseen data, free from optimization bias.
3.2 Mandatory Implementation of Nested Cross-Validation (Nested CV)
To mitigate optimization bias and provide statistically robust generalization estimates, the methodology must be elevated to Nested Cross-Validation.
The Nested CV procedure utilizes two distinct loops:
1. Inner Loop: This loop is dedicated exclusively to hyperparameter optimization (e.g., searching for the optimal λ in Lasso or the optimal (λ,ρ) pair in Elastic Net). This loop minimizes the inner error estimate (e.g., the CV error).
2. Outer Loop: This loop uses the hyperparameters selected by the inner loop to train a final model on the inner training fold, and then evaluates its performance on the completely held-out outer test fold. The average performance across all outer folds provides an unbiased estimate of the model's true generalization error.
By contrasting the initial, potentially "optimistic" performance (such as the reported 68.57% accuracy) with the "unbiased" performance metrics derived from the Nested CV outer loop (MCC and PR-AUC), the degree of overfitting inherent in the small-sample, high-dimensional QSAR setup can be statistically quantified.
3.3 Robust Hyperparameter Search for Elastic Net
For the Elastic Net model, which combines the L 
1
​
  and L 
2
​
  penalties, the search space involves two critical hyperparameters: the mixing ratio (ρ, balancing L 
1
​
  and L 
2
​
 ) and the overall penalty strength (λ or α). Given the likely high multicollinearity in the 1203 molecular descriptors, Elastic Net is expected to perform superiorly to Lasso because it encourages the selection of correlated feature groups.[17] Rigorous tuning across a dense grid for both ρ and λ is essential, exploring the range from pure Lasso (ρ=1) to intermediate values where the stabilizing L 
2
​
  component takes effect.
Phase II: Resolving Feature Selection Instability and Divergence
The student's central observation—that the models selected a consistent set of features different from the 13 descriptors identified in the original study—is a critical finding that points directly to the statistical pathologies of feature selection in p≫n QSAR, specifically the effects of multicollinearity and differing selection philosophies.
4. The Multicollinearity Crisis: Explaining Feature Selection Divergence
The inherent redundancy within the 1203 molecular descriptors leads to high multicollinearity. This environment dictates how L 
1
​
  and L 
2
​
  penalties behave, directly explaining the feature selection differences observed.
4.1 Lasso’s Arbitrary Selection in Highly Correlated Groups
Lasso regression (employing the L 
1
​
  penalty) is designed to produce sparse models, achieving variable selection by setting many coefficients exactly to zero. However, when faced with a group of highly correlated features that provide similar predictive information (a common occurrence in comprehensive descriptor sets), Lasso tends to arbitrarily select only one feature from that group and dismiss the rest.[18, 19]
Because the selection of which single feature to retain is arbitrary, sensitive to the precise training data split or minute changes in the regularization path, Lasso’s feature selection is highly unstable. The student’s finding of a "consistent set of another features" that performs equally well validates the fact that this new set is likely comprised of alternative representatives from the same underlying correlated clusters that the original study identified. While this new feature set is statistically sufficient for prediction, its individual components lack intrinsic stability or robust chemical interpretation.[9] The goal of the analysis must therefore differentiate between features selected primarily for high prediction power and those selected for robust causal interpretation.[7, 8]
4.2 The Superiority of Elastic Net for QSAR Feature Groups
Elastic Net, by incorporating both L 
1
​
  (for sparsity) and L 
2
​
  (for shrinkage and grouping) penalties, provides a crucial advantage in the context of redundant QSAR descriptors. The L 
2
​
  penalty component encourages correlated features to "share the credit" more evenly, leading Elastic Net to select entire groups of highly correlated predictors together, rather than arbitrarily picking just one representative.[17, 18]
This characteristic makes Elastic Net's feature selection mechanism chemically more coherent and statistically more robust than pure Lasso, particularly when P is much larger than N and multicollinearity is expected.[17] For the final interpretive model, Elastic Net should be prioritized, as its resultant feature set is expected to provide a more stable and representative chemical description of the toxicity mechanism.
5. Wrapper vs. Embedded Selection: DTC/RFE vs. Penalized Regression
The non-concurrence between the student's penalized regression features and the 13 molecular descriptors reported in the introductory paper [5] is primarily a consequence of using fundamentally different feature selection philosophies.
5.1 Disparity in Selection Philosophy
The original study utilized Recursive Feature Elimination (RFE) guided by a Decision Tree Classifier (DTC).[5, 20] RFE is classified as a wrapper method, meaning it uses a specific predictive model (DTC) to evaluate successive subsets of features until an optimal set is found.[21] Decision Trees are inherently non-linear models that determine feature importance based on hierarchical splits and information gain, making them effective at capturing local, non-linear interactions.
In contrast, Lasso and Elastic Net are embedded methods.[21] They integrate feature selection directly into the model training process by modifying the objective function. They select features based on their marginal contribution to the model's global linear fit, constrained by the L 
1
​
  or combined L 
1
​
 /L 
2
​
  penalties.
5.2 Causal Reasons for Divergence
The divergence in feature sets is rooted in these methodological differences:
1. Linearity vs. Non-linearity: Penalized regression seeks the best sparse linear approximation of the toxicity outcome. If the underlying structure of the CRY1 toxicity mechanism involves complex, non-linear interactions between molecular descriptors, the DTC/RFE approach will select features optimized for those non-linear boundaries. The linear Lasso model would then select a different, often minimal, set of features that approximate the non-linear relationship linearly.
2. Global vs. Local Optimization: Lasso seeks a global minimum for the penalized linear fit. DTCs derive importance from local splits.
The conclusion must frame the feature mismatch as an expected, justifiable outcome of comparing distinct statistical optimization goals (linear sparsity for penalized models versus non-linear importance for DTC). The comparison of these feature sets, rather than being treated as an error, provides richer insight into the functional form of the QSAR relationship.
6. Mandatory Enhancement: Stability Selection for Robust Feature Sets
To elevate the selected features from merely "consistent across different models" to a scientifically robust and publishable set, the stability of the feature selection process itself must be rigorously quantified and controlled.
6.1 The Instability of Feature Selection in QSAR
Penalized regression techniques are highly sensitive to specific data characteristics, including outliers and unusual observations, which are recognized problems in QSAR modeling.[10] In the P≫N setting, where the feature selection space is vast, even minor perturbations in the training data can cause the Lasso regularization path to select an entirely different, but equally predictive, feature subset—the very definition of selection instability.[9] This instability compromises the interpretability and trustworthiness of the coefficients.
To obtain results suitable for publication, the research must employ advanced techniques that provide control over the error rates associated with variable selection in finite-sample, high-dimensional regimes.[2]
6.2 Operationalizing Stability Selection (SS)
Stability Selection (SS), introduced by Meinshausen and Bühlmann, is a flexible framework that adds error control to high-dimensional variable selection procedures like Lasso by leveraging resampling.[2, 22]
The detailed workflow requires the following steps:
1. Resampling: Repeatedly draw B subsamples (e.g., B=100 or B=1000) of size ⌊n/2⌋ from the full dataset without replacement.[22, 23]
2. Model Fitting: For each subsample, fit the penalized regression model (Lasso or Elastic Net), exploring a broad, pre-specified range of regularization parameters Λ.
3. Frequency Calculation: Calculate the empirical selection frequency ( 
π
^
  
j
​
 ) for each molecular descriptor j, defined as the proportion of subsamples in which that feature received a non-zero coefficient.[22]
4. Thresholding: Define the final, Stable Set  
S
^
  
stable
​
  by applying a frequency threshold π 
thr
​
  (typically 0.6≤π 
thr
​
 ≤0.8).[22] Only features selected above this frequency are considered robust.
The student’s qualitative finding that certain features "appeared consistently across different models" is the empirical evidence that Stability Selection formalizes. Applying SS transforms this qualitative observation into a quantitative, statistically validated statement about feature robustness.
6.3 Quantifying Feature Robustness
A crucial component of the revised thesis is the quantitative comparison of feature stability between Lasso and Elastic Net.[24] Because Elastic Net encourages the selection of correlated groups, its features are often observed to be more stable than those selected by pure Lasso in high-dimensional settings with correlation.[24, 25]
The analysis must present the histograms of selection frequencies ( 
π
^
  
j
​
 ) for both Lasso and Elastic Net. If Elastic Net features demonstrate superior stability (higher  
π
^
  
j
​
  for the final selected subset), it provides strong statistical justification that the inclusion of the L 
2
​
  penalty component is methodologically essential for stabilizing the variable selection process for this specific QSAR dataset. This quantification ensures that the resulting interpretive model is built upon a foundation of statistically defensible predictors.
Phase III: Scientific Interpretation and Publishable Insights
The final phase transforms the rigorous statistical results into a contribution to computational chemistry, providing plausible mechanistic hypotheses regarding CRY1 toxicity.
7. Interpreting Stable Molecular Descriptors and Biological Meaning
The transition from a set of stable mathematical coefficients ( 
β
^
​
  
j
​
 ) to a chemical mechanism requires a deep understanding of the molecular descriptors themselves.
7.1 Categorization and Structural Meaning
The selected molecular descriptors must first be categorized by their physicochemical basis. QSAR descriptors typically fall into categories such as:
• Topological descriptors: Based on molecular graphs, representing atom connectivity, branching, and ring counts.[4, 26]
• Constitutional descriptors: Reflecting molecular composition (e.g., atom counts, molecular weight).[26]
• Electronic descriptors: Quantifying electronic aspects, such as partial atomic charges, HOMO/LUMO energies, or polarizability.[4, 27]
• Geometric descriptors: Calculated from 3D coordinates, capturing size, shape, and surface areas.[4, 26]
For example, a robustly selected descriptor might be "Average complementary information content (order 2) (ACIC2)," which is a topological descriptor that quantifies molecular size, shape, and branching.[28]
7.2 Linking Statistical Coefficients to Chemical Mechanisms
The interpretability of the penalized linear model stems from the sign and magnitude of the estimated coefficients ( 
β
^
​
 ). This is where statistical analysis connects directly to chemical hypothesis generation.
If a specific electronic descriptor, such as the fractional positive surface area (a charged partial surface area descriptor [4]), is robustly selected and exhibits a positive coefficient ( 
β
^
​
 >0), the chemical hypothesis is that increasing this molecular property—i.e., making the molecule more prone to polar interactions via positive charge—increases the probability of toxicity. Conversely, a negative coefficient for a descriptor related to molecular size (a geometric descriptor) would suggest that smaller molecules are more likely to be toxic, perhaps due to easier access to the protein’s binding site.[29]
Interpretation must rely not merely on statistical significance but on developing a plausible chemical rationale consistent with the known properties of the CRY1 core clock protein.[4]
7.3 Developing a Cohesive Toxicity Hypothesis
The final step is the synthesis of the individual descriptor effects into a holistic hypothesis. The collection of stable features selected by the penalized model (e.g., one electronic, one topological, and one geometric descriptor) should collectively define a "toxin profile" that relates to the required steric factors, electronic charge distribution, or connectivity patterns necessary for the molecule to disrupt the CRY1 function.[3, 30]
The fact that the derived feature set differs from the original DTC/RFE study should be framed as a novel contribution. The stability-selected penalized model provides a globally consistent and statistically verified linear relationship that is highly interpretable [31], potentially contrasting with the locally optimal, non-linear feature subset identified by the tree-based method. This provides two distinct, yet valid, chemical interpretations derived from the same data.
8. Comparative Analysis of Predictive and Interpretive Models
To finalize the thesis and adhere to publication standards, a clear separation between models optimized for prediction and models optimized for interpretation must be maintained, supported by the unbiased metrics derived from Nested CV.
8.1 Separation of Objectives
The analysis should establish that prediction is the primary goal of the black-box non-linear models (such as SVM), which, due to their ability to capture complex descriptor interactions, likely yield the highest true generalization MCC and PR-AUC.[12, 32] This performance sets the empirical upper bound for predictive capacity on this dataset.
Interpretation, conversely, is the primary goal of the penalized regression models (Lasso, Elastic Net). The Elastic Net model, utilizing the Stability Selection subset, yields the most parsimonious and scientifically defensible model.[8]
8.2 Final Benchmarking
A comprehensive performance table must be presented, summarizing the unbiased generalization error estimates obtained via Nested CV for all tested models, focusing on the three penalized regression models and the best non-linear model (SVM).
Table 8.2: Comparative Benchmarking of Models (Unbiased Nested CV Results)
Model
Primary Metric (MCC)
Secondary Metric (PR-AUC)
Balanced Accuracy (BACC)
Sparsity (Non-Zero  
β
^
​
 )
Feature Stability Score ( 
π
^
  
min
​
 )
Ridge Regression
X 
R
​
 
Y 
R
​
 
Z 
R
​
 
1203
N/A
Lasso Regression
X 
L
​
 
Y 
L
​
 
Z 
L
​
 
S 
L
​
 
π
^
  
L
​
 
Elastic Net (SS)
X 
EN
​
 
Y 
EN
​
 
Z 
EN
​
 
S 
EN
​
 
π
^
  
EN
​
 
SVM (Best C)
X 
SVM
​
 
Y 
SVM
​
 
Z 
SVM
​
 
1203
N/A
Note: S represents the number of selected features, and  
π
^
  
min
​
  is the lowest selection frequency allowed by the Stability Selection threshold for the final set.
The table must explicitly quantify the trade-off.[33] For instance, it might show that the SVM achieves the highest MCC (X 
SVM
​
 ), but the Elastic Net model (X 
EN
​
 ) achieves a comparable MCC with significantly higher sparsity (S 
EN
​
 ≪1203) and a high stability score ( 
π
^
  
EN
​
 ≥0.7).
8.3 Final Discussion on Model Selection
The discussion should conclude that the best choice of model is determined by the objective. While SVM may achieve the marginally best predictive power, in QSAR and chemometrics, the requirement for an easily interpretable model structure and mechanistic insight often necessitates accepting a minimal loss in predictive accuracy.[7, 31] The Stability-selected Elastic Net provides the optimal balance of high predictive performance, quantifiable robustness, and chemical interpretability.
9. Conclusion: Pathway to Publication and Future Directions
To prepare this refined undergraduate thesis for potential publication in a chemometrics or statistical learning journal, the following rigorous requirements and future avenues of research must be established.
The synthesis of key findings confirms the effective application of penalized regression in a high-dimensional QSAR context. The methodological pivot to Nested Cross-Validation ensured unbiased performance estimates using prevalence-independent metrics (MCC and PR-AUC). Crucially, the mandatory use of Stability Selection transformed the arbitrary feature subset generated by Lasso into a statistically robust, stable feature set, enabling the development of a scientifically plausible linear toxicity hypothesis for the CRY1 protein.
9.1 Publication Requirements: Applicability Domain and Robustness
A critical requirement for any publishable QSAR model is the definition of its Applicability Domain (AD).[31] The thesis must commit to analyzing the AD of the final Elastic Net model to guarantee that the model's predictions are reliable only for molecules that are structurally similar to the training set compounds.
Furthermore, penalized regression methods are known to be sensitive to the presence of outliers in the data.[10] Future work should explore robust alternatives to the standard penalized regression objective function, such as incorporating robust estimation techniques or employing specialized methods like Sparse Least Trimmed Squares (LTS) to confirm that the stability-selected features are not artifacts of unusual observations.
9.2 Future Methodological Extensions
The scope of the project can be expanded by incorporating advanced regularization techniques:
• Adaptive Lasso: This technique uses preliminary estimates of the coefficients to apply differential penalties, potentially resulting in higher prediction consistency and reduced estimation bias.
• Group Lasso: If the 1203 descriptors can be logically grouped (e.g., into topological, electronic, and geometric families), Group Lasso could be used to select entire families of descriptors together, potentially simplifying the interpretive process.[6]
Finally, the highest level of predictive validation requires evaluation against an external test set (a set of molecules not used in any part of the Nested CV process). This step provides the ultimate, unbiased assessment of the model's predictive reliability, ensuring its fitness for broader application in drug design and toxicity screening.
--------------------------------------------------------------------------------
1. On cross-validated Lasso in high dimensions - Project Euclid, https://projecteuclid.org/journals/annals-of-statistics/volume-49/issue-3/On-cross-validated-Lasso-in-high-dimensions/10.1214/20-AOS2000.pdf
2. Summary and discussion of: “Stability Selection” - Statistics & Data Science, https://www.stat.cmu.edu/~ryantibs/journalclub/stability.pdf
3. Quantitative Structure Activity Relationships: An overview, https://www.toxicology.org/groups/ss/MDCPSS/docs/Basics_of_QSAR-PPradeep.pdf
4. MOLECULAR DESCRIPTORS USED IN QSAR - HUFOCW, https://www.hufocw.org/Download/file/16462
5. Toxicity - UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/728/toxicity-2
6. High-Dimensional LASSO-Based Computational Regression Models: Regularization, Shrinkage, and Selection - MDPI, https://www.mdpi.com/2504-4990/1/1/21
7. What are disadvantages of using the lasso for variable selection for regression?, https://stats.stackexchange.com/questions/7935/what-are-disadvantages-of-using-the-lasso-for-variable-selection-for-regression
8. Chapter 11 Model Selection and Sparsity | Causal Inference and Machine Learning: In Economics, Social, and Health Sciences, https://www.causalmlbook.com/model-selection-and-sparsity.html
9. On the Stability of Feature Selection Algorithms - Journal of Machine Learning Research, https://jmlr.csail.mit.edu/papers/volume18/17-514/17-514.pdf
10. Robust hybrid algorithms for regularization and variable selection in QSAR studies, https://journal.nsps.org.ng/index.php/jnsps/article/download/1708/279/8493
11. QSAR Modeling of Imbalanced High-Throughput Screening Data in PubChem, https://pubs.acs.org/doi/10.1021/ci400737s
12. Class-imbalanced classifiers for high-dimensional data | Briefings in Bioinformatics | Oxford Academic, https://academic.oup.com/bib/article/14/1/13/304457
13. Mind your prevalence! - PubMed, https://pubmed.ncbi.nlm.nih.gov/38622648/
14. Averages of Matthews correlation coefficient (MCC) for validation sets... - ResearchGate, https://www.researchgate.net/figure/Averages-of-Matthews-correlation-coefficient-MCC-for-validation-sets-over-all_fig2_391615801
15. ROC AUC vs Precision-Recall for Imbalanced Data ..., https://machinelearningmastery.com/roc-auc-vs-precision-recall-for-imbalanced-data/
16. ROC and precision-recall with imbalanced datasets, https://classeval.wordpress.com/simulation-analysis/roc-and-precision-recall-with-imbalanced-datasets/
17. A Robust Variable Selection Method for Sparse Online Regression via the Elastic Net Penalty - MDPI, https://www.mdpi.com/2227-7390/10/16/2985
18. Multicollinearity and Regularization in Regression Models | by Dilip ..., https://dilipkumar.medium.com/multicollinearity-and-regularization-in-regression-models-25c24b9107a7
19. Understanding L1 and L2 regularization: techniques for optimized model training - Wandb, https://wandb.ai/mostafaibrahim17/ml-articles/reports/Understanding-L1-and-L2-regularization-techniques-for-optimized-model-training--Vmlldzo3NzYwNTM5
20. Toxicity - UCI Machine Learning Repository - UC Irvine, https://archive-beta.ics.uci.edu/dataset/728/toxicity-2
21. Feature selection - Wikipedia, https://en.wikipedia.org/wiki/Feature_selection
22. Controlling false discoveries in high-dimensional situations: boosting with stability selection - PMC - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC4464883/
23. On the Selection Stability of Stability Selection and Its Applications - arXiv, https://arxiv.org/html/2411.09097v1
24. Utilizing stability criteria in choosing feature selection methods yields reproducible results in microbiome data - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC9787628/
25. The Performance of Feature Selection with the Stability Selection Method in Human Activity Recognition (HAR) Classification Using the Lasso Regression Algorithm - IEEE Xplore, https://ieeexplore.ieee.org/document/11087273/
26. Untitled, https://www.scribd.com/document/457716789/Descriptors-and-their-selection-methods-in-QSAR-analysis-paradigm-for-drug-design#:~:text=Topological%20descriptors%20are%20based%20on,%2C%20shape%2C%20and%20atom%20distribution.
27. Tutorial: Molecular Descriptors in QSAR - UC Santa Barbara, https://people.chem.ucsb.edu/kahn/kalju/chem162/public/molecules_qsar.html
28. Prediction of the Toxicity of Binary Mixtures by QSAR Approach Using the Hypothetical Descriptors - PMC - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC6274693/
29. Molecular Field Extrema as Descriptors of Biological Activity: Definition and Validation | Journal of Chemical Information and Modeling - ACS Publications, https://pubs.acs.org/doi/10.1021/ci050357s
30. A Survey of Quantitative Descriptions of Molecular Structure - PMC - NIH, https://pmc.ncbi.nlm.nih.gov/articles/PMC3809149/
31. High-dimensional QSAR modelling using penalized linear regression model with L1/2-norm, https://tandf.figshare.com/articles/journal_contribution/High-dimensional_QSAR_modelling_using_penalized_linear_regression_model_with_i_L_i_sub_1_2_sub_-norm/3830025
32. Rep3Net: An Approach Exploiting Multimodal Representation for Molecular Bioactivity Prediction - arXiv, https://arxiv.org/html/2512.00521v1
33. Trade-Offs in Sparsity vs. Model Accuracy - Newline.co, https://www.newline.co/@zaoyang/trade-offs-in-sparsity-vs-model-accuracy--84959633