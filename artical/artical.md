Paper Title: HAFE-ABSA: Fair Heterogeneous Graph Neural Networks for
  Aspect-Based Sentiment Analysis via Information Theory

  1. Abstract

  1.1 Purpose: This paper addresses the fairness problem in aspect-based
  sentiment analysis where low-frequency aspects receive worse predictions
  than high-frequency ones.
  1.2 Key Methods: We propose HAFE-ABSA, which combines information
  theory-based heterogeneous edge detection with type-aware graph
  convolutional networks and an unsupervised fairness enhancement module.
  1.3 Main Results: Experiments on SemEval-2014 and SemEval-2016 datasets
  demonstrate improved fairness metrics while maintaining competitive
  accuracy compared to baseline methods.
  1.4 Significance: This work provides a novel approach to mitigate
  frequency-based bias in ABSA systems, ensuring more equitable performance
  across both common and rare aspect terms.

  2. Introduction

  2.1 Background & Motivation: Aspect-based sentiment analysis systems
  exhibit systematic bias against low-frequency aspect terms, leading to
  unfair predictions in practical applications such as restaurant review
  analysis.
  2.2 Research Gap & Objectives: Existing ABSA methods focus primarily on
  accuracy improvements while neglecting fairness considerations across
  aspects with different frequency distributions.
  2.3 Key Contributions: We introduce a fairness-aware framework that
  employs information theory to detect heterogeneous graph structures,
  type-aware message passing to leverage edge semantics, and unsupervised
  feature enhancement to reduce frequency-based bias.
  2.4 Paper Structure: The remainder of this paper presents related work,
  formalizes the fairness problem in ABSA, describes our methodology,
  reports experimental results, and concludes with limitations and future
  directions.

  3. Related Work

  3.1 Aspect-Based Sentiment Analysis: Traditional and neural approaches to
  ABSA have evolved from attention mechanisms to graph neural networks that
  leverage syntactic dependency structures.
  3.2 Fairness in Machine Learning: Recent work on algorithmic fairness has
  addressed demographic parity and equalized odds in classification tasks,
  though application to NLP tasks like ABSA remains limited.
  3.3 Graph Neural Networks for NLP: Graph-based models have shown success
  in capturing structural relationships in text, with heterogeneous and
  type-aware variants improving performance on relation extraction and
  semantic role labeling tasks.
  3.4 Summary & Limitations: While existing methods excel at overall
  performance, they lack mechanisms to ensure equitable predictions across
  aspects with varying frequencies, motivating our fairness-aware approach.

  4. Problem Definition

  4.1 Formal Statement: Given a sentence and aspect terms with varying
  frequencies in the training distribution, the goal is to predict sentiment
   polarities while minimizing performance disparity between high-frequency
  and low-frequency aspects.
  4.2 Assumptions & Scope: We assume access to dependency parse trees and
  pre-trained language models, and focus on restaurant domain reviews where
  aspect frequency imbalance is pronounced.
  4.3 Evaluation Metrics: We measure both standard metrics (accuracy,
  macro-F1, micro-F1) and fairness metrics (Gini coefficient, min-max gap,
  demographic parity between frequency groups, per-aspect F1 variance).

  5. Methodology

  5.1 Graph Construction with Typed Edges: We construct heterogeneous graphs
   from dependency parses where edges are classified into four types:
  aspect-opinion edges, core syntactic relations, aspect co-reference edges,
   and auxiliary edges.

  Figure 1: Architecture overview showing the pipeline from input sentence
  through typed graph construction, Type-Aware GCN layers, F3 fairness
  enhancement, to sentiment classification.

  5.2 Type-Aware Graph Convolutional Network: Our Type-Aware GCN layer
  learns separate weight matrices for each edge type with learnable
  importance weights, enabling differentiated message passing based on edge
  semantics.

  Figure 2: Illustration of Type-Aware GCN message passing mechanism showing
   how different edge types use distinct transformation matrices and
  importance weights.

  5.3 Information Theory-Based Fairness Enhancement: The F3 module employs
  mutual information computation to detect heterogeneous neighbors and
  trains an unsupervised estimator to adjust node features, reducing bias
  toward high-frequency aspects.

  Table 1: Edge type definitions and their initialized importance weights in
   the Type-Aware GCN architecture.

  5.4 Training Procedure: The model undergoes two-phase training with
  offline F3 preprocessing on merged training graphs followed by end-to-end
  supervised training with cross-entropy loss.

  Figure 3: Detailed workflow of the F3 preprocessing phase including
  heterogeneous neighbor detection and fairness estimator training.

  6. Experiments

  6.1 Experimental Setup: We train on SemEval-2014 and SemEval-2016
  restaurant review datasets using BERT-base embeddings with Stanza
  dependency parsing, comparing our full model against baseline ASGCN and
  ablated variants.

  Table 2: Hyperparameter settings including hidden dimensions, learning
  rates, dropout rates, and training epochs used across all experiments.

  6.2 Dataset & Baselines: SemEval-2014 contains 3,041 training and 800 test
   sentences while SemEval-2016 contains 2,000 training and 676 test
  sentences, with baselines including standard GCN, ASGCN, and BERT-based
  classifiers.

  Table 3: Statistics of datasets including sentence counts, aspect term
  distributions, and frequency imbalance ratios across training and test
  sets.

  6.3 Results & Analysis: HAFE-ABSA achieves comparable or better accuracy
  than baselines while significantly reducing fairness metrics, with
  improvements particularly pronounced for low-frequency aspects.

  Figure 4: Comparison of per-aspect F1 scores sorted by aspect frequency,
  showing reduced performance gap between high-frequency and low-frequency
  aspects for HAFE-ABSA versus baselines.

  Table 4: Main results showing accuracy, macro-F1, micro-F1, and fairness
  metrics (Gini, Gap, DP-Aspect, variance) for all methods on both datasets.

  6.4 Ablation Study: We analyze contributions of individual components by
  removing Type-Aware GCN, F3 module, and edge type classification,
  demonstrating that all components contribute to both accuracy and fairness
   improvements.

  Table 5: Ablation study results showing performance degradation when
  removing Type-Aware GCN, F3 module, edge typing, or combinations thereof.

  6.5 Qualitative Analysis: Case studies reveal that HAFE-ABSA correctly
  predicts sentiments for rare aspects that baseline models misclassify,
  particularly when aspect-opinion edges are properly weighted.

  Figure 5: Example sentences with visualized attention weights and edge
  type distributions, highlighting how Type-Aware GCN focuses on critical
  aspect-opinion connections.

  7. Conclusion

  7.1 Summary of Findings: We presented HAFE-ABSA, a fairness-aware
  framework for aspect-based sentiment analysis that reduces frequency-based
   bias through information theory-guided heterogeneous graph modeling and
  type-aware message passing.
  7.2 Limitations: The approach requires dependency parsing which may
  introduce errors, relies on pre-defined edge type categories, and has only
   been evaluated on restaurant domain data.
  7.3 Future Work: Extensions include automatic edge type discovery,
  multi-domain evaluation, integration with other fairness definitions
  beyond demographic parity, and investigation of fairness-accuracy
  trade-offs through multi-objective optimization.