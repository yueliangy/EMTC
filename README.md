# EMTC: Evolving-masked Multivariate Time-Series Clustering
Official implementation of **EMTC (Evolving-masked MTS Clustering)** from the paper:

> **"Mask the Redundancy: Evolving Masking Representation Learning for Multivariate Time-Series Clustering"**  
> *Zexi Tan, Xiaopeng Luo, Yunlin Liu, Yiqun Zhang*  
> School of Computer Science and Technology, Guangdong University of Technology, Guangzhou, China  
> AAAI Conference on Artificial Intelligence (AAAI), 2026

## 📖 Abstract

Multivariate Time-Series (MTS) clustering discovers intrinsic grouping patterns of temporal data samples. However, MTS often contains substantial redundancy that diminishes attention to discriminative timestamps, leading to performance bottlenecks. EMTC addresses this through:

- **Importance-aware Variate-wise Masking (IVM)**: Dynamically adapts masking to exclude redundant timestamps
- **Multi-Endogenous Views (MEV)**: Provides comprehensive data perspectives to prevent premature convergence
- **Dual-path learning**: Combines reconstruction and contrastive learning for robust representation learning

EMTC achieves state-of-the-art performance on 15 benchmark datasets with an average improvement of 4.85% in F1-Score over strongest baselines.

## 🏗️ Architecture

The EMTC framework consists of:
- **IVM Module**: Content-aware attention mechanism for dynamic timestamp masking
- **MEV Generation**: Multi-view representation learning
- **CRL Pathway**: Consistency and reconstruction learning
- **CMC Pathway**: Clustering-guided contrastive learning