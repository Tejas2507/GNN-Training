# Fraud Graph Preprocessing Pipeline

This document explains the universal preprocessing pipeline designed for the Fraud Graph Foundation Model. The goal is to build a unified representational space that bridges multiple heterogeneous fraud datasets (credit card/wire transfers, telecommunication call records, and blockchain transaction flows) into a single universal representation before downstream pretraining.

This preprocessing pipeline is **architecture-independent**. While it is set up to prepare datasets for the **GIT (Graph-Interactive Transformer)** encoder pretraining, the resulting graphs, natural-language features, and structured tables can be integrated directly with any Graph Foundation Model or Graph Neural Network (GNN).

---

## Pipeline Architecture

```
Raw Dataset (CSV / Space-Separated Text / Edgelist)
          ↓
Universal FraudGraph (Standardized Python Object Schema)
          ↓
Validation (Integrity checks, Node/Edge Type counts, Degree checks)
          ↓
Natural Language Description Generation (Rich templates mapping data to human-readable paragraphs)
          ↓
Text Normalization (Vector statistical summarizing & token optimization for ST Models)
          ↓
SentenceTransformer Embedding (Next stage - Kaggle)
          ↓
PyTorch Geometric Dataset (Next stage - Kaggle)
          ↓
GIT Pretraining (Next stage - Kaggle)
```

---

## Supported Datasets & Formats

### 1. IBM AML (Anti-Money Laundering)
* **Node Meaning**: A `bank_account` represented uniquely as `"{BankID}_{AccountID}"` to prevent conflicts across different banks.
* **Edge Meaning**: A `bank_transfer` directed from Sender Bank Account $\rightarrow$ Receiver Bank Account.
* **Node Features**: Vectorized aggregates of transaction histories:
  * Incoming/Outgoing transaction counts
  * Total & average transaction amounts (in/out)
  * Account lifetime (seconds/days)
  * Node degrees (in, out, total)
  * Fraud ratio (laundering transactions / total transactions)
* **Labels**: A node is labeled `1` (money laundering) if it participates in at least one laundering transaction (either as sender or receiver). Otherwise, it is labeled `0`.
* **Graph Statistics (Full HI-Small_Trans.csv)**:
  * **Nodes**: 515,088
  * **Edges**: 5,078,345
  * **Average Degree**: 19.7184
  * **Fraud Nodes**: 6,357 (1.2342%)
  * **Benign Nodes**: 508,731

### 2. BUPT (Telecommunications Call Record Graph)
* **Node Meaning**: A `phone_number` node participating in communication events.
* **Edge Meaning**: A `call_or_sms` relationship between two phone numbers.
* **Feature Vector**: 39 anonymized numeric features representing user behaviors and subscription attributes, plus computed graph degrees (in, out, total).
* **Labels**: Preserved multiclass classification labels representing distinct user categories:
  * `0`: Normal User
  * `1`: Fraud/Spammer
  * `2`: Call Center / Marketing
  * `3`: Financial Fraud / Smishing
* **Graph Statistics**:
  * **Nodes**: 125,713
  * **Edges**: 226,108
  * **Average Degree**: 3.5972
  * **Class Distribution**: `{0: 99861, 1: 8448, 2: 8074, 3: 9330}`

### 3. Elliptic++ (Bitcoin Transaction Flow Graph)
* **Node Meaning**: A Bitcoin `transaction` entity.
* **Edge Meaning**: A transaction `flow` showing coins moving from one transaction input to another.
* **Feature Vector**: 165 numeric features (timestep + 165 original features) representing local transaction details (degree, fee, inputs/outputs) and aggregated features from 1-hop and 2-hop neighborhoods.
* **Labels**: Mapped binary classification labels:
  * `0`: Licit (legitimate Bitcoin transactions)
  * `1`: Illicit (confirmed darknet, scam, or ransomware-related transactions)
  * `None`: Unlabeled (unknown transaction status, representing ~77% of nodes)
* **Graph Statistics**:
  * **Nodes**: 203,769
  * **Edges**: 234,355
  * **Average Degree**: 2.3002
  * **Class Distribution**: `{0: 42019, 1: 4545}` (with 157,205 unlabeled nodes)

---

## Description Generation & Normalization

To feed these heterogeneous datasets into deep language models (e.g., SentenceTransformers) for generating node embeddings, we translate all structural and tabular features into natural language descriptions.

To prevent truncation by encoder models (which typically enforce a 512-token limit), we run a **Normalization Stage** which optimizes semantic richness while keeping word counts below 512 words.

### Normalization Logic:
1. **IBM AML**: Synthesizes transaction statistics, lifetimes, and degrees into a clean, concise paragraph.
2. **BUPT**: Instead of listing all 39 unnamed features individually, it summarizes them using statistical metrics (dimension, mean, min, max, standard deviation) alongside degrees and labels.
3. **Elliptic++**: Condenses the 165 features into statistics (mean, min, max, standard deviation, L2 norm) and lists only the first 5 features explicitly. This drops descriptions from ~1600 characters down to ~42 words.
