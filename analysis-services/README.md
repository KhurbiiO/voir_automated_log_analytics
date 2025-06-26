# 🧰 Analysis Services - Voir Automated Log Analytics

The **Analysis Services** module is a core component of the *Voir Automated Log Analytics* pipeline. This service is responsible for the log analytics, metric analytics and data retrieval.

## 🔧 Core Components

### 1. Log analytics
The set of log analytics model loaders with custom processing log streams 

#### 🔍 Key Functionality:
- Analyse sequence of log messages.

#### 📁 Global Configuration (YAML):
Each system instance is initialized via YAML, but **LogBERT** and **DeepLog** each have different configuration sets.

##### LogBERT
**Logbert** is a log parsing tool that uses a tree-based algorithm to extract structured information from unstructured log data.

| Section         | Variable   | Description                                                   |
|----------------|------------|---------------------------------------------------------------|
| *Instance key* | `type`     | Type of model to use. (LB)                                    |
| *Instance key* | `dir`      | Path to directory containing vocabulary.                      |
| *Instance key* | `model_dir`| Path to directory containing model state.                     |
| *Instance key* | `seq_len`  | Default sequence length for analysis.                         |
| *Instance key* | `n_pred`   | Number of predictions per sequence inference.                 |

##### ⚠️ Note:
A vocabulary generated during the training session should be in this directory.

##### DeepLog
**DeepLog** is a deep learning-based log anomaly detection system. It models normal log sequences using LSTM networks and identifies deviations as anomalies, enabling real-time detection of abnormal system behavior.

| Section         | Variable   | Description                                                   |
|----------------|------------|---------------------------------------------------------------|
| *Instance key* | `type`     | Type of model to use. (DL)                                    |
| *Instance key* | `model`    | Path to model state                |
| *Instance key* | `seq_len`  | Default sequence length for analysis.                         |
| *Instance key* | `n_pred`   | Number of predictions per sequence inference.                 |


---
### 2. Metric Analytics

#### 🔍 Key Functionality:
- Analyze streams of metric values

#### 📁 Global Configuration (YAML):
Each system instance is initialized via YAML, where the instance key includes:

- `state`: Path to Drain's serialized state.
- `config` : Model type to load initialize. 
