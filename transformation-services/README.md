# 🧠 Transformation Services - Voir Automated Log Analytics

The **Transformation Services** module is a core component of the *Voir Automated Log Analytics* pipeline. This service is responsible for converting raw system logs into structured, analyzable formats by leveraging a combination of advanced parsing, PII detection, and intelligent filtering.

## 🔧 Core Components

### 1. **Data Parser (Drain Template Miner)**
The data parser uses the [Drain](https://github.com/logpai/Drain3) algorithm to cluster logs into templates and assign event IDs based on structural similarity. Drain works iteratively by:

- Starting with the entire log message as a template.
- Updating the template when new but similar logs are seen.
- Producing structured templates with variable segments that reflect real-time log changes.

#### 🔍 Key Functionality:
- Translates a log messages into a structured event ID.
- Returns the corresponding template for a given message.

#### 🛠 Drain Configuration Options:

| Section     | Variable               | Description                                                                 |
|-------------|------------------------|-----------------------------------------------------------------------------|
| SNAPSHOT    | `snapshot_interval_minutes` | Interval (minutes) between parser state snapshots.                         |
| SNAPSHOT    | `compress_state`       | Enables compression to reduce snapshot size.                               |
| MASKING     | `masking`              | Regex-based rules for masking sensitive log content (e.g., IPs).           |
| MASKING     | `mask_prefix`/`mask_suffix` | Prefix/suffix around masked values.                                   |
| DRAIN       | `engine`               | Specifies the parser engine (Drain).                                       |
| DRAIN       | `sim_th`               | Similarity threshold for log clustering (0.0–1.0).                         |
| DRAIN       | `depth`                | Max depth of the Drain parsing tree.                                       |
| DRAIN       | `max_children`         | Max children per tree node.                                                |
| DRAIN       | `max_clusters`         | Max allowable log clusters (limits memory/CPU).                            |
| DRAIN       | `extra_delimiters`     | Extra delimiters used during log tokenization.                             |
| PROFILING   | `enabled`              | Enables performance profiling.                                             |
| PROFILING   | `report_sec`           | Frequency of profiling reports in seconds.                                 |

#### 📁 Global Configuration (YAML):
Each system instance is initialized via YAML, where the instance key includes:

- `state`: Path to Drain's serialized state.
- `config`: Path to Drain's configuration.
- `support`: Path to Smart Filter whitelist CSV.

---

### 2. **PII Detection (Presidio)**
To maintain compliance with **GDPR** and the **EU AI Act**, the Presidio Analyzer is integrated for detecting and optionally redacting Personally Identifiable Information (PII).

#### 🧩 Supported Features:
- Named entity recognition (e.g., Names, IPs, Phone Numbers).
- Custom model/language integrations (e.g., SpaCy).
- Configurable score thresholds.
- Context-aware precision improvements.
- Metadata output (confidence, recognizer name, etc.).

> ⚙️ For configuration options, refer to the [Presidio Analyzer documentation](https://microsoft.github.io/presidio/analyzer/)

#### 📁 Global Configuration (YAML):
Each system instance is initialized via YAML, where the instance key includes:

- `state`: Path to Drain's serialized state.
- `config`: Path to Drain's configuration.
- `support`: Path to Smart Filter whitelist CSV.
---

### 3. **Smart Filter (Heuristic-Based Anomaly Detection)**
This is a custom-built filter system designed to replicate the heuristics of experienced support engineers.

#### ⚙️ How It Works:
1. Parses incoming messages into structured events.
2. Compares the event against a known whitelist of important/anomalous templates.
3. Uses a transformer model (`MiniLM-L6-v2`) to compute cosine similarity between variable parts of logs.
4. Makes a decision whether to flag or pass the log.

#### ⚠️ Note:
As a heuristic algorithm, it does not guarantee perfect accuracy but provides high practical effectiveness in real-world log triage scenarios.

---

## 📦 Setup & Usage

### 🧰 Prerequisites:
- Python 3.8+
- `pip install -r requirements.txt`

### 🚀 Starting the Service:
```bash
python main.py --config path/to/global_config.yaml