# Vital Operation Insights Reporter - VOIR

> A framework for AI-powered log analytics 


---

## 🧾 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Build and Run](#️-build-and-run)
- [Access Points](#-access-points)

---

## ✅ Features

- 🔹 Configuration-based initialisation of each component.
- 🔹 Data Ingestion Pipeline with pre-processing routines through LogStash.
- 🔹 Smart Filter Custom Algorithm Implementation (using Drain)
- 🔹 Support for LogBERT, DeepLog, and PySAD models.

---

## 📂 Project Structure

```
project-root/
├── analysis-services/
├── transformation-services/
├── frontend/
├── logstash/
├── docker-compose.yml
└── README.md
```

---

## 💾 Installation

### Prerequisites

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## 🛠️ Build and Run

```bash
docker-compose up --build
```

> This will build all services and run them in attached mode.

To stop:

```bash
docker-compose down
```

### ⚠️ Note:
 To run the solution properly, you need to make the necessary configurations for each component and add the necessary dependencies such as models.


## 🌐 Access Points

- **Analysis API** → [http://localhost:8000](http://localhost:8000)
- **Transformation API** → [http://localhost:8001](http://localhost:8001)
- **MongoDB UI (mongo-express)** → [http://localhost:8081](http://localhost:8081)
- **Frontend (Streamlit)** → [http://localhost:8501](http://localhost:8501)

---

