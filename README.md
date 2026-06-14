<!-- @format -->

# Kayfa Platform - Student Analytics Dashboard

<!-- One-line tagline: what it does and who it's for -->

> A comprehensive analytics Dashboard for tracking and visualizing student academic performance, engagement, and learning outcomes.

<!-- Shields / badges -->

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-1.0.0-green.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Data management](#data-management)
- [License](#license)

---

## Overview

The Kayfa Platform is a student analytics system built for educators and administrators to gain actionable insights into student performance and engagement. It aggregates data from multiple sources—including grades, assignment submissions, course performance, and engagement events—into an interactive Streamlit dashboard.

The system addresses the need for comprehensive academic analytics by providing data-driven insights through intuitive visualizations, clustering analysis, and performance trends, enabling institutions to make informed decisions about student support and intervention.

---

## Features

- ✅ **Interactive Analytics Dashboard** — Real-time visualization of student performance metrics and trends using Plotly
- ✅ **Multi-Source Data Aggregation** — Integrates student data, grades, courses, assignments, engagement events, and more
- ✅ **Cluster Analysis** — K-means clustering for identifying student performance groups using scikit-learn
- ✅ **MongoDB Integration** — Seamless data persistence and retrieval from MongoDB
- ✅ **Customizable Visualizations** — Dynamic charts, graphs, and performance heatmaps
- ✅ **Data Preprocessing Pipeline** — Automated cleaning and standardization of student data

---

## Tech stack

| Layer           | Technology         | Why                                                             |
| --------------- | ------------------ | --------------------------------------------------------------- |
| Frontend        | Streamlit          | Rapid dashboard development with Python; minimal UI/UX overhead |
| Visualization   | Plotly             | Interactive, publication-quality charts and subplots            |
| Data Processing | Pandas, NumPy      | Efficient tabular data manipulation and numerical operations    |
| Analytics       | Scikit-learn       | K-means clustering and data standardization                     |
| Database        | MongoDB            | Flexible document storage for heterogeneous student data        |
| Statistics      | SciPy, StatsModels | Statistical analysis and hypothesis testing                     |

---

## Getting started

### Prerequisites

Make sure you have the following installed before proceeding:

- **Python** ≥ 3.9 — [Download Python](https://www.python.org/downloads/)
- **pip** — Included with Python 3.9+
- **MongoDB URI** — Connection string for MongoDB instance (local or cloud-based)

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Platform_students_Analytics

# 2. Create a virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# 1. Create MongoDB secrets file
mkdir -p .streamlit
echo 'MONGO_URI = "mongodb+srv://username:password@cluster.mongodb.net/"' > .streamlit/secrets.toml

# 2. Alternatively, set as environment variable
export MONGO_URI="mongodb+srv://username:password@cluster.mongodb.net/"
```

---

## Usage

[Click Here to open Dashboard](https://platformstudentsanalytics.streamlit.app/)

This script loads all CSV files from the `data/` directory and uploads them to MongoDB collections.

---

## Project structure

```
Platform_students_Analytics/
├── app.py                              # Main Streamlit dashboard application
├── mongo_loader.py                     # MongoDB connection and data loading utilities
├── upload_to_mongodb.py                # Script to upload CSV data to MongoDB
├── requirements.txt                    # Python dependencies
├── data/                               # CSV data files
│   ├── students_clean.csv              # Student information
│   ├── courses_clean.csv               # Course data
│   ├── grades_clean.csv                # Student grades
│   ├── assignment_submissions_clean.csv  # Assignment submission records
│   ├── concepts_performance_clean.csv  # Concept performance metrics
│   ├── engagement_events_clean.csv     # Student engagement events
│   └── groups_cleaned.csv              # Student groups/cohorts
├── .streamlit/
│   └── secrets.toml                    # MongoDB credentials (git-ignored)
├── LICENSE
└── README.md                           # This file
```

---

## Data management

### Data Sources

The platform ingests data from the following CSV files in the `data/` directory:

- **students_clean.csv** — Student demographic and enrollment information
- **courses_clean.csv** — Course metadata and structure
- **grades_clean.csv** — Academic grades and scores
- **assignment_submissions_clean.csv** — Assignment completion and submission data
- **concepts_performance_clean.csv** — Concept-level performance metrics
- **engagement_events_clean.csv** — Learning platform engagement events
- **groups_cleaned.csv** — Student cohort and group assignments

### Data Pipeline

1. **Data Ingestion** — CSV files are read and validated
2. **Data Cleaning** — Preprocessing and standardization applied
3. **MongoDB Upload** — Cleaned data persisted to MongoDB collections
4. **Dashboard Retrieval** — Streamlit app queries and displays aggregated insights

---

## License

This project is distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.
