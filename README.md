# 📊 GitHub Repository Analyzer

> A comprehensive tool for analyzing GitHub repositories and generating detailed progress reports for educational settings.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/YOUR_USERNAME/github-repo-analyzer.svg)](https://github.com/YOUR_USERNAME/github-repo-analyzer/issues)

## 🎯 Overview

GitHub Repository Analyzer is designed for educators and project managers who need to track student or team progress across multiple GitHub repositories. It provides comprehensive analysis including commit activity, content quality assessment, and generates PowerPoint-ready visualizations.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/github-repo-analyzer.git
cd github-repo-analyzer

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test installation
python test_setup.py
```

### Basic Usage

```bash
# Analyze repositories
python github_repo_analyzer.py "GROUP_NAME" "YOUR_GITHUB_TOKEN" "2024-01-01" "2024-12-31" "results.csv"

# Generate visual reports
python generate_visual_report.py results/ --output visual_reports/
```

📖 **[Complete Installation Guide](INSTALL.md)** | 🎮 **[Usage Examples](USAGE_EXAMPLES.md)**

## 📁 Project Structure

```
github-repo-analyzer/
├── 🔧 Core Analysis
│   ├── github_repo_analyzer.py     # Main analysis engine
│   ├── generate_visual_report.py   # Visualization generator
│   └── test_setup.py               # Installation verification
├── 🚀 Convenience Scripts
│   ├── run_analysis.sh             # Analysis automation
│   └── generate_report.sh          # Report generation
├── 📖 Documentation
│   ├── README.md                   # Project overview
│   ├── INSTALL.md                  # Installation guide
│   ├── USAGE_EXAMPLES.md           # Usage examples
├── ⚙️ Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore                  # Git ignore rules
│   └── .env.example                # Environment template
└── 📁 Output Directories
    ├── results/                    # CSV analysis results
    └── visual_reports/             # Generated visualizations
```

## 🌟 Example Workflows

### Weekly Progress Review
```bash
# Analyze current week's activity
python github_repo_analyzer.py "ClassGroup" "$TOKEN" "2024-10-01" "2024-10-07" "weekly.csv"

# Generate visual report
./generate_report.sh ClassGroup

# Review visualizations in visual_reports/
```

### Semester-End Assessment
```bash
# Full semester analysis
python github_repo_analyzer.py "AllStudents" "$TOKEN" "2024-09-01" "2024-12-31" "semester.csv"

# Generate comprehensive report
python generate_visual_report.py results/ --output final_reports/
```

### Multi-Group Comparison
```bash
# Analyze multiple groups
for group in A1 B2 C3; do
    ./run_analysis.sh "$group"
    ./generate_report.sh "$group"
done
```

