# GitHub Repository Analyzer - Installation Guide

## 📋 Overview

GitHub Repository Analyzer is a comprehensive tool for analyzing multiple GitHub repositories, designed specifically for educational settings. It provides detailed metrics on commit activity, content quality, and student progress.

## 🚀 Quick Installation

### Prerequisites

- Python 3.8 or higher
- Git
- GitHub Personal Access Token

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/github-repo-analyzer.git
cd github-repo-analyzer
```

### 2. Set up Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Test Installation

```bash
python test_setup.py
```

You should see:
```
✅ Setup complete! All packages are available.
```

## 🔧 Configuration

### GitHub Token Setup

1. **Generate Token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Create new token with `repo` and `read:user` permissions
   - Copy the token

2. **Set Environment Variable:**
   ```bash
   export GITHUB_TOKEN="your_github_token_here"
   ```

   Or create a `.env` file:
   ```bash
   echo "GITHUB_TOKEN=your_github_token_here" > .env
   ```

## 📊 Usage

### Basic Analysis

```bash
# Analyze a group's repositories
python github_repo_analyzer.py GROUP_NAME $GITHUB_TOKEN 2024-01-01 2024-12-31 results/output.csv

# Example:
python github_repo_analyzer.py "B3" $GITHUB_TOKEN "2024-01-01" "2024-12-31" "results/analysis_B3.csv"
```

### Using Convenience Scripts

```bash
# Run analysis (edit run_analysis.sh first)
./run_analysis.sh

# Generate visual reports
./generate_report.sh B3
```