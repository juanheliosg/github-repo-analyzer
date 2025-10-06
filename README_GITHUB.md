# 📊 GitHub Repository Analyzer

> A comprehensive tool for analyzing GitHub repositories and generating detailed progress reports for educational settings.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Issues](https://img.shields.io/github/issues/YOUR_USERNAME/github-repo-analyzer.svg)](https://github.com/YOUR_USERNAME/github-repo-analyzer/issues)

## 🎯 Overview

GitHub Repository Analyzer is designed for educators and project managers who need to track student or team progress across multiple GitHub repositories. It provides comprehensive analysis including commit activity, content quality assessment, and generates PowerPoint-ready visualizations.

### ✨ Key Features

- 🔍 **Automated Repository Discovery**: Scans GitHub directories to find student repositories
- 📈 **Commit Analysis**: Tracks activity patterns, commit sizes, and frequency
- 📝 **Content Quality Assessment**: Analyzes markdown and PDF files for quality metrics
- 🌍 **Spanish Language Support**: Optimized for Spanish educational environments
- 📊 **Visual Reporting**: Generates PowerPoint-ready charts and graphs
- 🔧 **Comprehensive Error Handling**: Detailed error reporting and troubleshooting

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

## 📊 Analysis Features

### Repository Metrics
- Repository accessibility verification
- Commit count and activity patterns
- Commit size analysis (additions/deletions)
- Date-range filtered analysis

### Content Quality Analysis
- **Text Extraction**: From both Markdown and PDF files
- **Readability Assessment**: Flesch Reading Ease scoring
- **Sentiment Analysis**: Positive/negative content detection
- **Vocabulary Diversity**: Unique word ratios and complexity
- **Composite Quality Score**: 0-100 rating system

### Visual Reports
- Repository overview dashboards
- Commit activity analysis
- Content quality metrics
- Error analysis and troubleshooting
- Individual student progress tracking

## 🎨 Sample Visualizations

The tool generates six comprehensive visualizations:

1. **📋 Overview Report** - General statistics and status overview
2. **🚨 Error Analysis** - Repository access issues and solutions  
3. **📈 Repository Metrics** - Individual repository performance
4. **❌ Error Repositories** - Detailed error tracking
5. **📊 Summary Statistics** - Key metrics and KPIs
6. **📝 Repository List** - Complete student/repository listing

## 🛠️ Technical Details

### Supported File Types
- **Markdown (.md)**: Full text analysis, formatting cleanup
- **PDF (.pdf)**: Text extraction, content analysis
- **Repository metadata**: Commit history, file structure

### Analysis Capabilities
- **Multi-language Support**: Spanish and English content
- **Quality Scoring**: Combines multiple factors for comprehensive assessment
- **Error Recovery**: Graceful handling of inaccessible repositories
- **Performance Optimization**: Efficient processing of large datasets

### Output Formats
- **CSV**: Detailed metrics for further analysis
- **PNG**: High-resolution visualizations (300 DPI)
- **Structured Data**: Easy integration with other tools

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
│   ├── VISUAL_REPORT_GUIDE.md      # Report interpretation
│   └── CONTENT_ANALYSIS_DOCS.md    # Technical documentation
├── ⚙️ Configuration
│   ├── requirements.txt            # Python dependencies
│   ├── .gitignore                  # Git ignore rules
│   └── .env.example                # Environment template
└── 📁 Output Directories
    ├── results/                    # CSV analysis results
    └── visual_reports/             # Generated visualizations
```

## 🎓 Educational Use Cases

### For Instructors
- **Progress Monitoring**: Track student activity over time
- **Quality Assessment**: Evaluate documentation and code quality
- **Engagement Analysis**: Identify students who may need support
- **Comparative Analysis**: Compare performance across groups

### For Project Managers
- **Team Performance**: Monitor team productivity and collaboration
- **Code Quality**: Assess documentation and development practices
- **Resource Allocation**: Identify areas needing additional support
- **Progress Reporting**: Generate stakeholder-ready reports

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

## 📈 Metrics and Interpretation

### Quality Score Interpretation
- **90-100**: Excellent - Comprehensive, well-documented work
- **70-89**: Good - Solid work with room for minor improvements
- **50-69**: Average - Meets basic requirements
- **30-49**: Below Average - Needs significant improvement
- **0-29**: Poor - Requires immediate attention

### Commit Activity Levels
- **High (30+ commits)**: Very active development
- **Medium (15-30 commits)**: Regular, steady progress
- **Low (5-15 commits)**: Minimal but acceptable activity
- **Very Low (0-5 commits)**: May indicate issues

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/YOUR_USERNAME/github-repo-analyzer.git
cd github-repo-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

### Running Tests
```bash
python test_setup.py
pytest tests/  # If test suite is available
```

## 🐛 Troubleshooting

Common issues and solutions:

- **Token Issues**: Verify GitHub token permissions
- **Installation Problems**: Check Python version (3.8+ required)
- **Memory Issues**: Reduce file size limits for large repositories
- **Network Errors**: Check internet connection and GitHub API status

See [INSTALL.md](INSTALL.md) for detailed troubleshooting.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyGithub** - GitHub API integration
- **NLTK** - Natural language processing
- **Matplotlib/Seaborn** - Data visualization
- **PyPDF2** - PDF text extraction
- **Pandas** - Data manipulation

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/YOUR_USERNAME/github-repo-analyzer/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/YOUR_USERNAME/github-repo-analyzer/discussions)
- 📧 **Email**: your-email@domain.com
- 📖 **Documentation**: [Wiki](https://github.com/YOUR_USERNAME/github-repo-analyzer/wiki)

---

<div align="center">

**⭐ Star this repository if it helps you!** 

Made with ❤️ for educators and developers

</div>