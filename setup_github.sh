#!/bin/bash

# GitHub Repository Setup Script
# This script prepares the repository for upload to GitHub

echo "🚀 GitHub Repository Analyzer - Setup for GitHub Upload"
echo "========================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "📁 Git repository already exists"
fi

# Add all files
echo "📝 Adding files to Git..."
git add .

# Check git status
echo "📊 Git status:"
git status --short

# Create initial commit if needed
if [ -z "$(git log --oneline 2>/dev/null)" ]; then
    echo "💾 Creating initial commit..."
    git commit -m "Initial commit: GitHub Repository Analyzer

Features:
- Repository analysis and metrics
- Content quality assessment  
- PDF and Markdown text analysis
- Spanish language support
- Visual report generation
- PowerPoint-ready visualizations
- Comprehensive error handling

Ready for educational use!"
    echo "✅ Initial commit created"
else
    echo "💾 Repository already has commits"
fi

# Instructions for GitHub upload
echo ""
echo "🎯 Next Steps for GitHub Upload:"
echo "================================="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   📍 Go to: https://github.com/new"
echo "   📝 Repository name: github-repo-analyzer"
echo "   📄 Description: Comprehensive tool for analyzing GitHub repositories in educational settings"
echo "   ✅ Make it public (recommended) or private"
echo "   ❌ Don't initialize with README, .gitignore, or license (we have them)"
echo ""
echo "2. Add the GitHub remote:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/github-repo-analyzer.git"
echo ""
echo "3. Push to GitHub:"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Update README.md:"
echo "   📝 Replace 'YOUR_USERNAME' with your actual GitHub username"
echo "   📝 Update email addresses and contact information"
echo ""
echo "5. Optional - Enable GitHub Pages:"
echo "   ⚙️  Go to repository Settings > Pages"
echo "   📖 Set source to 'Deploy from a branch'"
echo "   🌿 Select 'main' branch and '/ (root)' folder"
echo ""

# Check if remote exists
if git remote -v | grep -q origin; then
    echo "📡 Remote 'origin' already configured:"
    git remote -v
    echo ""
    echo "🚀 Ready to push:"
    echo "   git push -u origin main"
else
    echo "📡 No remote configured yet"
    echo "   Add remote: git remote add origin https://github.com/YOUR_USERNAME/github-repo-analyzer.git"
fi

echo ""
echo "📋 Repository Contents:"
echo "======================="
echo "📄 Core files:"
ls -la *.py *.md *.txt *.sh LICENSE 2>/dev/null | head -10

echo ""
echo "📁 Directories:"
ls -la | grep "^d" | grep -v "\.venv\|__pycache__\|\.git"

echo ""
echo "🔒 Files excluded by .gitignore:"
echo "   - .env files (GitHub tokens)"
echo "   - .venv/ (virtual environment)"
echo "   - results/ (analysis output)"
echo "   - visual_reports/ (generated charts)"
echo "   - __pycache__/ (Python cache)"

echo ""
echo "✅ Repository is ready for GitHub upload!"
echo "📚 Don't forget to update the README with your GitHub username!"