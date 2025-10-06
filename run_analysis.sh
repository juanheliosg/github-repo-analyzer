#!/bin/bash

# Example usage script for GitHub Repository Analyzer
# Make sure to replace the variables below with your actual values

# Configuration
GROUP_NAME="B3"  # Replace with actual group name
#GITHUB_TOKEN=""  # Replace with your GitHub token
START_DATE="2025-09-22"  # Adjust date range as needed
END_DATE="2025-10-06"
OUTPUT_CSV="./results/analysis_test_results_B3.csv"

# Check if Python virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activating Python virtual environment..."
    source .venv/bin/activate
fi

# Run the analysis
echo "Starting GitHub repository analysis..."
echo "Group: $GROUP_NAME"
echo "Date range: $START_DATE to $END_DATE"
echo "Output file: $OUTPUT_CSV"

python github_repo_analyzer.py "$GROUP_NAME" "$GITHUB_TOKEN" "$START_DATE" "$END_DATE" "$OUTPUT_CSV"

echo "Analysis complete! Check $OUTPUT_CSV for results."