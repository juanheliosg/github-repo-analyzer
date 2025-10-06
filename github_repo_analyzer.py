#!/usr/bin/env python3
"""
GitHub Repository Analysis Tool
Analyzes multiple private GitHub repositories and outputs information to CSV.

This tool performs comprehensive analysis of student repositories including:
- Commit analysis (count, size metrics)
- Content quality analysis using NLP techniques
- Repository accessibility verification
"""

import argparse
import base64
import csv
import logging
import os
import re
import statistics
import sys
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import nltk
import pandas as pd
import PyPDF2
import textstat
from dateutil import parser as date_parser
from github import Github, GithubException

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GitHubRepoAnalyzer:
    """Main class for analyzing GitHub repositories."""

    def __init__(self, github_token: str, start_date: str, end_date: str):
        """
        Initialize the analyzer with GitHub credentials and date range.

        Args:
            github_token: GitHub personal access token
            start_date: Start date for commit analysis (YYYY-MM-DD)
            end_date: End date for commit analysis (YYYY-MM-DD)
        """
        self.github = Github(github_token)
        self.start_date = self._parse_date(start_date)
        self.end_date = self._parse_date(end_date)

        # Download required NLTK data (including Spanish support)
        try:
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("vader_lexicon", quiet=True)
            # Download Spanish-specific data
            nltk.download("punkt_tab", quiet=True)
            # Ensure we have Spanish stop words
            try:
                from nltk.corpus import stopwords

                stopwords.words("spanish")  # Test if Spanish is available
            except:
                logger.info(
                    "Spanish stop words not available, continuing without stop word filtering"
                )
        except:
            logger.warning("Could not download NLTK data. Some features may not work.")

    def _parse_date(self, date_string: str) -> datetime:
        """Parse date string to datetime object."""
        try:
            return date_parser.parse(date_string)
        except Exception as e:
            logger.error(f"Error parsing date {date_string}: {e}")
            raise ValueError(f"Invalid date format: {date_string}")

    def extract_repos_from_github_directory(self, group_name: str) -> List[Dict]:
        """
        Extract repository URLs from GitHub directory using GitHub API.

        Args:
            group_name: Name of the group directory to analyze

        Returns:
            List of dictionaries with repo info
        """
        repos_info = []

        try:
            logger.info(f"Fetching directory contents for group: {group_name}")

            # Get the repository containing the bitacoras
            bitacoras_repo = self.github.get_repo(
                "juanheliosg/bitacoras-ISE-25-26-VIERNES"
            )

            # Get the contents of the group directory
            try:
                directory_contents = bitacoras_repo.get_contents(group_name)
            except GithubException as e:
                if e.status == 404:
                    logger.error(f"Directory '{group_name}' not found in repository")
                    repos_info.append(
                        {
                            "md_filename": "ERROR",
                            "repo_url": "",
                            "status": "ERROR",
                            "error_message": f"Directory '{group_name}' not found",
                        }
                    )
                    return repos_info
                else:
                    raise e

            # Filter for .md files
            md_files = [
                content
                for content in directory_contents
                if content.type == "file" and content.name.endswith(".md")
            ]

            logger.info(f"Found {len(md_files)} .md files")

            for md_file in md_files:
                filename = md_file.name

                # Get file content using GitHub API
                repo_info = self._extract_repo_url_from_md_content(md_file, filename)
                repos_info.append(repo_info)

        except GithubException as e:
            logger.error(f"GitHub API error: {e}")
            repos_info.append(
                {
                    "md_filename": "ERROR",
                    "repo_url": "",
                    "status": "ERROR",
                    "error_message": f"GitHub API error: {str(e)}",
                }
            )
        except Exception as e:
            logger.error(f"Error fetching directory: {e}")
            repos_info.append(
                {
                    "md_filename": "ERROR",
                    "repo_url": "",
                    "status": "ERROR",
                    "error_message": f"Could not access directory: {str(e)}",
                }
            )

        return repos_info

    def _extract_repo_url_from_md_content(self, md_file_content, filename: str) -> Dict:
        """
        Extract repository URL from a markdown file content object.

        Args:
            md_file_content: GitHub API content object for the markdown file
            filename: Name of the markdown file

        Returns:
            Dictionary with repo information
        """
        try:
            # Decode the base64 content from GitHub API
            content = base64.b64decode(md_file_content.content).decode(
                "utf-8", errors="ignore"
            )

            # Look for GitHub URLs in the format [username](https://github.com/username/repo)
            github_url_pattern = r"\[([^\]]+)\]\((https://github\.com/[^)]+)\)"
            matches = re.findall(github_url_pattern, content)

            if matches:
                username, repo_url = matches[0]  # Take the first match
                return {
                    "md_filename": filename,
                    "repo_url": repo_url,
                    "username": username,
                    "status": "OK",
                    "error_message": "",
                }
            else:
                # Try alternative patterns
                url_pattern = r"https://github\.com/[^\s\)\]>]+"
                urls = re.findall(url_pattern, content)

                if urls:
                    return {
                        "md_filename": filename,
                        "repo_url": urls[0],
                        "username": "",
                        "status": "OK",
                        "error_message": "",
                    }
                else:
                    return {
                        "md_filename": filename,
                        "repo_url": "",
                        "username": "",
                        "status": "ERROR",
                        "error_message": "wrong URL in repo",
                    }

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return {
                "md_filename": filename,
                "repo_url": "",
                "username": "",
                "status": "ERROR",
                "error_message": f"Error reading file: {str(e)}",
            }

    def analyze_repository(self, repo_url: str) -> Dict:
        """
        Analyze a single GitHub repository.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract owner and repo name from URL
            parsed_url = urlparse(repo_url)
            path_parts = parsed_url.path.strip("/").split("/")

            if len(path_parts) < 2:
                return {"status": "ERROR", "error_message": "Invalid GitHub URL format"}

            owner, repo_name = path_parts[0], path_parts[1]

            # Get repository
            repo = self.github.get_repo(f"{owner}/{repo_name}")

            # Analyze commits
            commit_analysis = self._analyze_commits(repo)

            # Analyze content quality
            content_analysis = self._analyze_content_quality(repo)

            # Combine results
            analysis_result = {
                "status": "OK",
                "error_message": "",
                "repo_name": repo.name,
                "repo_owner": owner,
                **commit_analysis,
                **content_analysis,
            }

            return analysis_result

        except GithubException as e:
            if e.status == 404:
                return {"status": "ERROR", "error_message": "could not access to Repo"}
            else:
                return {
                    "status": "ERROR",
                    "error_message": f"GitHub API error: {str(e)}",
                }
        except Exception as e:
            logger.error(f"Error analyzing repository {repo_url}: {e}")
            return {"status": "ERROR", "error_message": f"Analysis error: {str(e)}"}

    def _analyze_commits(self, repo) -> Dict:
        """Analyze commit metrics for the repository."""
        try:
            commits = repo.get_commits(since=self.start_date, until=self.end_date)
            commit_list = list(commits)

            if not commit_list:
                return {
                    "num_commits": 0,
                    "total_commit_size": 0,
                    "mean_commit_size": 0,
                    "median_commit_size": 0,
                    "commit_analysis_status": "No commits in date range",
                }

            # Calculate commit sizes (additions + deletions)
            commit_sizes = []
            total_size = 0

            for commit in commit_list:
                try:
                    stats = commit.stats
                    size = stats.additions + stats.deletions
                    commit_sizes.append(size)
                    total_size += size
                except:
                    # If we can't get stats, assume size 0
                    commit_sizes.append(0)

            mean_size = statistics.mean(commit_sizes) if commit_sizes else 0
            median_size = statistics.median(commit_sizes) if commit_sizes else 0

            return {
                "num_commits": len(commit_list),
                "total_commit_size": total_size,
                "mean_commit_size": round(mean_size, 2),
                "median_commit_size": round(median_size, 2),
                "commit_analysis_status": "Success",
            }

        except Exception as e:
            logger.error(f"Error analyzing commits: {e}")
            return {
                "num_commits": 0,
                "total_commit_size": 0,
                "mean_commit_size": 0,
                "median_commit_size": 0,
                "commit_analysis_status": f"Error: {str(e)}",
            }

    def _analyze_content_quality(self, repo) -> Dict:
        """Analyze content quality of PDF and MD files in the repository."""
        try:
            content_analysis = {
                "total_files_analyzed": 0,
                "md_files_count": 0,
                "pdf_files_count": 0,
                "avg_readability_score": 0,
                "avg_text_complexity": 0,
                "total_word_count": 0,
                "unique_word_count": 0,
                "sentiment_score": 0,
                "content_quality_score": 0,
                "content_analysis_status": "Success",
            }

            md_files = []
            pdf_files = []
            all_text_content = ""
            word_set = set()
            readability_scores = []
            complexity_scores = []

            # Get all files from repository
            try:
                contents = repo.get_contents("")
                self._collect_files_recursive(repo, contents, md_files, pdf_files)
            except Exception as e:
                logger.warning(f"Could not access repository contents: {e}")
                content_analysis["content_analysis_status"] = (
                    f"Limited access: {str(e)}"
                )
                return content_analysis

            # Analyze markdown files
            for file_path in md_files:
                try:
                    file_content = repo.get_contents(file_path)

                    text = base64.b64decode(file_content.content).decode(
                        "utf-8", errors="ignore"
                    )

                    # Remove markdown formatting for analysis
                    clean_text = self._clean_markdown_text(text)

                    if len(clean_text.strip()) > 10:  # Only analyze substantial content
                        all_text_content += clean_text + " "
                        # Better word tokenization for Spanish
                        words = self._tokenize_spanish_text(clean_text.lower())
                        word_set.update(words)

                        # Calculate readability
                        readability = textstat.flesch_reading_ease(clean_text)
                        readability_scores.append(readability)

                        # Calculate complexity (average sentence length)
                        sentences = clean_text.split(".")
                        if sentences:
                            avg_sentence_length = len(clean_text.split()) / len(
                                sentences
                            )
                            complexity_scores.append(avg_sentence_length)

                except Exception as e:
                    logger.warning(f"Could not analyze file {file_path}: {e}")
                    continue

            # Analyze PDF files (text extraction and analysis)
            for file_path in pdf_files:
                try:
                    file_content = repo.get_contents(file_path)

                    # Skip very large PDF files to avoid memory issues

                    # Decode and extract text from PDF
                    pdf_content = base64.b64decode(file_content.content)
                    pdf_text = self._extract_text_from_pdf(pdf_content)

                    if pdf_text and len(pdf_text.strip()) > 10:
                        # Clean and process PDF text similar to markdown
                        clean_text = self._clean_pdf_text(pdf_text)

                        if len(clean_text.strip()) > 10:
                            all_text_content += clean_text + " "
                            # Tokenize and add words
                            words = self._tokenize_spanish_text(clean_text.lower())
                            word_set.update(words)

                            # Calculate readability for PDF content
                            try:
                                readability = textstat.flesch_reading_ease(clean_text)
                                readability_scores.append(readability)
                            except:
                                pass  # Skip readability if it fails

                            # Calculate complexity (average sentence length)
                            sentences = clean_text.split(".")
                            if sentences and len(sentences) > 1:
                                avg_sentence_length = len(clean_text.split()) / len(
                                    sentences
                                )
                                complexity_scores.append(avg_sentence_length)

                    content_analysis["pdf_files_count"] += 1

                except Exception as e:
                    logger.warning(f"Could not analyze PDF file {file_path}: {e}")
                    content_analysis["pdf_files_count"] += 1
                    continue

            # Calculate final metrics
            content_analysis["total_files_analyzed"] = len(md_files) + len(pdf_files)
            content_analysis["md_files_count"] = len(md_files)
            content_analysis["total_word_count"] = len(all_text_content.split())
            content_analysis["unique_word_count"] = len(word_set)

            if readability_scores:
                content_analysis["avg_readability_score"] = round(
                    statistics.mean(readability_scores), 2
                )

            if complexity_scores:
                content_analysis["avg_text_complexity"] = round(
                    statistics.mean(complexity_scores), 2
                )

            # Calculate sentiment score using Spanish heuristics
            positive_words = [
                # Spanish positive words
                "bueno",
                "bien",
                "excelente",
                "perfecto",
                "correcto",
                "completado",
                "terminado",
                "finalizado",
                "éxito",
                "exitoso",
                "funcionando",
                "funciona",
                "resuelto",
                "solucionado",
                "satisfactorio",
                "logrado",
                "conseguido",
                "realizado",
                # English equivalents (for mixed content)
                "good",
                "great",
                "excellent",
                "complete",
                "finished",
                "success",
                "working",
                "solved",
                "done",
            ]
            negative_words = [
                # Spanish negative words
                "error",
                "fallo",
                "problema",
                "inconveniente",
                "dificultad",
                "incompleto",
                "fallido",
                "roto",
                "no funciona",
                "pendiente",
                "por hacer",
                "todo",
                "hacer",
                "arreglar",
                "corregir",
                "mal",
                "malo",
                "defecto",
                "bug",
                "falla",
                # English equivalents (for mixed content)
                "bug",
                "problem",
                "issue",
                "failed",
                "incomplete",
                "broken",
                "not working",
                "todo",
                "fix",
                "repair",
            ]

            text_lower = all_text_content.lower()
            positive_count = sum(text_lower.count(word) for word in positive_words)
            negative_count = sum(text_lower.count(word) for word in negative_words)

            if positive_count + negative_count > 0:
                sentiment_score = (positive_count - negative_count) / (
                    positive_count + negative_count
                )
                content_analysis["sentiment_score"] = round(sentiment_score, 2)

            # Calculate overall content quality score (0-100)
            quality_factors = []

            # Factor 1: File count (more files = more content)
            if content_analysis["total_files_analyzed"] > 0:
                file_score = min(content_analysis["total_files_analyzed"] * 10, 30)
                quality_factors.append(file_score)

            # Factor 2: Word count (more words = more content)
            if content_analysis["total_word_count"] > 0:
                word_score = min(content_analysis["total_word_count"] / 100, 25)
                quality_factors.append(word_score)

            # Factor 3: Readability (higher readability = better quality)
            if content_analysis["avg_readability_score"] > 0:
                readability_normalized = (
                    max(0, min(content_analysis["avg_readability_score"], 100)) / 4
                )
                quality_factors.append(readability_normalized)

            # Factor 4: Vocabulary diversity
            if (
                content_analysis["total_word_count"] > 0
                and content_analysis["unique_word_count"] > 0
            ):
                diversity = (
                    content_analysis["unique_word_count"]
                    / content_analysis["total_word_count"]
                ) * 20
                quality_factors.append(diversity)

            content_analysis["content_quality_score"] = round(sum(quality_factors), 2)

            return content_analysis

        except Exception as e:
            logger.error(f"Error in content quality analysis: {e}")
            return {
                "total_files_analyzed": 0,
                "md_files_count": 0,
                "pdf_files_count": 0,
                "avg_readability_score": 0,
                "avg_text_complexity": 0,
                "total_word_count": 0,
                "unique_word_count": 0,
                "sentiment_score": 0,
                "content_quality_score": 0,
                "content_analysis_status": f"Error: {str(e)}",
            }

    def _collect_files_recursive(
        self, repo, contents, md_files, pdf_files, max_depth=3, current_depth=0
    ):
        """Recursively collect MD and PDF files from repository."""
        if current_depth >= max_depth:
            return

        for content in contents:
            if content.type == "file":
                if content.path.endswith(".md"):
                    md_files.append(content.path)
                elif content.path.endswith(".pdf"):
                    pdf_files.append(content.path)
            elif content.type == "dir" and current_depth < max_depth - 1:
                try:
                    sub_contents = repo.get_contents(content.path)
                    self._collect_files_recursive(
                        repo,
                        sub_contents,
                        md_files,
                        pdf_files,
                        max_depth,
                        current_depth + 1,
                    )
                except:
                    continue  # Skip directories we can't access

    def _clean_markdown_text(self, text: str) -> str:
        """Clean markdown text for analysis."""
        # Remove markdown formatting
        text = re.sub(r"#{1,6}\s+", "", text)  # Headers
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Bold
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # Italic
        text = re.sub(r"`([^`]+)`", r"\1", text)  # Code
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # Links
        text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)  # Images
        text = re.sub(r"```[\s\S]*?```", "", text)  # Code blocks
        text = re.sub(r"---+", "", text)  # Horizontal rules

        return text.strip()

    def _tokenize_spanish_text(self, text: str) -> List[str]:
        """Tokenize Spanish text properly, handling accents and Spanish-specific patterns."""
        # Remove punctuation but preserve Spanish characters
        text = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", text)

        # Split into words and filter empty strings
        words = [word.strip() for word in text.split() if word.strip()]

        # Filter out very short words (likely not meaningful)
        words = [word for word in words if len(word) > 2]

        # Optional: Remove Spanish stop words if available
        try:
            from nltk.corpus import stopwords

            spanish_stops = set(stopwords.words("spanish"))
            words = [word for word in words if word.lower() not in spanish_stops]
        except:
            # If Spanish stop words not available, continue without filtering
            pass

        return words

    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text content from PDF bytes."""
        try:
            from io import BytesIO

            import PyPDF2

            # Create a BytesIO object from the PDF content
            pdf_stream = BytesIO(pdf_content)

            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_stream)

            text_content = ""

            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num}: {e}")
                    continue

            return text_content

        except Exception as e:
            logger.warning(f"Could not extract text from PDF: {e}")
            return ""

    def _clean_pdf_text(self, text: str) -> str:
        """Clean extracted PDF text for analysis."""
        # Remove excessive whitespace and normalize
        text = re.sub(r"\s+", " ", text)

        # Remove common PDF artifacts
        text = re.sub(r"\x0c", "", text)  # Form feed characters
        text = re.sub(r"\uf0b7", "", text)  # Bullet points
        text = re.sub(r"\u2022", "", text)  # Bullet points

        # Remove page numbers (common patterns)
        text = re.sub(r"\n\d+\n", "\n", text)
        text = re.sub(r"Página \d+", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Page \d+", "", text, flags=re.IGNORECASE)

        # Remove headers/footers (lines with very few words)
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Keep lines with substantial content (more than 3 words)
            if len(line.split()) > 3:
                cleaned_lines.append(line)

        text = " ".join(cleaned_lines)

        # Remove URLs and email addresses
        text = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            text,
        )
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Normalize spaces again
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def analyze_all_repositories(self, group_name: str, output_csv: str):
        """
        Main method to analyze all repositories and save results to CSV.

        Args:
            group_name: Name of the group to analyze
            output_csv: Path to output CSV file
        """
        logger.info(f"Starting analysis for group: {group_name}")

        # Step 1: Extract repository URLs
        repos_info = self.extract_repos_from_github_directory(group_name)

        # Step 2 & 3: Analyze each repository
        results = []

        for repo_info in repos_info:
            logger.info(f"Processing: {repo_info['md_filename']}")

            if repo_info["status"] == "ERROR":
                # Repository URL extraction failed
                result = {
                    "md_filename": repo_info["md_filename"],
                    "repo_url": repo_info["repo_url"],
                    "status": repo_info["status"],
                    "error_message": repo_info["error_message"],
                    **self._get_empty_analysis_dict(),
                }
            else:
                # Analyze the repository
                analysis = self.analyze_repository(repo_info["repo_url"])
                result = {
                    "md_filename": repo_info["md_filename"],
                    "repo_url": repo_info["repo_url"],
                    "username": repo_info.get("username", ""),
                    **analysis,
                    **self._get_empty_analysis_dict(),
                }
                # Update with actual analysis results
                result.update(analysis)

            results.append(result)

        # Step 5: Save to CSV
        self._save_to_csv(results, output_csv)
        logger.info(f"Analysis complete. Results saved to: {output_csv}")

    def _get_empty_analysis_dict(self) -> Dict:
        """Return a dictionary with empty analysis fields."""
        return {
            "repo_name": "",
            "repo_owner": "",
            "num_commits": 0,
            "total_commit_size": 0,
            "mean_commit_size": 0,
            "median_commit_size": 0,
            "commit_analysis_status": "",
            "total_files_analyzed": 0,
            "md_files_count": 0,
            "pdf_files_count": 0,
            "avg_readability_score": 0,
            "avg_text_complexity": 0,
            "total_word_count": 0,
            "unique_word_count": 0,
            "sentiment_score": 0,
            "content_quality_score": 0,
            "content_analysis_status": "",
        }

    def _save_to_csv(self, results: List[Dict], output_csv: str):
        """Save analysis results to CSV file."""
        fieldnames = [
            "md_filename",
            "repo_url",
            "username",
            "repo_name",
            "repo_owner",
            "status",
            "error_message",
            "num_commits",
            "total_commit_size",
            "mean_commit_size",
            "median_commit_size",
            "commit_analysis_status",
            "total_files_analyzed",
            "md_files_count",
            "pdf_files_count",
            "avg_readability_score",
            "avg_text_complexity",
            "total_word_count",
            "unique_word_count",
            "sentiment_score",
            "content_quality_score",
            "content_analysis_status",
        ]

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                # Ensure all fieldnames are present
                row = {field: result.get(field, "") for field in fieldnames}
                writer.writerow(row)


def main():
    """Main function to run the GitHub repository analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze GitHub repositories and output metrics to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python github_repo_analyzer.py GROUP_NAME your_github_token 2024-01-01 2024-12-31 output.csv

Content Quality Analysis includes:
  - File count and types (MD, PDF)
  - Readability scores using Flesch Reading Ease
  - Text complexity (average sentence length)
  - Word count and vocabulary diversity
  - Sentiment analysis using keyword matching
  - Overall content quality score (0-100)

The content quality score combines multiple factors:
  - Number of files (more content = higher score)
  - Total word count (more comprehensive = higher score)  
  - Readability (easier to read = higher score)
  - Vocabulary diversity (varied language = higher score)
        """,
    )

    parser.add_argument("group_name", help="Name of the group directory to analyze")
    parser.add_argument("github_token", help="GitHub personal access token")
    parser.add_argument(
        "start_date", help="Start date for commit analysis (YYYY-MM-DD)"
    )
    parser.add_argument("end_date", help="End date for commit analysis (YYYY-MM-DD)")
    parser.add_argument("output_csv", help="Output CSV file path")

    args = parser.parse_args()

    try:
        analyzer = GitHubRepoAnalyzer(args.github_token, args.start_date, args.end_date)
        analyzer.analyze_all_repositories(args.group_name, args.output_csv)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
