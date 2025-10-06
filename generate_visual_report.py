#!/usr/bin/env python3
"""
Visual Report Generator for GitHub Repository Analysis
Creates PowerPoint-ready visualizations from CSV analysis results.

This script generates comprehensive visual reports including:
- Repository accessibility overview
- Commit analysis statistics
- Content quality metrics
- Error analysis and recommendations
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set up matplotlib for high-quality output
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10

# Color scheme for consistent branding
COLORS = {
    "success": "#2E8B57",  # Sea Green
    "error": "#DC143C",  # Crimson
    "warning": "#FF8C00",  # Dark Orange
    "info": "#4169E1",  # Royal Blue
    "neutral": "#708090",  # Slate Gray
    "background": "#F8F9FA",  # Light Gray
    "accent": "#6A5ACD",  # Slate Blue
}


class VisualReportGenerator:
    """Generates visual reports from CSV analysis data."""

    def __init__(
        self,
        results_folder: str,
        output_folder: str = "visual_reports",
        group_filter: str = None,
    ):
        """
        Initialize the report generator.

        Args:
            results_folder: Path to folder containing CSV files
            output_folder: Path to save generated visualizations
            group_filter: Filter for specific group (e.g., 'A1', 'B3')
        """
        self.results_folder = Path(results_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.group_filter = group_filter

        # Load CSV files (filtered by group if specified)
        self.data = self._load_csv_files()

    def _load_csv_files(self) -> pd.DataFrame:
        """Load and combine CSV files in the results folder (filtered by group if specified)."""
        if self.group_filter:
            # Filter CSV files by group
            csv_files = list(self.results_folder.glob(f"*{self.group_filter}*.csv"))
            filter_msg = f" for group {self.group_filter}"
        else:
            # Load all CSV files
            csv_files = list(self.results_folder.glob("*.csv"))
            filter_msg = ""

        if not csv_files:
            if self.group_filter:
                available_files = list(self.results_folder.glob("*.csv"))
                available_msg = (
                    f"\nAvailable files: {[f.name for f in available_files]}"
                )
                raise ValueError(
                    f"No CSV files found for group '{self.group_filter}' in {self.results_folder}{available_msg}"
                )
            else:
                raise ValueError(f"No CSV files found in {self.results_folder}")

        print(f"Found {len(csv_files)} CSV files{filter_msg}:")
        for file in csv_files:
            print(f"  - {file.name}")

        # Load and combine all CSV files
        all_data = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df["source_file"] = csv_file.stem  # Add source file info
                all_data.append(df)
                print(f"  ✓ Loaded {len(df)} records from {csv_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file.name}: {e}")

        if not all_data:
            raise ValueError("No valid CSV data could be loaded")

        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal records loaded: {len(combined_data)}")

        return combined_data

    def generate_overview_report(self) -> str:
        """Generate overview statistics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Create title with group filter if specified
        title = "Análisis General de Repositorios - Resumen Ejecutivo"
        if self.group_filter:
            title += f" - Grupo {self.group_filter}"

        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.95)

        # 1. Repository Status Overview
        status_counts = self.data["status"].value_counts()
        colors = [
            COLORS["success"] if status == "OK" else COLORS["error"]
            for status in status_counts.index
        ]

        wedges, texts, autotexts = ax1.pie(
            status_counts.values,
            labels=status_counts.index,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 11},
        )

        ax1.set_title(
            f"Estado de Repositorios\n(Total: {len(self.data)})",
            fontweight="bold",
            pad=20,
        )

        # Add legend with counts
        legend_labels = [
            f"{status}: {count}" for status, count in status_counts.items()
        ]
        ax1.legend(
            wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
        )

        # 2. Commits Distribution
        successful_repos = self.data[self.data["status"] == "OK"]
        if not successful_repos.empty:
            commits_data = successful_repos["num_commits"].dropna()
            if not commits_data.empty:
                ax2.hist(
                    commits_data,
                    bins=15,
                    color=COLORS["info"],
                    alpha=0.7,
                    edgecolor="black",
                )
                ax2.set_title(
                    "Distribución de Commits por Repositorio", fontweight="bold"
                )
                ax2.set_xlabel("Número de Commits")
                ax2.set_ylabel("Frecuencia")
                ax2.grid(True, alpha=0.3)

                # Add statistics text
                stats_text = f"Media: {commits_data.mean():.1f}\nMediana: {commits_data.median():.1f}\nMáximo: {commits_data.max()}"
                ax2.text(
                    0.7,
                    0.9,
                    stats_text,
                    transform=ax2.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["background"]),
                    verticalalignment="top",
                )

        # 3. Content Quality Score Distribution
        if "content_quality_score" in successful_repos.columns:
            quality_data = successful_repos["content_quality_score"].dropna()
            if not quality_data.empty:
                # Create quality categories
                quality_categories = pd.cut(
                    quality_data,
                    bins=[0, 30, 50, 70, 90, 100],
                    labels=["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"],
                )
                quality_counts = quality_categories.value_counts()

                colors_quality = [
                    COLORS["error"],
                    COLORS["warning"],
                    COLORS["neutral"],
                    COLORS["info"],
                    COLORS["success"],
                ]

                bars = ax3.bar(
                    range(len(quality_counts)),
                    quality_counts.values,
                    color=colors_quality[: len(quality_counts)],
                )
                ax3.set_title("Distribución de Calidad de Contenido", fontweight="bold")
                ax3.set_xlabel("Nivel de Calidad")
                ax3.set_ylabel("Número de Repositorios")
                ax3.set_xticks(range(len(quality_counts)))
                ax3.set_xticklabels(quality_counts.index, rotation=45)

                # Add value labels on bars
                for bar, value in zip(bars, quality_counts.values):
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        str(value),
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # 4. Files Analysis
        if all(
            col in successful_repos.columns
            for col in ["md_files_count", "pdf_files_count"]
        ):
            md_total = successful_repos["md_files_count"].sum()
            pdf_total = successful_repos["pdf_files_count"].sum()

            file_types = ["Archivos MD", "Archivos PDF"]
            file_counts = [md_total, pdf_total]

            bars = ax4.bar(
                file_types, file_counts, color=[COLORS["info"], COLORS["accent"]]
            )
            ax4.set_title("Total de Archivos Analizados", fontweight="bold")
            ax4.set_ylabel("Número de Archivos")

            # Add value labels on bars
            for bar, value in zip(bars, file_counts):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(value),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=12,
                )

        plt.tight_layout()
        output_path = self.output_folder / "01_overview_report.png"
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return str(output_path)

    def generate_error_analysis(self) -> str:
        """Generate detailed error analysis visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # Create title with group filter if specified
        title = "Análisis de Errores y Problemas de Acceso"
        if self.group_filter:
            title += f" - Grupo {self.group_filter}"

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        # 1. Error Types Analysis
        error_repos = self.data[self.data["status"] == "ERROR"]

        if not error_repos.empty:
            error_messages = error_repos["error_message"].value_counts()

            # Create horizontal bar chart for better readability
            y_pos = range(len(error_messages))
            bars = ax1.barh(
                y_pos, error_messages.values, color=COLORS["error"], alpha=0.7
            )
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(
                [
                    msg[:50] + "..." if len(msg) > 50 else msg
                    for msg in error_messages.index
                ]
            )
            ax1.set_xlabel("Número de Repositorios")
            ax1.set_title("Tipos de Errores Encontrados", fontweight="bold")
            ax1.grid(True, axis="x", alpha=0.3)

            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, error_messages.values)):
                ax1.text(
                    bar.get_width() + 0.1,
                    bar.get_y() + bar.get_height() / 2,
                    str(value),
                    ha="left",
                    va="center",
                    fontweight="bold",
                )

        # 2. Success vs Error by Source File
        if "source_file" in self.data.columns:
            source_summary = (
                self.data.groupby(["source_file", "status"])
                .size()
                .unstack(fill_value=0)
            )

            if not source_summary.empty:
                source_summary.plot(
                    kind="bar", ax=ax2, color=[COLORS["success"], COLORS["error"]]
                )
                ax2.set_title("Estado por Archivo de Origen", fontweight="bold")
                ax2.set_xlabel("Archivo CSV")
                ax2.set_ylabel("Número de Repositorios")
                ax2.legend(title="Estado")
                ax2.tick_params(axis="x", rotation=45)
                ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_folder / "02_error_analysis.png"
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return str(output_path)

    def generate_repository_metrics(self) -> str:
        """Generate individual repository metrics visualization."""
        successful_repos = self.data[self.data["status"] == "OK"]

        if successful_repos.empty:
            print("No successful repositories found for repository metrics")
            return ""

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        # Create title with group filter if specified
        title = "Métricas Individuales por Repositorio"
        if self.group_filter:
            title += f" - Grupo {self.group_filter}"

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        # 1. Number of Commits per Repository
        if "num_commits" in successful_repos.columns:
            commits_data = successful_repos["num_commits"].dropna()
            repo_names = []

            for _, row in successful_repos.iterrows():
                if pd.notna(row["num_commits"]):
                    name = row.get("repo_owner", "") or row["md_filename"].replace(
                        ".md", ""
                    )
                    repo_names.append(name[:15] + "..." if len(name) > 15 else name)

            if len(commits_data) > 0:
                # Sort by commits for better visualization
                sorted_data = sorted(
                    zip(repo_names, commits_data), key=lambda x: x[1], reverse=True
                )
                sorted_names, sorted_commits = zip(*sorted_data)

                y_pos = range(len(sorted_commits))
                bars = ax1.barh(y_pos, sorted_commits, color=COLORS["info"], alpha=0.7)
                ax1.set_yticks(y_pos)
                ax1.set_yticklabels(sorted_names, fontsize=9)
                ax1.set_xlabel("Número de Commits")
                ax1.set_title("Commits por Repositorio", fontweight="bold")
                ax1.grid(True, axis="x", alpha=0.3)

                # Add value labels
                for bar, value in zip(bars, sorted_commits):
                    ax1.text(
                        bar.get_width() + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        str(int(value)),
                        ha="left",
                        va="center",
                        fontweight="bold",
                        fontsize=8,
                    )

        # 2. Number of Words per Repository
        if "total_word_count" in successful_repos.columns:
            # Include ALL successful repositories, even those with 0 words
            word_data = successful_repos.copy()

            if not word_data.empty:
                repo_names = []
                word_counts = []

                for _, row in word_data.iterrows():
                    name = row.get("repo_owner", "") or row["md_filename"].replace(
                        ".md", ""
                    )
                    repo_names.append(name[:15] + "..." if len(name) > 15 else name)
                    # Handle NaN values and convert to int
                    word_count = (
                        int(row["total_word_count"])
                        if pd.notna(row["total_word_count"])
                        else 0
                    )
                    word_counts.append(word_count)

                # Sort by word count
                sorted_data = sorted(
                    zip(repo_names, word_counts), key=lambda x: x[1], reverse=True
                )
                sorted_names, sorted_words = zip(*sorted_data)

                y_pos = range(len(sorted_words))
                # Use different colors: blue for repos with content, gray for empty ones
                colors = [
                    COLORS["accent"] if count > 0 else COLORS["neutral"]
                    for count in sorted_words
                ]
                bars = ax2.barh(y_pos, sorted_words, color=colors, alpha=0.7)
                ax2.set_yticks(y_pos)
                ax2.set_yticklabels(sorted_names, fontsize=9)
                ax2.set_xlabel("Número de Palabras")
                ax2.set_title("Palabras por Repositorio (Todos)", fontweight="bold")
                ax2.grid(True, axis="x", alpha=0.3)

                # Add value labels
                max_value = max(sorted_words) if sorted_words else 0
                for bar, value in zip(bars, sorted_words):
                    if value > 0:
                        ax2.text(
                            bar.get_width() + max_value * 0.02,
                            bar.get_y() + bar.get_height() / 2,
                            str(int(value)),
                            ha="left",
                            va="center",
                            fontweight="bold",
                            fontsize=8,
                        )
                    else:
                        # For zero values, place text at a fixed position and make it red
                        ax2.text(
                            max_value * 0.05 if max_value > 0 else 10,
                            bar.get_y() + bar.get_height() / 2,
                            "0",
                            ha="left",
                            va="center",
                            fontweight="bold",
                            fontsize=8,
                            color="red",
                        )

        # 3. Quality Score per Repository
        if "content_quality_score" in successful_repos.columns:
            quality_data = successful_repos[
                successful_repos["content_quality_score"] > 0
            ]

            if not quality_data.empty:
                repo_names = []
                quality_scores = []

                for _, row in quality_data.iterrows():
                    name = row.get("repo_owner", "") or row["md_filename"].replace(
                        ".md", ""
                    )
                    repo_names.append(name[:15] + "..." if len(name) > 15 else name)
                    quality_scores.append(row["content_quality_score"])

                # Sort by quality score
                sorted_data = sorted(
                    zip(repo_names, quality_scores), key=lambda x: x[1], reverse=True
                )
                sorted_names, sorted_quality = zip(*sorted_data)

                # Color code by quality level
                colors = []
                for score in sorted_quality:
                    if score >= 80:
                        colors.append(COLORS["success"])
                    elif score >= 60:
                        colors.append(COLORS["info"])
                    elif score >= 40:
                        colors.append(COLORS["warning"])
                    else:
                        colors.append(COLORS["error"])

                y_pos = range(len(sorted_quality))
                bars = ax3.barh(y_pos, sorted_quality, color=colors, alpha=0.7)
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(sorted_names, fontsize=9)
                ax3.set_xlabel("Puntuación de Calidad (0-100)")
                ax3.set_title("Calidad por Repositorio", fontweight="bold")
                ax3.grid(True, axis="x", alpha=0.3)

                # Add value labels
                for bar, value in zip(bars, sorted_quality):
                    ax3.text(
                        bar.get_width() + 1,
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}",
                        ha="left",
                        va="center",
                        fontweight="bold",
                        fontsize=8,
                    )

        plt.tight_layout()
        output_path = self.output_folder / "03_repository_metrics.png"
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return str(output_path)

    def generate_error_repositories(self) -> str:
        """Generate visualization of repositories with errors."""
        error_repos = self.data[self.data["status"] == "ERROR"]

        if error_repos.empty:
            print("No repositories with errors found")
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        # Create title with group filter if specified
        title = "Repositorios con Errores"
        if self.group_filter:
            title += f" - Grupo {self.group_filter}"

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        # 1. List of Error Repositories
        repo_names = []
        error_messages = []

        for _, row in error_repos.iterrows():
            name = row.get("repo_owner", "") or row["md_filename"].replace(".md", "")
            repo_names.append(name[:20] + "..." if len(name) > 20 else name)
            # Truncate error messages for display
            error_msg = (
                row["error_message"][:30] + "..."
                if len(row["error_message"]) > 30
                else row["error_message"]
            )
            error_messages.append(error_msg)

        y_pos = range(len(repo_names))
        bars = ax1.barh(y_pos, [1] * len(repo_names), color=COLORS["error"], alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(repo_names, fontsize=10)
        ax1.set_xlabel("Repositorios con Error")
        ax1.set_title(
            f"Lista de Repositorios con Errores ({len(error_repos)})", fontweight="bold"
        )
        ax1.set_xlim(0, 1.2)

        # Add error messages as text
        for i, (bar, msg) in enumerate(zip(bars, error_messages)):
            ax1.text(
                0.05,
                bar.get_y() + bar.get_height() / 2,
                msg,
                ha="left",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

        # 2. Error Types Distribution
        error_type_counts = error_repos["error_message"].value_counts()

        # Simplify error messages for the chart
        simplified_errors = {}
        for error, count in error_type_counts.items():
            if "could not access to Repo" in error:
                simplified_errors["Repositorio Inaccesible"] = (
                    simplified_errors.get("Repositorio Inaccesible", 0) + count
                )
            elif "wrong URL" in error:
                simplified_errors["URL Incorrecta"] = (
                    simplified_errors.get("URL Incorrecta", 0) + count
                )
            elif "Directory" in error and "not found" in error:
                simplified_errors["Directorio No Encontrado"] = (
                    simplified_errors.get("Directorio No Encontrado", 0) + count
                )
            else:
                simplified_errors["Otros Errores"] = (
                    simplified_errors.get("Otros Errores", 0) + count
                )

        if simplified_errors:
            colors_errors = [
                COLORS["error"],
                COLORS["warning"],
                COLORS["neutral"],
                COLORS["info"],
            ]
            wedges, texts, autotexts = ax2.pie(
                simplified_errors.values(),
                labels=simplified_errors.keys(),
                colors=colors_errors[: len(simplified_errors)],
                autopct="%1.1f%%",
                startangle=90,
            )
            ax2.set_title("Distribución de Tipos de Error", fontweight="bold")

        plt.tight_layout()
        output_path = self.output_folder / "04_error_repositories.png"
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return str(output_path)

    def generate_summary_statistics(self) -> str:
        """Generate summary statistics table visualization."""
        successful_repos = self.data[self.data["status"] == "OK"]

        fig, ax = plt.subplots(figsize=(14, 10))
        # Create title with group filter if specified
        title = "Estadísticas Resumidas de la Clase"
        if self.group_filter:
            title += f" - Grupo {self.group_filter}"

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.95)

        # Prepare statistics
        stats_data = []

        # Repository statistics
        total_repos = len(self.data)
        successful_repos_count = len(successful_repos)
        error_repos_count = len(self.data[self.data["status"] == "ERROR"])
        success_rate = (
            (successful_repos_count / total_repos * 100) if total_repos > 0 else 0
        )

        stats_data.extend(
            [
                ["Repositorios Totales", str(total_repos), "General"],
                ["Repositorios Accesibles", str(successful_repos_count), "General"],
                ["Repositorios con Error", str(error_repos_count), "General"],
                ["Tasa de Éxito", f"{success_rate:.1f}%", "General"],
            ]
        )

        # Commit statistics
        if not successful_repos.empty and "num_commits" in successful_repos.columns:
            commits_data = successful_repos["num_commits"].dropna()
            if not commits_data.empty:
                stats_data.extend(
                    [
                        ["Commits Totales", str(commits_data.sum()), "Actividad"],
                        ["Commits Promedio", f"{commits_data.mean():.1f}", "Actividad"],
                        [
                            "Commits Mediana",
                            f"{commits_data.median():.1f}",
                            "Actividad",
                        ],
                        ["Commits Máximo", str(commits_data.max()), "Actividad"],
                    ]
                )

        # Content statistics
        if not successful_repos.empty:
            for col, label in [
                ("total_files_analyzed", "Archivos Analizados (Total)"),
                ("md_files_count", "Archivos MD (Total)"),
                ("pdf_files_count", "Archivos PDF (Total)"),
                ("total_word_count", "Palabras Totales"),
                ("content_quality_score", "Calidad Promedio"),
                ("avg_readability_score", "Legibilidad Promedio"),
            ]:
                if col in successful_repos.columns:
                    col_data = successful_repos[col].dropna()
                    if not col_data.empty:
                        if "Total" in label:
                            value = str(col_data.sum())
                        else:
                            value = f"{col_data.mean():.1f}"
                        stats_data.append([label, value, "Contenido"])

        # Create table
        if stats_data:
            df_stats = pd.DataFrame(
                stats_data, columns=["Métrica", "Valor", "Categoría"]
            )

            # Group by category
            categories = df_stats["Categoría"].unique()

            y_pos = 0.9
            for category in categories:
                cat_data = df_stats[df_stats["Categoría"] == category]

                # Category header
                ax.text(
                    0.05,
                    y_pos,
                    category.upper(),
                    fontsize=14,
                    fontweight="bold",
                    color=COLORS["info"],
                    transform=ax.transAxes,
                )
                y_pos -= 0.08

                # Category data
                for _, row in cat_data.iterrows():
                    ax.text(
                        0.1, y_pos, row["Métrica"], fontsize=12, transform=ax.transAxes
                    )
                    ax.text(
                        0.7,
                        y_pos,
                        row["Valor"],
                        fontsize=12,
                        fontweight="bold",
                        transform=ax.transAxes,
                    )
                    y_pos -= 0.06

                y_pos -= 0.04  # Extra space between categories

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Add background
        ax.add_patch(
            plt.Rectangle(
                (0.05, 0.05), 0.9, 0.9, facecolor=COLORS["background"], alpha=0.3
            )
        )

        output_path = self.output_folder / "05_summary_statistics.png"
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return str(output_path)

    def generate_repository_list(self) -> str:
        """Generate a list of repositories with their status."""
        fig, ax = plt.subplots(figsize=(14, max(8, len(self.data) * 0.3)))
        # Create title with group filter if specified
        title = "Lista Detallada de Repositorios"
        if self.group_filter:
            title += f" - Grupo {self.group_filter}"

        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

        # Prepare data for display
        display_data = []

        for _, row in self.data.iterrows():
            student_name = row.get("repo_owner", "") or row["md_filename"].replace(
                ".md", ""
            )
            status = "✓" if row["status"] == "OK" else "✗"
            commits = (
                str(int(row["num_commits"])) if pd.notna(row["num_commits"]) else "0"
            )
            quality = (
                f"{row['content_quality_score']:.1f}"
                if pd.notna(row["content_quality_score"])
                else "0"
            )

            display_data.append(
                [
                    student_name[:25],  # Truncate long names
                    status,
                    commits,
                    quality,
                    row["status"],  # For coloring
                ]
            )

        # Create table
        if display_data:
            table_data = []
            colors = []

            for i, (name, status, commits, quality, full_status) in enumerate(
                display_data
            ):
                table_data.append([name, status, commits, quality])

                # Color rows based on status (using light colors)
                if full_status == "OK":
                    colors.append(["lightgreen"] * 4)  # Light green
                else:
                    colors.append(["lightcoral"] * 4)  # Light red

            table = ax.table(
                cellText=table_data,
                colLabels=["Estudiante", "Estado", "Commits", "Calidad"],
                cellColours=colors,
                cellLoc="center",
                loc="center",
                colWidths=[0.4, 0.15, 0.2, 0.25],
            )

            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Style the header
            for i in range(4):
                table[(0, i)].set_facecolor(COLORS["info"])
                table[(0, i)].set_text_props(weight="bold", color="white")

        ax.axis("off")

        output_path = self.output_folder / "06_repository_list.png"
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        plt.close()

        return str(output_path)

    def generate_complete_report(self) -> List[str]:
        """Generate all report sections and return list of output files."""
        print("Generating visual report...")

        output_files = []

        try:
            print("1. Generating overview report...")
            output_files.append(self.generate_overview_report())

            print("2. Generating error analysis...")
            output_files.append(self.generate_error_analysis())

            print("3. Generating repository metrics...")
            output_files.append(self.generate_repository_metrics())

            print("4. Generating error repositories...")
            output_files.append(self.generate_error_repositories())

            print("5. Generating summary statistics...")
            output_files.append(self.generate_summary_statistics())

            print("6. Generating repository list...")
            output_files.append(self.generate_repository_list())

        except Exception as e:
            print(f"Error generating report section: {e}")

        # Filter out empty results
        output_files = [f for f in output_files if f]

        print(f"\nReport generation complete!")
        print(f"Generated {len(output_files)} visualization files:")
        for file in output_files:
            print(f"  - {file}")

        return output_files


def main():
    """Main function to run the visual report generator."""
    parser = argparse.ArgumentParser(
        description="Generate visual reports from GitHub repository analysis CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python generate_visual_report.py results/
  python generate_visual_report.py results/ --output reports/

This will generate PowerPoint-ready visualizations including:
  - Repository overview and accessibility
  - Error analysis and troubleshooting
  - Commit activity analysis
  - Content quality metrics
  - Summary statistics
  - Detailed repository listing
        """,
    )

    parser.add_argument("results_folder", help="Folder containing CSV analysis results")
    parser.add_argument(
        "--output",
        "-o",
        default="visual_reports",
        help="Output folder for generated visualizations",
    )
    parser.add_argument("--group", "-g", help="Filter by specific group (e.g., A1, B3)")

    args = parser.parse_args()

    try:
        # Validate results folder
        if not os.path.exists(args.results_folder):
            print(f"Error: Results folder '{args.results_folder}' does not exist")
            sys.exit(1)

        # Generate report
        generator = VisualReportGenerator(args.results_folder, args.output, args.group)
        output_files = generator.generate_complete_report()

        if output_files:
            print(f"\n✅ Success! Generated {len(output_files)} visualization files")
            print(f"📁 Output folder: {args.output}")
            print("\n🎯 These images are ready for PowerPoint presentations!")
        else:
            print("❌ No visualizations were generated")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
