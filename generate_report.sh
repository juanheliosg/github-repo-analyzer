#!/bin/bash

# Visual Report Generator Script
# Generates PowerPoint-ready visualizations from CSV analysis results

# Function to show usage
show_usage() {
    echo "Uso: $0 [GRUPO] [CARPETA_SALIDA]"
    echo ""
    echo "Parámetros:"
    echo "  GRUPO           Grupo específico a analizar (ej: A1, B3, ALL)"
    echo "                  Si no se especifica, analiza todos los grupos"
    echo "  CARPETA_SALIDA  Carpeta donde guardar las visualizaciones"
    echo "                  Por defecto: ./visual_reports"
    echo ""
    echo "Ejemplos:"
    echo "  $0              # Analizar todos los grupos"
    echo "  $0 A1           # Solo grupo A1"
    echo "  $0 B3           # Solo grupo B3"
    echo "  $0 ALL reportes_completos  # Todos los grupos en carpeta personalizada"
    echo ""
}

# Parse arguments
GROUP_FILTER=""
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    show_usage
    exit 0
fi

if [ ! -z "$1" ]; then
    GROUP_FILTER="$1"
fi

# Configuration
RESULTS_FOLDER="./results"
if [ ! -z "$2" ]; then
    OUTPUT_FOLDER="$2"
else
    if [ ! -z "$GROUP_FILTER" ] && [ "$GROUP_FILTER" != "ALL" ]; then
        OUTPUT_FOLDER="./visual_reports_${GROUP_FILTER}"
    else
        OUTPUT_FOLDER="./visual_reports"
    fi
fi

echo "=== Generador de Informes Visuales ==="
if [ ! -z "$GROUP_FILTER" ] && [ "$GROUP_FILTER" != "ALL" ]; then
    echo "🎯 Generando informe para: GRUPO $GROUP_FILTER"
else
    echo "🎯 Generando informe para: TODOS LOS GRUPOS"
fi
echo "📂 Analizando archivos CSV en: $RESULTS_FOLDER"
echo "📁 Generando visualizaciones en: $OUTPUT_FOLDER"
echo ""

# Check if results folder exists
if [ ! -d "$RESULTS_FOLDER" ]; then
    echo "❌ Error: La carpeta de resultados '$RESULTS_FOLDER' no existe"
    echo "   Por favor, ejecuta primero el análisis de repositorios"
    exit 1
fi

# Check if there are CSV files and filter by group if specified
if [ ! -z "$GROUP_FILTER" ] && [ "$GROUP_FILTER" != "ALL" ]; then
    # Filter CSV files by group
    CSV_FILES=$(find "$RESULTS_FOLDER" -name "*${GROUP_FILTER}*.csv")
    CSV_COUNT=$(echo "$CSV_FILES" | wc -w)
    
    if [ "$CSV_COUNT" -eq 0 ] || [ -z "$CSV_FILES" ]; then
        echo "❌ Error: No se encontraron archivos CSV para el grupo '$GROUP_FILTER' en '$RESULTS_FOLDER'"
        echo "   Archivos disponibles:"
        find "$RESULTS_FOLDER" -name "*.csv" -exec basename {} \; | sed 's/^/   - /'
        exit 1
    fi
    
    echo "📊 Encontrados $CSV_COUNT archivos CSV para el grupo $GROUP_FILTER:"
    echo "$CSV_FILES" | sed 's/^/   - /' | xargs -I {} basename {}
else
    # All CSV files
    CSV_COUNT=$(find "$RESULTS_FOLDER" -name "*.csv" | wc -l)
    if [ "$CSV_COUNT" -eq 0 ]; then
        echo "❌ Error: No se encontraron archivos CSV en '$RESULTS_FOLDER'"
        echo "   Por favor, ejecuta primero el análisis de repositorios"
        exit 1
    fi
    
    echo "📊 Encontrados $CSV_COUNT archivos CSV para analizar:"
    find "$RESULTS_FOLDER" -name "*.csv" -exec basename {} \; | sed 's/^/   - /'
fi

# Check if Python virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Activando entorno virtual de Python..."
    source .venv/bin/activate
fi

# Prepare arguments for the Python script
PYTHON_ARGS="$RESULTS_FOLDER --output $OUTPUT_FOLDER"
if [ ! -z "$GROUP_FILTER" ] && [ "$GROUP_FILTER" != "ALL" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --group $GROUP_FILTER"
fi

# Run the visual report generator
echo "🎯 Generando informes visuales..."
python generate_visual_report.py $PYTHON_ARGS

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ¡Generación completada con éxito!"
    echo "📁 Los archivos están disponibles en: $OUTPUT_FOLDER"
    echo ""
    echo "🎯 Visualizaciones generadas:"
    echo "   1. 01_overview_report.png - Resumen general de repositorios"
    echo "   2. 02_error_analysis.png - Análisis de errores y problemas"
    echo "   3. 03_repository_metrics.png - Análisis de actividad de commits"
    
    # Open the output folder if possible
    if command -v xdg-open >/dev/null 2>&1; then
        echo "🔍 Abriendo carpeta de resultados..."
        xdg-open "$OUTPUT_FOLDER" 2>/dev/null || true
    fi
else
    echo ""
    echo "❌ Error durante la generación del informe"
    exit 1
fi