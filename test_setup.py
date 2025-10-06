#!/usr/bin/env python3
"""
Test script to validate the GitHub Repository Analyzer setup.
"""

import importlib
import sys


def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        "github",
        "pandas",
        "dateutil",
        "nltk",
        "textstat",
        "PyPDF2",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)

    return missing_packages


def test_nltk_data():
    """Test if required NLTK data is available."""
    try:
        import nltk

        # Test downloading data
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("vader_lexicon", quiet=True)
        nltk.download("punkt_tab", quiet=True)

        print("✓ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ NLTK data download failed: {e}")
        return False


def test_spanish_support():
    """Test Spanish language support features."""
    try:
        import nltk
        from nltk.corpus import stopwords

        # Test Spanish stop words
        try:
            spanish_stops = stopwords.words("spanish")
            print(f"✓ Spanish stop words available ({len(spanish_stops)} words)")
            spanish_ok = True
        except:
            print(
                "⚠️  Spanish stop words not available (will work without stop word filtering)"
            )
            spanish_ok = False

        # Test Spanish text processing
        import re

        test_text = (
            "Esta es una prueba de análisis de texto en español con acentos áéíóú"
        )
        cleaned = re.sub(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ]", " ", test_text.lower())
        words = [word for word in cleaned.split() if len(word) > 2]

        if len(words) > 0:
            print("✓ Spanish text tokenization working")
            return True
        else:
            print("✗ Spanish text tokenization failed")
            return False

    except Exception as e:
        print(f"✗ Spanish support test failed: {e}")
        return False


def main():
    """Main test function."""
    print("GitHub Repository Analyzer - Setup Test (Spanish Language Support)")
    print("=" * 65)

    print("\n1. Testing package imports...")
    missing = test_imports()

    print("\n2. Testing NLTK data...")
    nltk_ok = test_nltk_data()

    print("\n3. Testing Spanish language support...")
    spanish_ok = test_spanish_support()

    print("\n" + "=" * 65)

    if missing:
        print("❌ Setup incomplete!")
        print(f"Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    elif not nltk_ok:
        print("⚠️  Setup mostly complete, but NLTK data issues detected")
        print("NLTK features may not work properly")
        return False
    elif not spanish_ok:
        print("⚠️  Setup complete, but Spanish language features may be limited")
        print("Spanish stop word filtering will be disabled")
        print("\n✅ Ready for Spanish content analysis!")
        print("\nYou can now run the analyzer with:")
        print(
            "python github_repo_analyzer.py GROUP_NAME GITHUB_TOKEN START_DATE END_DATE OUTPUT_CSV"
        )
        return True
    else:
        print("✅ Setup complete! All packages and Spanish support available.")
        print("\n🇪🇸 Optimized for Spanish student content analysis!")
        print("\nYou can now run the analyzer with:")
        print(
            "python github_repo_analyzer.py GROUP_NAME GITHUB_TOKEN START_DATE END_DATE OUTPUT_CSV"
        )
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
