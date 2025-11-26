#!/usr/bin/env python3
"""Verify project structure and dependencies."""

import os
import sys
from pathlib import Path

def check_file_exists(filepath: str) -> bool:
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"{status} {filepath}")
    return exists

def check_directory_exists(dirpath: str) -> bool:
    """Check if a directory exists."""
    exists = os.path.isdir(dirpath)
    status = "✓" if exists else "✗"
    print(f"{status} {dirpath}/")
    return exists

def main():
    """Run verification checks."""
    print("=" * 60)
    print("Fraud Detection System - Project Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check core files
    print("\n📄 Core Files:")
    core_files = [
        "README.md",
        "QUICKSTART.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "requirements.txt",
        "setup.py",
        "Makefile",
        "Dockerfile",
        "docker-compose.yml",
        ".gitignore",
        "train.py",
    ]
    for file in core_files:
        all_good &= check_file_exists(file)
    
    # Check directories
    print("\n📁 Directories:")
    directories = [
        "src",
        "src/api",
        "src/data",
        "src/models",
        "src/utils",
        "src/visualization",
        "tests",
        "config",
        "examples",
        "data",
        "models",
    ]
    for directory in directories:
        all_good &= check_directory_exists(directory)
    
    # Check source files
    print("\n🐍 Source Files:")
    source_files = [
        "src/__init__.py",
        "src/api/app.py",
        "src/data/generator.py",
        "src/data/preprocessing.py",
        "src/models/isolation_forest.py",
        "src/models/autoencoder.py",
        "src/models/evaluation.py",
        "src/utils/config.py",
        "src/utils/logger.py",
        "src/visualization/dashboard.py",
        "src/visualization/explainability.py",
    ]
    for file in source_files:
        all_good &= check_file_exists(file)
    
    # Check test files
    print("\n🧪 Test Files:")
    test_files = [
        "tests/__init__.py",
        "tests/test_models.py",
    ]
    for file in test_files:
        all_good &= check_file_exists(file)
    
    # Check examples
    print("\n📚 Examples:")
    example_files = [
        "examples/usage_example.py",
        "examples/api_client_example.py",
    ]
    for file in example_files:
        all_good &= check_file_exists(file)
    
    # Check configuration
    print("\n⚙️  Configuration:")
    config_files = [
        "config/config.yaml",
        ".env.example",
    ]
    for file in config_files:
        all_good &= check_file_exists(file)
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ All checks passed! Project structure is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train models: python train.py")
        print("3. Start API: ./run_api.sh")
        print("4. Start dashboard: ./run_dashboard.sh")
        return 0
    else:
        print("❌ Some checks failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
