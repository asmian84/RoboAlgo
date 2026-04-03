"""
Root-level pytest conftest.
Ensures the project root is on sys.path so test modules can import
smc_engine and other first-party packages without a package install.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
