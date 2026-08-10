import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.assistant_service import get_health_report

def test_get_health_report_normal_high_confidence():
    report = get_health_report("Normal", 0.9)
    assert report["risk_level"] == "Low"

def test_get_health_report_pneumonia_high_confidence():
    report = get_health_report("Pneumonia", 0.8)
    assert report["risk_level"] == "High"

def test_get_health_report_pneumonia_low_confidence():
    report = get_health_report("Pneumonia", 0.6)
    assert report["risk_level"] == "Moderate"

def test_get_health_report_unknown():
    # Test an invalid prediction string
    report = get_health_report("Indeterminate", 0.5)
    assert report["risk_level"] == "Unknown"
    assert "could not confidently classify" in report["summary"]

    # Test an empty string prediction
    report = get_health_report("", 0.5)
    assert report["risk_level"] == "Unknown"
