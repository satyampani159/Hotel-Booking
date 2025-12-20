#!/usr/bin/env python
"""
Model Verification Script
Loads and inspects the trained logistic regression model.
Run this script standalone to verify model integrity.
"""
import sys
import os

# Add the inner package to path
sys.path.insert(0, os.path.join(os.getcwd(), 'Hotel Booking'))

from utils.model_inspector import inspect_model

if __name__ == '__main__':
    model_path = os.path.join(os.getcwd(), 'artifacts', 'models', 'logistic_model.joblib')
    
    print("\n" + "="*60)
    print("MODEL VERIFICATION SCRIPT")
    print("="*60 + "\n")
    
    model = inspect_model(model_path)
    
    print("\n" + "="*60)
    print("✓ Model loaded and verified successfully!")
    print("="*60 + "\n")
