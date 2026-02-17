#!/usr/bin/env python3
"""
Setup script for Disease Prediction System
Generates dataset and trains the ML model
"""

import os
import sys

def setup_project():
    """Initialize the project by generating data and training model"""
    
    print("🏥 Disease Prediction System Setup")
    print("=" * 40)
    
    # Step 1: Generate dataset
    print("📊 Generating synthetic dataset...")
    try:
        from generate_dataset import generate_dataset
        df = generate_dataset(n_samples=5000)
        df.to_csv('disease_dataset.csv', index=False)
        print(f"✅ Dataset created: {len(df)} samples, {df['disease'].nunique()} diseases")
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return False
    
    # Step 2: Train model
    print("\n🤖 Training ML model...")
    try:
        from train_model import train_model
        predictor = train_model()
        print("✅ Model trained and saved successfully!")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    
    print("\n🚀 Setup complete! Run: streamlit run app.py")
    return True

if __name__ == "__main__":
    success = setup_project()
    sys.exit(0 if success else 1)
