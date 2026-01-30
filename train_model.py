import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

class DiseasePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = None
        self.disease_names = None
        
    def train(self, df):
        """Train the model on dataset"""
        # Separate features and target
        X = df.drop('disease', axis=1)
        y = df['disease']
        
        self.feature_names = X.columns.tolist()
        self.disease_names = sorted(y.unique())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.3f}")
        return accuracy
    
    def predict_probabilities(self, symptoms_dict):
        """Predict disease probabilities for given symptoms"""
        # Create feature vector
        features = np.zeros(len(self.feature_names))
        
        for i, feature in enumerate(self.feature_names):
            if feature in symptoms_dict:
                features[i] = symptoms_dict[feature]
        
        # Get probabilities
        probabilities = self.model.predict_proba([features])[0]
        
        # Create result dictionary
        results = {}
        for i, disease in enumerate(self.disease_names):
            results[disease] = float(probabilities[i])
        
        return results
    
    def get_feature_importance(self):
        """Get feature importance for explainability"""
        importance = self.model.feature_importances_
        feature_importance = {}
        
        for i, feature in enumerate(self.feature_names):
            feature_importance[feature] = float(importance[i])
        
        return feature_importance
    
    def save_model(self, filepath='disease_model.pkl'):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'disease_names': self.disease_names
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath='disease_model.pkl'):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.disease_names = model_data['disease_names']

def train_model():
    """Train and save the disease prediction model"""
    # Load dataset
    try:
        df = pd.read_csv('disease_dataset.csv')
    except FileNotFoundError:
        print("Dataset not found. Generating new dataset...")
        from generate_dataset import generate_dataset
        df = generate_dataset()
        df.to_csv('disease_dataset.csv', index=False)
    
    # Initialize and train model
    predictor = DiseasePredictor()
    accuracy = predictor.train(df)
    
    # Save model
    predictor.save_model()
    
    # Save feature importance
    importance = predictor.get_feature_importance()
    with open('feature_importance.json', 'w') as f:
        json.dump(importance, f, indent=2)
    
    print("Model trained and saved successfully!")
    return predictor

if __name__ == "__main__":
    train_model()