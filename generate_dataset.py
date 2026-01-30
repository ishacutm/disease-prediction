import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Symptom taxonomy with categories
SYMPTOMS = {
    'Respiratory': ['cough', 'shortness_of_breath', 'chest_tightness', 'wheezing', 'sore_throat'],
    'Digestive': ['nausea', 'vomiting', 'abdominal_pain', 'diarrhea', 'loss_of_appetite'],
    'Neurological': ['headache', 'dizziness', 'memory_issues', 'seizures', 'numbness'],
    'Metabolic': ['fatigue', 'weight_loss', 'excessive_thirst', 'frequent_urination', 'fever'],
    'Cardiac': ['chest_pain', 'palpitations', 'leg_swelling']
}

# Disease-symptom probability mapping (many-to-many)
DISEASE_SYMPTOMS = {
    'Asthma': {'cough': 0.9, 'shortness_of_breath': 0.95, 'wheezing': 0.9, 'chest_tightness': 0.8},
    'Pneumonia': {'cough': 0.85, 'fever': 0.9, 'shortness_of_breath': 0.7, 'chest_pain': 0.6},
    'Diabetes': {'excessive_thirst': 0.9, 'frequent_urination': 0.9, 'fatigue': 0.8, 'weight_loss': 0.7},
    'Hypertension': {'headache': 0.7, 'dizziness': 0.6, 'chest_pain': 0.5, 'shortness_of_breath': 0.4},
    'Gastritis': {'abdominal_pain': 0.9, 'nausea': 0.8, 'vomiting': 0.6, 'loss_of_appetite': 0.7},
    'Migraine': {'headache': 0.95, 'nausea': 0.7, 'dizziness': 0.6, 'numbness': 0.4},
    'Heart_Disease': {'chest_pain': 0.9, 'shortness_of_breath': 0.8, 'palpitations': 0.8, 'fatigue': 0.7},
    'Flu': {'fever': 0.9, 'cough': 0.8, 'headache': 0.7, 'fatigue': 0.8, 'sore_throat': 0.7},
    'COVID-19': {'fever': 0.8, 'cough': 0.8, 'shortness_of_breath': 0.6, 'fatigue': 0.8, 'loss_of_appetite': 0.5},
    'Epilepsy': {'seizures': 0.95, 'memory_issues': 0.6, 'headache': 0.5, 'dizziness': 0.4},
    'IBS': {'abdominal_pain': 0.9, 'diarrhea': 0.8, 'nausea': 0.6, 'fatigue': 0.5},
    'Anemia': {'fatigue': 0.9, 'dizziness': 0.7, 'shortness_of_breath': 0.6, 'headache': 0.5},
    'Thyroid': {'fatigue': 0.8, 'weight_loss': 0.7, 'palpitations': 0.6, 'excessive_thirst': 0.4},
    'Anxiety': {'palpitations': 0.8, 'shortness_of_breath': 0.7, 'dizziness': 0.6, 'headache': 0.5},
    'Depression': {'fatigue': 0.9, 'loss_of_appetite': 0.8, 'memory_issues': 0.6, 'headache': 0.5},
    'Eczema': {'rash': 0.95, 'itching': 0.9, 'dry_skin': 0.8, 'skin_redness': 0.7},
    'Psoriasis': {'rash': 0.9, 'dry_skin': 0.85, 'skin_redness': 0.8, 'itching': 0.6},
    'Acne': {'skin_bumps': 0.9, 'skin_redness': 0.7, 'rash': 0.6},
    'Dermatitis': {'rash': 0.9, 'itching': 0.85, 'skin_redness': 0.8, 'blisters': 0.4},
    'Fungal_Infection': {'rash': 0.8, 'itching': 0.9, 'skin_discoloration': 0.7, 'dry_skin': 0.6}
}

def generate_dataset(n_samples=5000):
    """Generate synthetic dataset with mixed symptoms"""
    np.random.seed(42)
    
    # Flatten symptoms
    all_symptoms = [s for symptoms in SYMPTOMS.values() for s in symptoms]
    all_symptoms = list(set(all_symptoms))  # Remove duplicates
    
    data = []
    diseases = list(DISEASE_SYMPTOMS.keys())
    
    for _ in range(n_samples):
        # Random disease
        disease = np.random.choice(diseases)
        
        # Generate symptoms based on disease probabilities
        row = {symptom: 0 for symptom in all_symptoms}
        
        # Primary disease symptoms
        for symptom, prob in DISEASE_SYMPTOMS[disease].items():
            if np.random.random() < prob:
                row[symptom] = 1
        
        # Add noise symptoms from other diseases (mixed symptoms)
        for other_disease in diseases:
            if other_disease != disease and np.random.random() < 0.2:  # 20% chance
                for symptom, prob in DISEASE_SYMPTOMS[other_disease].items():
                    if np.random.random() < prob * 0.3:  # Reduced probability
                        row[symptom] = 1
        
        # Medical history
        row['family_history'] = np.random.choice([0, 1], p=[0.7, 0.3])
        row['chronic_conditions'] = np.random.choice([0, 1], p=[0.8, 0.2])
        row['medications'] = np.random.choice([0, 1], p=[0.6, 0.4])
        
        # Lifestyle
        row['smoking'] = np.random.choice([0, 1], p=[0.8, 0.2])
        row['alcohol'] = np.random.choice([0, 1], p=[0.7, 0.3])
        row['exercise'] = np.random.choice([0, 1], p=[0.4, 0.6])
        row['stress_level'] = np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2])
        
        # Age group
        row['age_group'] = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.3, 0.2])
        
        row['disease'] = disease
        data.append(row)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset()
    
    # Save dataset
    df.to_csv('disease_dataset.csv', index=False)
    
    print(f"Dataset generated: {len(df)} samples")
    print(f"Features: {len(df.columns)-1}")
    print(f"Diseases: {df['disease'].nunique()} (including 5 skin diseases)")
    print("\nDisease distribution:")
    print(df['disease'].value_counts())