from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib

## Step 1: Load Datasets for Different Age Groups
# datasets = {
#     "Child": "datstchild_dataset.csv",
#     "Teen": "datstteen_dataset.csv",
#     "Adult": "datstadult_dataset.csv",
#     "Middle-aged": "datstmiddle_aged_dataset.csv",
#     "Senior": "datstsenior_dataset.csv"
# }

data = pd.read_csv("adult_dataset.csv")

# Step 2: Define Parameter Weights for Each Age Group
weights = {
    "Child": {
        "sleep_patterns": 0.3,
        "social_interaction": 0.2,
        "attention_span": 0.4,
        "emotional_regulation": 0.5
    },
    "Teen": {
        "sleep_patterns": 0.3,
        "social_media_use": 0.4,
        "mood_swings": 0.5,
        "academic_stress": 0.6
    },
    "Adult": {
        "sleep_patterns": 0.2,
        "work_performance": 0.4,
        "social_interaction": 0.3,
        "exercise": 0.2,
        "eating_habits": 0.3,
        "substance_use": 0.7,
        "digital_behavior": 0.3,
        "emotional_expression": 0.4
    },
    "Middle-aged": {
        "sleep_patterns": 0.2,
        "work_life_balance": 0.4,
        "family_responsibilities": 0.5,
        "health_concerns": 0.6,
        "financial_stress": 0.5
    },
    "Senior": {
        "sleep_patterns": 0.3,
        "cognitive_decline": 0.5,
        "loneliness": 0.4,
        "physical_health": 0.6,
        "mobility": 0.5,
        "emotional_expression": 0.3
    }
}

# Step 3: Preprocess Data
data['severity'] = data['severity'].astype('category').cat.codes  # Convert severity to numeric codes

# Assign weights based on age group (e.g., Adult)
age_group = "Adult"  # Change this based on the dataset being used
age_weights = weights[age_group]

# Apply weights to the dataset
for parameter in age_weights:
    data[parameter] = data[parameter] * age_weights[parameter]

# Separate features (X) and target (y)
X = data.drop(columns=['severity', 'age'], errors='ignore')  # Features: all columns except 'severity'
y = data['severity']  # Target: 'severity'

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 4: Train and Compare Classifiers
classifiers = {
    "Decision Tree (ID3)": DecisionTreeClassifier(criterion='entropy', random_state=42),  # Using entropy for ID3
    "Random Forest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42, algorithm='SAMME'),  # Use 'SAMME' to avoid warnings
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"\n{name} Accuracy: {accuracy}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    # # Step 6: Visualize the Decision Tree
    # tree_rules = export_text(clf, feature_names=list(X.columns))
    # print("\nDecision Tree Rules:\n", tree_rules)

# Step 5: Cross-Validation
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f"\n{name} Cross-Validation Accuracy: {scores.mean()}")

# Step 6: Find the Best Model
best_model_name = max(results, key=results.get)
print(f"\nBest Model: {best_model_name} with Accuracy: {results[best_model_name]}")




# Step 7: Save the Best Model
best_model = classifiers[best_model_name]
joblib.dump(best_model, f"{best_model_name.replace(' ', '_').lower()}2_model.pkl")

# Step 8: Prediction for New User
# Ensure new_user matches the number of features in X
new_user = [6, 4, 3, 1, 2, 3, 5, 4]   # Adjust for the correct number of features
new_user_weighted = [
    value * age_weights.get(param, 1) for param, value in zip(X.columns, new_user)
]

# Convert to DataFrame for compatibility with the model
new_user_df = pd.DataFrame([new_user_weighted], columns=X.columns)

# Predict the severity for the new user using the best model
severity_prediction = best_model.predict(new_user_df)
print(f"\nPredicted Severity for New User ({best_model_name}): {severity_prediction[0]}")
