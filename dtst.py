import pandas as pd
import random

# Define function to create group-specific datasets
def create_dataset_for_group(age_range, parameters, severity_levels, num_rows=1000):
    """
    Create a synthetic dataset for a specific age group.
    
    Parameters:
        age_range (tuple): Min and max age for the group.
        parameters (list): List of parameter names.
        severity_levels (list): Possible severity levels.
        num_rows (int): Number of rows in the dataset.
        
    Returns:
        pd.DataFrame: Generated dataset.
    """
    dataset = {
        "age": [random.randint(age_range[0], age_range[1]) for _ in range(num_rows)],
        **{param: [random.randint(1, 10) for _ in range(num_rows)] for param in parameters},
        "severity": [random.choice(severity_levels) for _ in range(num_rows)],
    }
    return pd.DataFrame(dataset)

# Define age groups and parameters
age_groups_parameters = {
    "Child": {
        "age_range": (1, 12),
        "parameters": ["sleep_patterns", "social_interaction", "attention_span", "emotional_regulation"],
    },
    "Teen": {
        "age_range": (12, 20),
        "parameters": ["sleep_patterns", "social_media_use", "mood_swings", "academic_stress"],
    },
    "Adult": {
        "age_range": (20, 40),
        "parameters": ["sleep_patterns", "work_performance", "social_interaction", "exercise", "eating_habits", "substance_use", "digital_behavior", "emotional_expression"],
    },
    "Middle-aged": {
        "age_range": (40, 60),
        "parameters": ["sleep_patterns", "work_life_balance", "family_responsibilities", "health_concerns", "financial_stress"],
    },
    "Senior": {
        "age_range": (60, 80),
        "parameters": ["sleep_patterns", "cognitive_decline", "loneliness", "physical_health", "mobility", "emotional_expression"],
    },
}

severity_levels = ["Mild", "Moderate", "Severe"]

# Generate datasets for each age group
datasets = {}
for group, config in age_groups_parameters.items():
    datasets[group] = create_dataset_for_group(config["age_range"], config["parameters"], severity_levels)

# Save datasets to CSV files
file_paths = {}
for group, df in datasets.items():
    file_path = f"datst{group.lower()}_dataset.csv"
    df.to_csv(file_path, index=False)
    file_paths[group] = file_path

# Output file paths for reference
file_paths
