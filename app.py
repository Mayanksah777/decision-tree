from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("decision_tree_(id3)2_model.pkl")

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

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request data
        data = request.get_json()
        if 'age' not in data or 'answers' not in data:
            return jsonify({"error": "Missing 'age' or 'answers' in the request"}), 400

        age = data['age']
        answers = data['answers']

        # Validate input types
        if not isinstance(age, int) or not isinstance(answers, list) or not all(isinstance(x, (int, float)) for x in answers):
            return jsonify({"error": "Invalid data format"}), 400

        # Determine age group
        if age <= 12:
            age_group = "Child"
        elif age <= 20:
            age_group = "Teen"
        elif age <= 40:
            age_group = "Adult"
        elif age <= 60:
            age_group = "Middle-aged"
        else:
            age_group = "Senior"

        # Get age weights and validate parameters
        age_weights = weights.get(age_group, {})
        if len(answers) != len(age_weights):
            return jsonify({"error": "Mismatch in the number of parameters"}), 400

        # Apply weights
        parameters = list(age_weights.keys())
        weighted_answers = [value * age_weights[param] for param, value in zip(parameters, answers)]

        # Create input DataFrame for the model
        user_data = pd.DataFrame([weighted_answers], columns=parameters)

        # Predict severity
        severity = model.predict(user_data)[0]

        # Convert severity to a native Python type (int)
        severity = int(severity)

        return jsonify({"severity": severity})

    except Exception as e:
        # Handle any errors and log for debugging
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
