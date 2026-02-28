from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("malnutrition_model.pkl")

# Route to serve the HTML UI
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # 1. Extract and Convert Data (Matching your Streamlit logic)
        age_months = float(data['age'])
        height_cm = float(data['height'])
        birth_weight = float(data['birth_weight'])
        weight_kg = float(data['weight'])
        mother_bmi = float(data['mother_bmi'])
        mother_height_cm = float(data['mother_height'])
        family_size = int(data['family_size'])

        # Categorical Conversions
        gender = 1 if data['gender'] == "Male" else 0
        recent_illness = 1 if data['recent_illness'] == "Yes" else 0
        immunization_status = 0 if data['immunized'] == "Yes" else 1 # Note: Logic from your code (Yes=0)
        mother_anemia = 1 if data['mother_anemia'] == "Yes" else 0
        
        # Education Map
        edu_map = {"No Education": 0, "Primary": 1, "Secondary": 2, "Higher": 3}
        mother_education = edu_map.get(data['mother_education'], 0)

        # Household Map
        income_map = {"Low": 0, "Medium": 1, "High": 2}
        income_level = income_map.get(data['income'], 0)

        residence = 1 if data['residence'] == "Urban" else 0
        water_source = 1 if data['water'] == "Yes" else 0
        sanitation = 1 if data['sanitation'] == "Yes" else 0
        dietary_diversity = 1 if data['dietary'] == "Yes" else 0
        exclusive_breastfeeding = 1 if data['breastfeeding'] == "Yes" else 0

        # 2. Create Array for Model
        input_data = np.array([[
            age_months, gender, height_cm, weight_kg,
            mother_bmi, mother_height_cm, mother_anemia,
            mother_education, birth_weight, recent_illness,
            immunization_status, income_level, residence,
            family_size, water_source, sanitation,
            exclusive_breastfeeding, dietary_diversity
        ]])

        # 3. Predict
        # Note: Ensure your model supports predict_proba, otherwise remove probability logic
        # 3. Predict
        prediction = model.predict(input_data)[0]

        # Clinical risk scoring (REPLACE probability-only logic)
        risk_score = 0

        # Child risk factors
        if height_cm < 75:
            risk_score += 20

        if weight_kg < 10:
            risk_score += 20

        if birth_weight < 2.5:
            risk_score += 15

        # Mother risk factors
        if mother_bmi < 18.5:
            risk_score += 15

        if mother_anemia == 1:
            risk_score += 10

        # Household risk factors
        if sanitation == 0:
            risk_score += 10

        if dietary_diversity == 0:
            risk_score += 10

        # Limit max score
        risk_score = min(risk_score, 100)


        # Determine Risk Level
        if risk_score < 40:
            risk_level = "LOW"
            status = "Healthy"
        elif risk_score < 75:
            risk_level = "MODERATE"
            status = "At Risk"
        else:
            risk_level = "HIGH"
            status = "Malnourished"

        return jsonify({
            'success': True,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'status': status
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)