import json
from app import app

client = app.test_client()

sample_inputs = [
    {
        "profile": "Healthy",
        "Age": 30, "BMI": 22.0, "Smoking": 0, 
        "GeneticRisk": 0, "PhysicalActivity": 5.0, "AlcoholIntake": 0.0, "CancerHistory": 0
    },
    {
        "profile": "Mid-Range",
        "Age": 55, "BMI": 26.5, "Smoking": 1, 
        "GeneticRisk": 1, "PhysicalActivity": 3.0, "AlcoholIntake": 3.0, "CancerHistory": 0
    },
    {
        "profile": "High-Risk",
        "Age": 70, "BMI": 32.0, "Smoking": 2, 
        "GeneticRisk": 2, "PhysicalActivity": 1.0, "AlcoholIntake": 10.0, "CancerHistory": 1
    },
    {
        "profile": "Mixed",
        "Age": 60, "BMI": 20.0, "Smoking": 2, 
        "GeneticRisk": 0, "PhysicalActivity": 4.0, "AlcoholIntake": 0.0, "CancerHistory": 0
    }
]

# Run Gender Independence Check First on MID-range
print("--- GENDER INDEPENDENCE CHECK (Mid-Range Profile) ---")
test_profile = sample_inputs[1].copy()
genders = ["Male", "Female", "Other"]
probs = []
for g in genders:
    test_profile["Gender"] = g
    res = client.post('/predict', data=json.dumps(test_profile), content_type='application/json')
    probs.append(res.get_json()["probability"])
    print(f"Gender: {g:6} -> Probability: {probs[-1]}%")

if len(set(probs)) == 1:
    print("SUCCESS: Probability is identical regardless of Gender.\n")
else:
    print("FAILURE: Probability changed with Gender.\n")

print("--- RUNNING 4 SAMPLE PROFILES ---")
for data in sample_inputs:
    profile_name = data.pop("profile")
    data["Gender"] = "Female"
    print(f"\n--- Profile: {profile_name} ---")
    response = client.post('/predict', data=json.dumps(data), content_type='application/json')
    print("Status:", response.status_code)
    try:
        print(json.dumps(response.get_json(), indent=2))
    except:
        print(response.data)
