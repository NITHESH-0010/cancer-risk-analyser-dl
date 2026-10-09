import json
from app import app

client = app.test_client()

sample_inputs = [
    {
        "Age": 45, "BMI": 24.5, "Smoking": 0, 
        "GeneticRisk": 1, "PhysicalActivity": 3.5, "AlcoholIntake": 2.0, "CancerHistory": 0
    },
    {
        "Age": 65, "BMI": 30.2, "Smoking": 2, 
        "GeneticRisk": 2, "PhysicalActivity": 1.0, "AlcoholIntake": 14.0, "CancerHistory": 1
    },
    {
        "Age": 28, "BMI": 21.0, "Smoking": 0, 
        "GeneticRisk": 0, "PhysicalActivity": 6.0, "AlcoholIntake": 0.0, "CancerHistory": 0
    }
]

# Run Gender Independence Check First
print("--- GENDER INDEPENDENCE CHECK ---")
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

print("--- RUNNING 3 SAMPLE TESTS ---")
for i, data in enumerate(sample_inputs):
    data["Gender"] = "Female" # add for validation
    print(f"\n--- Test {i+1} ---")
    response = client.post('/predict', data=json.dumps(data), content_type='application/json')
    print("Status:", response.status_code)
    try:
        print(json.dumps(response.get_json(), indent=2))
    except:
        print(response.data)
