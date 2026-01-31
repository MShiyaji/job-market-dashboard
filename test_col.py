from salary_predictor import SalaryPredictor

sp = SalaryPredictor()

test_locations = [
    "San Francisco, CA",
    "New York, NY",
    "Austin, TX",
    "Chicago, IL",
    "Remote",
    "Smalltown, OH",
    "London, UK"
]

print("Testing Cost of Living Classification:")
for loc in test_locations:
    tier = sp._classify_col(loc)
    print(f"'{loc}': Tier {tier}")
