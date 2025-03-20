import pickle
import numpy as np

with open("forecaster.pkl", "rb") as f:
    forecaster = pickle.load(f)

test_input = np.random.rand(1, 720)  # Confirm this line works fine
print("Shape of test_input:", test_input.shape)

# Now call the model
prediction = forecaster.predict(test_input)
print("Prediction:", prediction)