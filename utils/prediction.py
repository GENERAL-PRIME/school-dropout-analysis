import numpy as np
from tensorflow.keras.models import Sequential

def predict_dropout(user_input, model, scaler, label_encoders):
    for key in label_encoders:
        if key in user_input:
            user_input[key] = label_encoders[key].transform([user_input[key]])[0]

    input_data = np.array([list(user_input.values())]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    if isinstance(model, Sequential):
        prediction = model.predict(input_scaled)
        return "Dropout" if prediction[0][0] > 0.5 else "Not Dropout"
    else:
        prediction = model.predict(input_scaled)
        return "Dropout" if prediction[0] == 1 else "Not Dropout"
