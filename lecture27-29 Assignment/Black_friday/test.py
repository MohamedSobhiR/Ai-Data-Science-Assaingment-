import joblib
import numpy as np 

model = joblib.load('model.h5')
scaler = joblib.load('scaler.h5')


custom_data = np.array([1, 2, 12, 1, 2, 1, 2, 2, 0])

custom_data = scaler.transform([custom_data])

prediction = model.predict(custom_data)

print(f'the bikes count is {prediction[0]}')