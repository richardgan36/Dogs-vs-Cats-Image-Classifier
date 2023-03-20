"""
This script loads a trained dogs vs. cats model and predicts whether an image contains a cat or a dog.
"""

import numpy as np
from keras import models
from PIL import Image

model = models.load_model('trained_models/dogsVcats_model3.h5')  # load trained model
test_img = Image.open('/Users/richardgan/Pictures/Machine Learning/test_dogsVcats/318.jpg').convert('L')  # load image
test_img = test_img.resize((100, 100))
test_img = np.asarray(test_img) / 255
test_img = np.reshape(test_img, (1, 100, 100, 1))

prediction = model.predict(test_img)

if np.argmax(prediction) == 1:
    print("Image is a dog")

else:
    print("Image is a cat")
