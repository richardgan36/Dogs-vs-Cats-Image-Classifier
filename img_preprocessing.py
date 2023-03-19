import numpy as np
import glob
import pickle

from PIL import Image

HEIGHT_MAX, WIDTH_MAX = 768, 1050

# x_train = []
# y_train = []
i = 0

with open('x_train_dogsVcats_padded_500.pkl', 'ab') as x_file, open('y_train_dogsVcats_padded_500.pkl', 'ab') as y_file:
    for image in glob.iglob('/Users/richardgan/Pictures/Machine Learning/train_dogVcats/*'):
        img = np.asarray(Image.open(image).convert('L')) / 255  # Grayscale image as array with values in [0, 1]
        img_height, img_width = img.shape
        height_to_pad, width_to_pad = HEIGHT_MAX - img_height, WIDTH_MAX - img_width
        # Pad image with black borders on the right and bottom
        img_width_padded = np.hstack(tup=(img, np.zeros(shape=(img_height, width_to_pad))))
        img_width_and_height_padded = np.vstack(tup=(img_width_padded, np.zeros(shape=(height_to_pad, WIDTH_MAX))))
        pickle.dump(img_width_and_height_padded, x_file, pickle.HIGHEST_PROTOCOL)

        if "dog" in image[59:]:  # Only look at image name and not entire path
            # y_train.append(np.array([1, 0]))  # One-hot encoding as [dog, cat]
            y_label = np.array([1, 0])

        else:
            # y_train.append(np.array([0, 1]))
            y_label = np.array([0, 1])

        pickle.dump(y_label, y_file, pickle.HIGHEST_PROTOCOL)

        i += 1
        if i >= 5000:
            break

# with open('x_train_dogsVcats_padded.pkl', 'wb') as outfile:
#     pickle.dump(x_train, outfile, pickle.HIGHEST_PROTOCOL)
#
# with open('y_train_dogsVcats_padded.pkl', 'wb') as outfile:
#     pickle.dump(y_train, outfile, pickle.HIGHEST_PROTOCOL)
