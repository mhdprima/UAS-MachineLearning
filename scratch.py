# Import library
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
from skimage.feature import hog
from skimage import exposure

# Load MNIST dataset
mnist = datasets.fetch_openml('mnist_784')
data = np.array(mnist.data, 'int16')
target = np.array(mnist.target, 'int')

# Extract HOG features
def extract_hog_features(images):
    features = []
    for img in images:
        fd, hog_image = hog(img.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True)
        features.append(fd)
    return np.array(features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Extract HOG features for training and testing sets
X_train_hog = extract_hog_features(X_train)
X_test_hog = extract_hog_features(X_test)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_hog, y_train)

# Predictions
y_pred = svm_model.predict(X_test_hog)

# Evaluate performance
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

# Print results
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
