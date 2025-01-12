from sklearn.metrics import accuracy_score, confusion_matrix
from data_training_svm import train_svm
from data_simulation_svm import simulate_svm_data


def evaluate_svm():
    """
    Evaluate the SVM model using accuracy and a confusion matrix.
    """


X, y = simulate_svm_data()
model, _, _ = train_svm()

# Predict on training data
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    evaluate_svm()

    

