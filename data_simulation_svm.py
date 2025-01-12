import numpy as np
import matplotlib.pyplot as plt

def simulate_svm_data():
    """
    Generate 2D synthetic data for classification using SVM
    """

    np.random.seed(42)
    X1= np.random.randn(50,2)+(2,2) #Class 1
    X2= np.random.randn(50,2)+(-2,-2) #Class 2
    XCombined=np.vstack([X1,X2])  #combine both classes
    YCombined=np.array([0]*50 +[1]*50)

    plt.scatter(XCombined[:, 0], XCombined[:, 1], c=YCombined, cmap="bwr", alpha=0.8)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Simulated SVM Data")
    plt.show()
    return XCombined, YCombined

if __name__ == "__main__":
    XCombined, YCombined = simulate_svm_data()






