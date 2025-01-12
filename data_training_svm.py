from scipy.stats import levene
from sklearn.svm import SVC
from data_simulation_svm import simulate_svm_data
import numpy as np
import matplotlib.pyplot as plt

def train_svm():
    """
    Train an SVM model and visualize the decision boundary
    """

    XCombined,YCombined= simulate_svm_data()
    model=SVC(kernel='linear',C=1.0)
    model.fit(XCombined,YCombined)

    plt.scatter(XCombined[:,0],XCombined[:,1],c=YCombined,cmap="bwr",alpha=0.5)

#plot decision summary
    x_min,x_max=XCombined[:,0].min()-1,XCombined[:,0].max()+1
    y_min,y_max=XCombined[:,1].min()-1,XCombined[:,1].max()+1
    xx,yy=np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))

#ravel the xx e.g. 2D array into a list or an array
#ravel the xx e.g. 2D array into a list or an array
#combines these two lists/1D arrays into a single array/list dataset.
#np.c_ combined these two 1D arrays into a numpy sci-learn versioned 2D array


    Z= model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z= Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,levels= np.linspace(Z.min(), Z.max(),50), cmap="coolwarm",alpha=0.6)
    plt.contour(xx,yy,levels=[0],colors="black",linestyle="[--]")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM with RBF Kernel")
    plt.show()


if __name__=="__main__":
    train_svm()
