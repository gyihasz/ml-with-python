from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def createConfusionMatrix(y_test, prediction):
    confMatrix = confusion_matrix(y_test, prediction)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(confMatrix, origin='lower', cmap='gist_earth')
    fig.colorbar(im)
    plt.show()


def createScatterPlot(X, y):
    plt.scatter(X[:,0], X[:,1], c = y, alpha=0.8)
    plt.show()