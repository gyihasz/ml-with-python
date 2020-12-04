def import_datasets():
    from sklearn.datasets import fetch_openml
    from scipy.io import loadmat
    from sklearn import datasets
    import numpy as np

    mnist = fetch_openml("mnist_784")
    print("MNIST imported")

    iris = datasets.load_iris()
    print("IRIS imported")

    orl = np.rot90(loadmat('orl_data.mat').get("data"))
    orl_labels = loadmat('orl_lbls.mat').get("lbls")
    print("ORL imported")
    return mnist, iris, orl, orl_labels