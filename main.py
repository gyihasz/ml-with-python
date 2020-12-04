from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from visualize import createConfusionMatrix
from dataset_import import import_datasets
from visualize import createScatterPlot
from subclass import NearestClassifiers
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

mnist, iris, orl_data, orl_labels = import_datasets()

runmode = "MNIST" # IRIS    or    ORL    or    MNIST
if runmode == "MNIST":
    #60000 / 10000 --> 1/7 --> 15%
    X = mnist.data[0:3000]
    y = mnist.target[0:3000]
    y = y.astype(int)
    pixels = X[1]
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    test_size = 0.15
    plt.imshow(pixels, cmap='gray')
    plt.show()
elif runmode == "IRIS":
    X = iris.data
    y = iris.target
else:
    X = orl_data
    y = np.ravel(orl_labels)
    pixels = X[3]
    pixels = np.rot90(pixels.reshape(30, 40), 3)
    test_size = 0.3
    plt.imshow(pixels, cmap='gray')
    plt.show()

print("------- Train/Test split --------")
print("Test ratio:", test_size)
random_state = 3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)

pca = PCA(n_components = 2)
pca.fit(X)
X_PCA = pca.transform(X)
X_train_PCA, X_test_PCA, y_train_PCA, y_test_PCA = train_test_split(X_PCA, y, test_size = test_size, random_state = random_state)


classification = "Centroid" #  KNN    -    Centroid   -    Sub
if classification == "KNN":
    print("Using KNN Classifier")
    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)

    classifier.fit(X_train_PCA, y_train_PCA)
    prediction_pca = classifier.predict(X_test_PCA)

    #Finding the best K for KNN
    k_range = range(1, 26)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        predictions = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, predictions))

    plt.plot(k_range, scores)
    plt.xlabel("Value of K for KNN")
    plt.ylabel("Testing Accuracy")

elif classification == "Centroid":
    print("Using Centroid Classifier")
    classifier = NearestCentroid()
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)

    classifier.fit(X_train_PCA, y_train_PCA)
    prediction_pca = classifier.predict(X_test_PCA)

else:
    
    subclass_number = 3
    properties = {'nr_clusters': subclass_number}
    print("Using SubClass Classifier with subclasses: ", subclass_number)
    model = NearestClassifiers.SubClassCentroid.train(X_train, y_train, properties)
    prediction = NearestClassifiers.SubClassCentroid.test(model, X_test, y_test)

    model_PCA = NearestClassifiers.SubClassCentroid.train(X_train_PCA, y_train_PCA, properties)
    prediction_pca = NearestClassifiers.SubClassCentroid.test(model_PCA, X_test_PCA, y_test_PCA)


print("Model default accuracy:", metrics.accuracy_score(y_test, prediction))
print("Model default accuracy:", metrics.accuracy_score(y_test_PCA, prediction_pca))

createConfusionMatrix(y_test, prediction)
createConfusionMatrix(y_test_PCA, prediction_pca)
createScatterPlot(X_PCA, y)