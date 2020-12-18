import numpy as np
from smartcore.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_array_equal


# Data is just 6 separable points in the plane
X = np.array([[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]])
y = np.array([1, 1, 1, 2, 2, 2])

# A bit more random tests
rng = np.random.RandomState(0)
X1 = rng.normal(size=(10, 3))
y1 = (rng.normal(size=(10)) > 0).astype(int)

# Data is 6 random integer points in a 100 dimensional space classified to
# three classes.
X2 = rng.randint(5, size=(6, 100))
y2 = np.array([1, 1, 2, 2, 3, 3])


def test_gnb():
    # Gaussian Naive Bayes classification.
    # This checks that GaussianNB implements fit and predict and returns
    # correct values for a simple toy dataset.

    clf = GaussianNB()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert_array_equal(y_pred, y)


def test_gnb_priors_sum_isclose():
    # test whether the class prior sum is properly tested"""
    X = np.array(
        [
            [-1, -1],
            [-2, -1],
            [-3, -2],
            [-4, -5],
            [-5, -4],
            [1, 1],
            [2, 1],
            [3, 2],
            [4, 4],
            [5, 5],
        ]
    )
    priors = np.array([0.08, 0.14, 0.03, 0.16, 0.11, 0.16, 0.07, 0.14, 0.11, 0.0])
    Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    clf = GaussianNB(priors=priors)
    # smoke test for issue #9633
    clf.fit(X, Y)


def test_gnb_naive_bayes_scale_invariance():
    # Scaling the data should not change the prediction results
    iris = load_iris()
    X, y = iris.data, iris.target
    labels = []
    for f in [1e-10, 1, 1e10]:
        clf = GaussianNB()
        clf.fit(f * X, y)
        labels += [clf.predict(f * X)]
    assert_array_equal(labels[0], labels[1])
    assert_array_equal(labels[1], labels[2])
