# WIP

Instruction to use this in python:

1. Create a virtualenvironment with python3
2. Install maturin and numpy i.e. pip install maturin numpy
3. Run maturin develop
4. Open a python shell
5. Compare interface with sklearn, for instance

```
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from smartcore.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict(np.array([[-0.8, -1]])))
```

Is equivalent to:
```
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, Y)

print(clf.predict([[-0.8, -1]]))
```
