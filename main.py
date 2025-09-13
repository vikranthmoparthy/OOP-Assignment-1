import numpy as np
from models.multiple_linear_regression import MultipleLinearRegression
from models.k_nearest_neighbors import K_Nearest_Neighbors
from models.sklearn_wrap import LassoWrapper

# Sample regression data (all continuous)
X_reg = np.array([
    [1.2, 3.4, 5.6],
    [2.3, 4.5, 6.7],
    [3.1, 5.9, 8.8],
    [4.4, 6.2, 9.1]
])
y_reg = np.array([10.0, 13.0, 17.1, 19.3])

# Sample classification data (features continuous, labels categorical)
X_cls = np.array([
    [1.2, 3.4, 5.6],
    [2.3, 4.5, 6.7],
    [3.1, 5.9, 8.8],
    [4.4, 6.2, 9.1]
])
y_cls = np.array(['A', 'B', 'A', 'B'])

print('--- Multiple Linear Regression ---')
mlr = MultipleLinearRegression()
mlr.fit(X_reg, y_reg)
pred_mlr = mlr.predict(X_reg)
print('Parameters:', mlr.parameters)
print('Predictions:', pred_mlr)

print('\n--- K-Nearest Neighbors Classifier ---')
knn = K_Nearest_Neighbors(k=3)
knn.fit(X_cls, y_cls)
pred_knn = knn.predict(X_cls)
print('Predictions:', pred_knn)

print('\n--- Lasso Regression ---')
lasso = LassoWrapper()
lasso.fit(X_reg, y_reg)
pred_lasso = lasso.predict(X_reg)
print('Parameters:', lasso.parameters)
print('Predictions:', pred_lasso)

