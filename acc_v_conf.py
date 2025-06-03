import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Step 1: Generate synthetic binary classification data
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=10,
    weights=[0.7, 0.3],
    random_state=42
)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Step 3: Train a logistic regression model
clf = LogisticRegression(C=0.01, max_iter=1000)
clf.fit(X_train, y_train)

# Step 4: Predict probabilities for class 1
prob_pos = clf.predict_proba(X_test)[:, 1]

# Step 5: Compute calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, prob_pos, n_bins=10
)

# Step 6: Plot the calibration curve with interpretation
plt.figure(figsize=(9, 7))
plt.plot(mean_predicted_value, fraction_of_positives, "o-", label="Model", color="blue")
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

# Highlight overconfident regions (model too confident)
plt.fill_between(
    mean_predicted_value, fraction_of_positives, mean_predicted_value,
    where=(fraction_of_positives < mean_predicted_value),
    color="red", alpha=0.3, label="Overconfident"
)

# Highlight underconfident regions (model too cautious)
plt.fill_between(
    mean_predicted_value, fraction_of_positives, mean_predicted_value,
    where=(fraction_of_positives > mean_predicted_value),
    color="green", alpha=0.3, label="Underconfident"
)

# Step 7: Add labels and style
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Model Calibration Curve with Confidence Interpretation")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Step 8: Save the plot (optional)
plt.savefig("enhanced_model_calibration_curve.png")
plt.show()





# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.calibration import calibration_curve
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification

# # Step 1: Generate synthetic binary classification data
# X, y = make_classification(
#     n_samples=10000, 
#     n_features=20, 
#     n_informative=10, 
#     weights=[0.7, 0.3],  # imbalance to simulate real-world scenario
#     random_state=42
# )

# # Step 2: Split into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=42
# )

# # Step 3: Train an intentionally under-regularized logistic regression model
# clf = LogisticRegression(C=0.01, max_iter=1000)  # small C = stronger regularization
# clf.fit(X_train, y_train)

# # Step 4: Get predicted probabilities
# prob_pos = clf.predict_proba(X_test)[:, 1]

# # Step 5: Compute calibration curve
# fraction_of_positives, mean_predicted_value = calibration_curve(
#     y_test, prob_pos, n_bins=10
# )

# # Step 6: Plot the calibration curve
# plt.figure(figsize=(8, 6))
# plt.plot(mean_predicted_value, fraction_of_positives, "o-", label="Model")
# plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
# plt.xlabel("Mean Predicted Probability")
# plt.ylabel("Fraction of Positives")
# plt.title("Model Calibration Curve (Reliability Diagram)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# # Optional: Save the plot
# plt.savefig("model_calibration_curve.png")
# plt.show()
