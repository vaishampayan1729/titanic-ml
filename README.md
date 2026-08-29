# Titanic ML

A complete machine learning workflow for predicting passenger survival on the Titanic dataset.

This project explores data preprocessing, feature engineering, model development, hyperparameter experiments, and evaluation using a reproducible train/validation workflow.

## Project Goals

- Build a reproducible end-to-end machine learning workflow.
- Explore and preprocess the Titanic passenger data.
- Handle missing values and categorical features appropriately.
- Compare different model configurations.
- Evaluate model performance using a held-out validation set.
- Investigate the effect of neural-network architecture and random seed on validation accuracy.

## Dataset

The project uses the classic Titanic passenger dataset.

The target variable is `Survived`, indicating whether a passenger survived.

The features used in the final modeling workflow include:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Alone`
- `Embarked_Q`
- `Embarked_S`

## Workflow

The project follows the following workflow:

1. Explore the Titanic dataset and examine relationships between features and survival.
2. Select the features used for modeling and define `Survived` as the target.
3. Split the data into training and validation sets.
4. Handle missing values using statistics computed from the training data only.
5. Encode categorical variables and engineer the `Alone` feature.
6. Standardize the numerical features using parameters learned from the training set.
7. Train and evaluate machine learning models on the validation set.
8. Perform hyperparameter experiments for the neural network.
9. Examine the sensitivity of the neural-network results to the random seed.
10. Retrain the selected models on the complete training dataset and generate predictions for the test set.

## Results

Five classification approaches were evaluated:

| Model | Kaggle Score |
|---|---:|
| **Random Forest** | **0.77990** |
| Decision Tree | 0.76794 |
| XGBoost | 0.76555 |
| Neural Network | 0.75598 |
| Logistic Regression | 0.74641 |

Random Forest achieved the best Kaggle score among the models tested.

The neural-network experiments explored different hidden-layer sizes and numbers of training epochs. The best individual validation result was obtained with **16 hidden units and 50 epochs**, reaching **87.71% validation accuracy**.

To assess sensitivity to initialization, the selected neural-network architecture was evaluated across five random seeds. It achieved a mean validation accuracy of **87.15% ± 1.25%**.

An important observation was that validation performance did not perfectly predict the final Kaggle ranking. In particular, Random Forest and XGBoost performed similarly during validation, while Random Forest achieved the better Kaggle score on the hidden test set.

