import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


# Load datasets
train_data = pd.read_csv('train_final.csv')
test_data = pd.read_csv('test_final.csv')

# Separate features and target
X_train = train_data.drop('income>50K', axis=1)
y_train = train_data['income>50K']

# Identify categorical and continuous features
categorical_features = ['workclass', 'education', 'marital.status', 'occupation', 
                        'relationship', 'race', 'sex', 'native.country']
continuous_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Model pipeline with XGBoost
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='auc', random_state=0))])

# Parameter grid for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__subsample': [0.8, 1.0],
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Validate the optimized model
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
best_model.fit(X_tr, y_tr)
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, y_val_pred_proba)
print("Validation AUC Score:", auc_score)

# Predict probabilities for the test set
test_predictions = best_model.predict_proba(test_data.drop(columns=['ID']))[:, 1]

# Create a DataFrame for submission
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'Prediction': test_predictions
})

# Save the predictions to a CSV file
submission.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
