# Survival Prediction Machine Learning Pipeline

## Technical Implementation

This repository implements a complete machine learning pipeline for the Spaceship Titanic classification challenge, targeting binary prediction of passenger transportation status during a spacetime anomaly incident.

## Dependencies

```
numpy==1.22.4
pandas==1.4.3
matplotlib==3.5.2
seaborn==0.11.2
scikit-learn==1.1.2
xgboost==1.6.1
```

## Dataset Structure

The implementation processes two primary datasets:
- `train.csv`: Training corpus (n samples with target variable)
- `test.csv`: Evaluation corpus (requires prediction)

Key features include demographic data (`HomePlanet`, `Age`), travel information (`Destination`, `Cabin`), amenity consumption metrics (`RoomService`, `FoodCourt`, etc.), and status indicators (`VIP`, `CryoSleep`).

## Technical Pipeline Components

### 1. Data Pre-processing

#### Exploratory Analysis
- Statistical profiling of feature distributions
- Missing value quantification (NaN analysis)
- Target variable distribution analysis

#### Feature Extraction & Engineering
- Cabin string parsing: `Deck`, `Cabin_num`, `Side` extraction via regex splitting
- Group identification via `PassengerId` prefix extraction
- Boolean features conversion to integer representation for compatibility with imputation strategies
- Aggregation of consumption features (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`) into `TotalSpent`
- Binary feature generation for spending behavior (`HasSpent`)

### 2. Preprocessing Architecture

The implementation utilizes `sklearn.compose.ColumnTransformer` to build parallel processing pipelines:

```python
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])
```

### 3. Model Evaluation Framework

The pipeline implements a systematic comparative analysis of multiple classifier architectures:
- `RandomForestClassifier`: Ensemble of decision trees with bagging
- `GradientBoostingClassifier`: Sequential tree building with gradient optimization
- `XGBClassifier`: Implementation of gradient boosting with second-order gradient optimization

Performance evaluation metrics:
- Accuracy score (primary selection metric)
- Precision, recall and F1-score stratified by class
- Confusion matrix visualization for error analysis

### 4. Hyperparameter Optimization

Model-specific grid search implementations:

```python
# XGBoost hyperparameter space
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.1]
}

# Cross-validation strategy
grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### 5. Inference & Submission Generation

The final prediction phase employs the following process:
1. Apply identical preprocessing transformations to test data
2. Generate predictions using optimized model
3. Format output to comply with submission requirements (boolean representation)

## Execution Instructions

```bash
# Clone repository
git clone https://github.com/username/spaceship-titanic-solution.git

# Navigate to directory
cd spaceship-titanic-solution

# Execute pipeline
python spaceship_titanic_solution.py
```

## Performance Analysis

The implementation performance should be evaluated through:
- Cross-validation accuracy metrics
- Validation set performance
- Kaggle submission scores

## Potential Optimization Strategies

- Feature selection using recursive feature elimination or L1 regularization
- Dimensionality reduction via PCA or t-SNE
- Advanced missing data imputation (KNN or model-based approaches)
- Neural network architectures for potentially capturing complex interactions
- Automated hyperparameter optimization via Bayesian methods
