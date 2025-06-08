import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
print("\n--- Loading Data ---")
df = pd.read_csv('Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())

# 2. Data Overview
print("\n--- Data Overview ---")
print(df.info())
print(df.describe(include='all'))
print("\nMissing values per column:")
print(df.isnull().sum())

# 3. Handle missing values
def handle_missing_values(df):
    df_clean = df.copy()
    categorical_columns = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    numerical_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

df_clean = handle_missing_values(df)
print("\nMissing values after cleaning:")
print(df_clean.isnull().sum())

# 4. Feature Engineering
def text_to_number(val):
    if pd.isnull(val):
        return np.nan
    val = str(val).strip().lower()
    # Handle words
    word_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'more than two': 3, 'more than 2': 3, 'more than five': 6,
                'more than 5': 6, 'more than six': 7, 'more than 6': 7,
                'more than eight': 9, 'more than 8': 9, 'none': 0}
    if val in word_map:
        return word_map[val]
    # Handle ranges like '6 to 8'
    if 'to' in val:
        parts = val.replace('+', '').split('to')
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        except:
            return np.nan
    # Handle months/years
    if 'month' in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] if nums else np.nan
    if 'year' in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] * 12 if nums else np.nan
    # Handle numbers
    try:
        return float(val)
    except:
        return np.nan

def create_new_features(df):
    df_features = df.copy()
    # Age groups
    df_features['Age_Group'] = pd.cut(df_features['Age'], bins=[0, 25, 30, 35, 100], labels=['18-25', '26-30', '31-35', '36+'])
    
    # Income mapping
    income_mapping = {
        'None': 0,
        'Less than 5000': 2500,
        '5000 to 10000': 7500,
        '10000 to 20000': 15000,
        '20000 to 30000': 25000,
        'More than 30000': 35000
    }
    df_features['Income_Before_Num'] = df_features['Monthly income before latest pregnancy'].map(income_mapping)
    df_features['Income_After_Num'] = df_features['Current monthly income'].map(income_mapping)
    df_features['Income_Change'] = df_features['Income_After_Num'] - df_features['Income_Before_Num']
    
    # Support score
    support_mapping = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
    df_features['Support_Score'] = df_features['Recieved Support'].map(support_mapping)
    
    # Convert problematic columns to numeric
    df_features['Total children'] = df_features['Total children'].apply(text_to_number)
    df_features['Number of household members'] = df_features['Number of household members'].apply(text_to_number)
    df_features['Pregnancy length'] = df_features['Pregnancy length'].apply(text_to_number)
    
    # New interaction features
    df_features['Age_Support_Interaction'] = df_features['Age'] * df_features['Support_Score']
    df_features['Income_Children_Interaction'] = df_features['Income_After_Num'] * df_features['Total children']
    df_features['Support_Children_Interaction'] = df_features['Support_Score'] * df_features['Total children']
    
    # Polynomial features for numerical columns
    numerical_cols = ['Age', 'Income_After_Num', 'Support_Score', 'Total children']
    for col in numerical_cols:
        df_features[f'{col}_squared'] = df_features[col] ** 2
    
    return df_features

df_features = create_new_features(df_clean)
print("\nSample of engineered features:")
print(df_features[['Age', 'Age_Group', 'Income_Before_Num', 'Income_After_Num', 'Income_Change', 'Support_Score']].head())

# Fill NaNs in modeling features
model_features = [
    'Age', 'Income_Before_Num', 'Income_After_Num', 'Income_Change',
    'Support_Score', 'Total children', 'Number of household members',
    'Pregnancy length'
]
df_features[model_features] = df_features[model_features].fillna(df_features[model_features].median())

# 5. EDA - Save plots to files
print("\n--- Saving EDA Plots ---")
plt.figure(figsize=(10, 5))
sns.countplot(data=df_features, x='PHQ9 Result')
plt.title('Distribution of PHQ9 Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('phq9_result_distribution.png')
plt.close()

plt.figure(figsize=(10, 5))
sns.countplot(data=df_features, x='EPDS Result')
plt.title('Distribution of EPDS Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('epds_result_distribution.png')
plt.close()

plt.figure(figsize=(10, 6))
correlation_matrix = df_features.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# 6. Prepare data for modeling
def prepare_data_for_modeling(df, target_column):
    # Define feature groups
    numerical_features = [
        'Age', 'Income_Before_Num', 'Income_After_Num', 'Income_Change',
        'Support_Score', 'Total children', 'Number of household members',
        'Pregnancy length', 'Age_Support_Interaction', 'Income_Children_Interaction',
        'Support_Children_Interaction'
    ]
    
    # Using only the most important categorical features with exact column names
    categorical_features = [
        'Education Level',
        'Marital status',
        'Residence',
        'Family type',
        'Relationship with husband',
        'Relationship with the in-laws',
        'Relationship with the newborn',
        'Feeling about motherhood',
        'Recieved Support',
        'Depression before pregnancy',
        'Depression during pregnancy',
        'Trust and share feelings',
        'Abuse',
        'Need for Support',
        'Major changes or losses during pregnancy',
        'Pregnancy plan',
        'Regular checkups',
        'Fear of pregnancy',
        'Mode of delivery',
        'Gender of newborn',
        'Birth compliancy',
        'Breastfeed',
        'Newborn illness',
        'Worry about newborn',
        'Angry after latest child birth'
    ]
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare target variable
    y = df[target_column]
    if y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

# 7. Model Training and Evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    # Define models with their parameter grids
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'class_weight': ['balanced', None]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1, 10],
                'class_weight': ['balanced', None],
                'solver': ['liblinear', 'saga']
            }
        }
    }
    
    best_models = {}
    
    for name, model_info in models.items():
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_info['model'])
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid={'model__' + k: v for k, v in model_info['params'].items()},
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        
        # Evaluate
        y_pred = grid_search.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Best Parameters: {grid_search.best_params_}")
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation score
        cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
        print(f"{name} Cross-validation scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create and evaluate voting classifier
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    acc_voting = accuracy_score(y_test, y_pred_voting)
    print("\nVoting Classifier Accuracy:", acc_voting)
    print(classification_report(y_test, y_pred_voting))

# 8. Run for PHQ9 Result
print("\n--- Model Training: PHQ9 Result ---")
X_train_phq9, X_test_phq9, y_train_phq9, y_test_phq9, preprocessor = prepare_data_for_modeling(df_features, 'PHQ9 Result')
train_and_evaluate_models(X_train_phq9, X_test_phq9, y_train_phq9, y_test_phq9, preprocessor)

# 9. Run for EPDS Result
print("\n--- Model Training: EPDS Result ---")
X_train_epds, X_test_epds, y_train_epds, y_test_epds, preprocessor = prepare_data_for_modeling(df_features, 'EPDS Result')
train_and_evaluate_models(X_train_epds, X_test_epds, y_train_epds, y_test_epds, preprocessor)

print("\nAnalysis complete. Key plots saved as PNG files in the current directory.") 