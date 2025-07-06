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
df = pd.read_csv('Data for Postpartum Depression Prediction in Bangladesh/PPD_dataset.csv')
print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.info())
print(df.describe(include='all'))
print("\nMissing values per column:")
print(df.isnull().sum())
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
print(df_clean)
def text_to_number(val):
    if pd.isnull(val):
        return np.nan
    val = str(val).strip().lower()
    word_map = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                'more than two': 3, 'more than 2': 3, 'more than five': 6,
                'more than 5': 6, 'more than six': 7, 'more than 6': 7,
                'more than eight': 9, 'more than 8': 9, 'none': 0}
    if val in word_map:
        return word_map[val]
    if 'to' in val:
        parts = val.replace('+', '').split('to')
        try:
            return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        except:
            return np.nan
    if 'month' in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] if nums else np.nan
    if 'year' in val:
        nums = [int(s) for s in val.split() if s.isdigit()]
        return nums[0] * 12 if nums else np.nan
    try:
        return float(val)
    except:
        return np.nan
    def create_new_features(df):
    df_features = df.copy()
    df_features['Age_Group'] = pd.cut(df_features['Age'], bins=[0, 25, 30, 35, 100], labels=['18-25', '26-30', '31-35', '36+'])
    
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
    
    support_mapping = {'None': 0, 'Low': 1, 'Medium': 2, 'High': 3}
    df_features['Support_Score'] = df_features['Recieved Support'].map(support_mapping)
    
    df_features['Total children'] = df_features['Total children'].apply(text_to_number)
    df_features['Number of household members'] = df_features['Number of household members'].apply(text_to_number)
    df_features['Pregnancy length'] = df_features['Pregnancy length'].apply(text_to_number)

    df_features['Age_Support_Interaction'] = df_features['Age'] * df_features['Support_Score']
    df_features['Income_Children_Interaction'] = df_features['Income_After_Num'] * df_features['Total children']
    df_features['Support_Children_Interaction'] = df_features['Support_Score'] * df_features['Total children']

    numerical_cols = ['Age', 'Income_After_Num', 'Support_Score', 'Total children']
    for col in numerical_cols:
        df_features[f'{col}_squared'] = df_features[col] ** 2

    return df_features

df_features = create_new_features(df_clean)
print("\nSample of engineered features:")
print(df_features[['Age', 'Age_Group', 'Income_Before_Num', 'Income_After_Num', 'Income_Change', 'Support_Score']].head())
model_features = [
    'Age', 'Income_Before_Num', 'Income_After_Num', 'Income_Change',
    'Support_Score', 'Total children', 'Number of household members',
    'Pregnancy length'
]
df_features[model_features] = df_features[model_features].fillna(df_features[model_features].median())
plt.figure(figsize=(10, 5))
sns.countplot(data=df_features, x='PHQ9 Result')
plt.title('Distribution of PHQ9 Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
sns.countplot(data=df_features, x='EPDS Result')
plt.title('Distribution of EPDS Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.figure(figsize=(20, 16))
correlation_matrix = df_features.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()
def prepare_data_for_modeling(df, target_column):
    numerical_features = [
        'Age', 'Income_Before_Num', 'Income_After_Num', 'Income_Change',
        'Support_Score', 'Total children', 'Number of household members',
        'Pregnancy length', 'Age_Support_Interaction', 'Income_Children_Interaction',
        'Support_Children_Interaction'
    ]

    categorical_features = [
        'Education Level', 'Marital status', 'Residence', 'Family type',
        'Relationship with husband', 'Relationship with the in-laws',
        'Relationship with the newborn', 'Feeling about motherhood',
        'Recieved Support', 'Depression before pregnancy', 'Depression during pregnancy',
        'Trust and share feelings', 'Abuse', 'Need for Support',
        'Major changes or losses during pregnancy', 'Pregnancy plan', 'Regular checkups',
        'Fear of pregnancy', 'Mode of delivery', 'Gender of newborn',
        'Birth compliancy', 'Breastfeed', 'Newborn illness', 'Worry about newborn',
        'Angry after latest child birth'
    ]

    numerical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    y = df[target_column]
    if y.dtype == 'O':
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor
def train_and_evaluate_models(X_train, X_test, y_train, y_test, preprocessor):
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced']
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.1, 1],
                'class_weight': ['balanced'],
                'solver': ['liblinear']
            }
        }
    }

    best_models = {}

    for name, model_info in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model_info['model'])
        ])

        grid_search = GridSearchCV(
            pipeline,
            param_grid={'model__' + k: v for k, v in model_info['params'].items()},
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_

        y_pred = grid_search.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{name} Best Parameters: {grid_search.best_params_}")
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

        cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
        print(f"{name} CV Score: {cv_scores.mean():.4f} ± {cv_scores.std() * 2:.4f}")

    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    y_pred_voting = voting_clf.predict(X_test)
    acc_voting = accuracy_score(y_test, y_pred_voting)
    print("\nVoting Classifier Accuracy:", acc_voting)
    print(classification_report(y_test, y_pred_voting))
X_train_phq9, X_test_phq9, y_train_phq9, y_test_phq9, preprocessor = prepare_data_for_modeling(df_features, 'PHQ9 Result')
train_and_evaluate_models(X_train_phq9, X_test_phq9, y_train_phq9, y_test_phq9, preprocessor)
X_train_epds, X_test_epds, y_train_epds, y_test_epds, preprocessor = prepare_data_for_modeling(df_features, 'EPDS Result')
train_and_evaluate_models(X_train_epds, X_test_epds, y_train_epds, y_test_epds, preprocessor)
print(df_features['PHQ9 Result'].value_counts())
# Convert PHQ9 to binary
df_features['PHQ9_Binary'] = df_features['PHQ9 Result'].apply(lambda x: 1 if x in ['Moderate', 'Moderately Severe', 'Severe'] else 0)

# Features to use
features = model_features + [
    'Age_Support_Interaction', 'Income_Children_Interaction', 
    'Support_Children_Interaction'
]

df_model = df_features.dropna(subset=features + ['PHQ9_Binary'])
X = df_model[features]
y = df_model['PHQ9_Binary']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

models_params = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'class_weight': ['balanced', None]
        }
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    'SVC': {
        'model': SVC(probability=True),
        'params': {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'class_weight': ['balanced', None]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'scale_pos_weight': [1, 2]  # helps for imbalance
        }
    }
}
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

best_models = {}

for name, config in models_params.items():
    print(f"\nTraining: {name}")
    grid = GridSearchCV(config['model'], config['params'], scoring='accuracy', cv=5, n_jobs=-1)
    grid.fit(X_train_bal, y_train_bal)
    
    best_model = grid.best_estimator_
    best_models[name] = best_model
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("Best Params:", grid.best_params_)
    print("Test Accuracy:", acc)
    print(classification_report(y_test, y_pred))
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(
    estimators=[(name, model) for name, model in best_models.items()],
    voting='soft'
)
voting.fit(X_train_bal, y_train_bal)
y_pred_voting = voting.predict(X_test_scaled)

print("\nEnsemble Voting Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_voting))
print(classification_report(y_test, y_pred_voting))
