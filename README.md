# Postpartum Depression Prediction Analysis

This project analyzes and predicts postpartum depression using machine learning techniques based on a dataset from Bangladesh. The analysis uses both PHQ-9 and EPDS (Edinburgh Postnatal Depression Scale) scores to assess depression levels.

## Project Overview

The project aims to:
- Analyze factors contributing to postpartum depression
- Build predictive models for depression assessment
- Identify key risk factors and their relationships
- Provide insights for healthcare professionals and researchers

## Dataset Description

The dataset contains 800 records with 51 features, including:
- Demographic information (Age, Education, Residence, etc.)
- Relationship status and support systems
- Pregnancy and delivery information
- Mental health indicators
- PHQ-9 and EPDS scores

### Key Features
- Age
- Education Level
- Marital Status
- Income (before and after pregnancy)
- Support Systems
- Relationship Factors
- Pregnancy and Delivery Information
- Mental Health Indicators

## Technical Implementation

### Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

### Key Components

1. **Data Preprocessing**
   - Handling missing values
   - Feature engineering
   - Categorical variable encoding
   - Numerical feature scaling

2. **Feature Engineering**
   - Age groups
   - Income mapping
   - Support score calculation
   - Interaction features
   - Polynomial features

3. **Model Training**
   - Random Forest
   - Gradient Boosting
   - Logistic Regression
   - Ensemble methods

4. **Evaluation Metrics**
   - Accuracy
   - Classification reports
   - Cross-validation scores
   - Feature importance analysis

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the analysis:
```bash
python ppd_analysis.py
```

## Output

The script generates:
- EDA plots
- Model performance metrics
- Feature importance visualizations
- Classification reports

## Future Improvements

1. Feature Selection
   - Implement feature importance analysis
   - Remove less significant features
   - Add more interaction features

2. Model Enhancement
   - Add more advanced ensemble methods
   - Implement class weight balancing
   - Add hyperparameter optimization

3. Additional Analysis
   - Add more sophisticated feature engineering
   - Include temporal analysis
   - Add more visualization options

## Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset providers
- Contributors and researchers in the field of maternal mental health
- Open-source community for the tools and libraries used 