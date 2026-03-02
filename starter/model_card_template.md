# Model Card

Census Income Predictor
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Developer: [Your Name/Alexander Kindle]

Model Type: Random Forest Classifier

Library: Scikit-learn

Version: 1.0.0

## Intended Use

Primary Use: Predicting whether an individual's annual income exceeds $50K based on census data.

Intended Users: Researchers and data science students exploring demographic trends.

Out-of-Scope: This model should not be used for making actual financial or hiring decisions, as the data is historical.

## Training Data

Source: 1994 Census database (Adult Dataset).

Size: ~32,000 rows.

Preprocessing: Categorical features were OneHotEncoded; labels were binarized.

## Evaluation Data

Split: 20% of the cleaned census data was reserved for testing.

## Metrics

Precision: 0.709 (70.9% of "High Income" predictions were correct).

Recall: 0.605 (Caught 60.5% of actual "High Income" individuals).

F1-Score: 0.653 (The harmonic mean of Precision and Recall).

## Ethical Considerations

Data Bias: The dataset is from 1994 and may reflect historical socioeconomic biases.

## Caveats and Recommendations

The data is over 30 years old; income thresholds and job types have changed significantly due to inflation and technology.

Recommendation: For modern use, retraining with updated American Community Survey (ACS) data is suggested.