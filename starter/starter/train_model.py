# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import joblib
import logging
import pandas as pd
import data

# Configure logging to show the time and the message
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# 1. Loading Data

df = pd.read_csv("data/census.csv")

# Clean header names from whitespace
df.columns = df.columns.str.strip()
# clean Column names from whitespaces
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# Logging
logger.info("Success: headers cleaned and removed whitespaces from rows.")
# Save cleaned dataframe
df.to_csv('data/clean_census.csv', index=False) # index false, so clean file has the exact same structure as the original, just without the messy spaces.

# Logging
logger.info("Success: Data saved to clean_census.csv.")

# Load data
cleaned_df = pd.read_csv("data/clean_census.csv")

# 2. Splitting Data
# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(cleaned_df, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# 3. Processing Training Data
X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)
# 4. Processing Testing Data
# Proces the test data with the process_data function.

X_test, y_test, _, _ = data.process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.

model = data.train_model(X_train, y_train)
logger.info("Success: Model training complete.")

# Save everything for the API and for future use
joblib.dump(model, "model/model.pkl")
joblib.dump(encoder, "model/encoder.pkl")
joblib.dump(lb, "model/label_binarizer.pkl")

logger.info("Success: Model, Encoder, and LabelBinarizer saved to /model folder.")

