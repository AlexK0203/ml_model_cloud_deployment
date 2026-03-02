from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from starter.ml.data import process_data


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    # Initialize a Random Forest
    # n_estimators = 100 meaning building 100 trees

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    # Use the model to predict the classes (0 or 1)
    preds = model.predict(X)
    return preds

def compute_slices(df, feature, model, encoder, lb, cat_features):
    """
    Computes performance on slices of the data based on a categorical feature.
    Outputs results to slice_output.txt.
    """
    # 1. Get every unique value in the column (e.g., 'Bachelors', 'Masters')
    unique_values = df[feature].unique()
    
    slice_results = []

    for value in unique_values:
        # 2. Filter the dataframe for this specific slice
        slice_df = df[df[feature] == value]
        
        # 3. Process the slice data (training=False)
        X_slice, y_slice, _, _ = process_data(
            slice_df, 
            categorical_features=cat_features, 
            label="salary", 
            training=False, 
            encoder=encoder, 
            lb=lb
        )

        # 4. Get predictions and metrics
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)

        # 5. Format the output string
        result = (f"Feature: {feature} | Value: {value:20} | "
                  f"Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {fbeta:.3f}")
        slice_results.append(result)

    # 6. Write all results to the file
    with open("slice_output.txt", "a") as f:
        for line in slice_results:
            f.write(line + "\n")