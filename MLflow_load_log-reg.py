import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


logged_model = 'runs:/f3e2034f62ac464eb1e3296be310cdfa/G_model'
mlflow.set_tracking_uri(uri="http://localhost:5000")

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

"""Load partition df_3 data."""
df = pd.read_csv('CSV/df_train_3.csv')
df.drop("Unnamed: 0", axis=1, inplace=True)
X = df.drop('Class', axis=1).values
y = df['Class'].values
# Split the on edge data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print(X_train)
# Predict on a Pandas DataFrame.
predictions = loaded_model.predict(pd.DataFrame(X_test))
classification = classification_report(y_test, loaded_model.predict(pd.DataFrame(X_test)), target_names=['Not Fraud', 'Fraud'], zero_division=0, output_dict=True)
print (classification)
# print(predictions)