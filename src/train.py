import os
import pandas as pd
import joblib
import yaml
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = 'experiments'
os.makedirs(BASE_DIR, exist_ok=True)

data = pd.read_csv('data/processed_iris.csv')
X = data.drop(columns=['target'])
y = data['target']

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

default_test_size = params['train']['test_size']
default_random_state = params['train']['random_state']

for exp_name, exp_params in params['experiments'].items():
    print(f"Running {exp_name}...")

    exp_dir = os.path.join(BASE_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=exp_params.get('test_size', default_test_size),
        random_state=exp_params.get('random_state', default_random_state),
        shuffle=True
    )

    print(f"Data distribution for {exp_name}:")
    print(f"Train: {y_train.value_counts()}")
    print(f"Test: {y_test.value_counts()}")

    print(f"Training model for {exp_name} with parameters:")
    print(f"n_estimators: {exp_params['n_estimators']}, max_depth: {exp_params['max_depth']}")
    model = RandomForestClassifier(
        n_estimators=exp_params['n_estimators'],
        max_depth=exp_params['max_depth'],
        random_state=exp_params['random_state']
    )
    model.fit(X_train, y_train)

    model_path = os.path.join(exp_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    y_pred = model.predict(X_test)
    metrics = {
        'train_score': model.score(X_train, y_train),
        'test_score': model.score(X_test, y_test)
    }

    from sklearn.metrics import accuracy_score, classification_report
    metrics.update({
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    })

    metrics_path = os.path.join(exp_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics for {exp_name} saved to {metrics_path}")

    print(f"Experiment {exp_name} completed.")
