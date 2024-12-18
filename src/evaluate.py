import os
import pandas as pd
import yaml
import json
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    def __init__(self, data_path, model_path, output_dir, exp_name, params_path):
        self.data_path = data_path
        self.model_path = model_path
        self.output_dir = output_dir
        self.exp_name = exp_name
        self.params_path = params_path
        self.test_size = None
        self._load_params()
        self._prepare_output_dir()

    def _load_params(self):
        with open(self.params_path, 'r') as file:
            params = yaml.safe_load(file)
        self.test_size = params['train']['test_size']
        print(f"Parameters loaded for experiment {self.exp_name}.")

    def _prepare_output_dir(self):
        self.exp_dir = os.path.join(self.output_dir, self.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        print(f"Results for {self.exp_name} will be saved in: {self.exp_dir}")

    def load_data(self):
        print(f"Loading data for {self.exp_name}...")
        data = pd.read_csv(self.data_path)
        self.X = data.drop(columns=['target'])
        self.y = data['target']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42
        )

    def load_model(self):
        print(f"Loading model for {self.exp_name}...")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def evaluate(self):
        print(f"Evaluating model for {self.exp_name}...")
        self.y_pred = self.model.predict(self.X_test)

        self.metrics = {
            "accuracy": accuracy_score(self.y_test, self.y_pred),
            "weighted_f1": f1_score(self.y_test, self.y_pred, average="weighted")
        }
        self.classification_report = classification_report(self.y_test, self.y_pred, output_dict=True)

    def save_metrics(self):
        print(f"Saving metrics for {self.exp_name}...")

        metrics_path = os.path.join(self.exp_dir, "metrics_ev.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        report_df = pd.DataFrame(self.classification_report).T
        report_df.reset_index(inplace=True)
        report_df.rename(columns={"index": "Class"}, inplace=True)

        report_path = os.path.join(self.exp_dir, "report.csv")
        report_df.to_csv(report_path, index=False)

        print(f"Metrics saved to {metrics_path}")
        print(f"Classification report saved to {report_path}")

    def save_roc_curves(self):
        print(f"Saving ROC Curves for {self.exp_name}...")
        y_test_bin = label_binarize(self.y_test, classes=list(range(len(set(self.y_test)))))
        n_classes = y_test_bin.shape[1]

        try:
            y_pred_proba = self.model.predict_proba(self.X_test)

            plt.figure(figsize=(10, 8))
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curves")
            plt.legend(loc="lower right")

            roc_path = os.path.join(self.exp_dir, "roc_curves.png")
            plt.savefig(roc_path)
            plt.close()
            print(f"ROC Curves saved to {roc_path}")
        except AttributeError:
            print(f"Model for {self.exp_name} does not support probability prediction.")

    def save_feature_importance(self):
        print(f"Saving Feature Importance for {self.exp_name}...")

        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
            features = self.X.columns

            plt.figure(figsize=(10, 6))
            sns.barplot(x=importance, y=features)
            plt.title("Feature Importance")
            plt.xlabel("Importance")
            plt.ylabel("Features")

            fi_path = os.path.join(self.exp_dir, "feature_importance.png")
            plt.savefig(fi_path)
            plt.close()
            print(f"Feature Importance saved to {fi_path}")
        else:
            print(f"Model for {self.exp_name} does not support feature importance.")

    def run(self):
        self.load_data()
        self.load_model()
        self.evaluate()
        self.save_metrics()
        self.save_roc_curves()
        self.save_feature_importance()
        print(f"Evaluation for {self.exp_name} completed successfully!")


if __name__ == "__main__":
    DATA_PATH = "data/processed_iris.csv"
    OUTPUT_DIR = "experiments"
    PARAMS_PATH = "params.yaml"

    with open(PARAMS_PATH, 'r') as f:
        params = yaml.safe_load(f)

    for exp_name in params['experiments']:
        MODEL_PATH = f"experiments/{exp_name}/model.pkl"
        evaluator = ModelEvaluator(DATA_PATH, MODEL_PATH, OUTPUT_DIR, exp_name, PARAMS_PATH)
        evaluator.run()
