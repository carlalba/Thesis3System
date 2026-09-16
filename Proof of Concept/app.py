from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np

app = Flask(__name__)

# Load FULL pipelines (contain scaler + PCA + model)
svm_grid = joblib.load("saved_models/Grid-SVM_final_run1.pkl")
svm_ga = joblib.load("saved_models/GA-SVM_final_run1.pkl")

# gcForest uses PCA separately
pca_gc = joblib.load("saved_models/pca_final_run1.pkl")
gc_forest = joblib.load("saved_models/gcForest_final_run1.pkl")

label_encoder = joblib.load("saved_models/label_encoder.pkl")


@app.route("/", methods=["GET"])
def index():
    return send_from_directory("static", "gene_expression_classifier.html")


def get_confidence(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        return float(np.max(proba))
    return 1.0


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        X = np.array(data).reshape(1, -1)

        # -------------------------------
        # GRID-SVM and GA-SVM require NO MANUAL preprocess
        # the pipeline handles:
        # scaler → PCA → model
        # -------------------------------
        pred_grid = svm_grid.predict(X)[0]
        pred_ga   = svm_ga.predict(X)[0]

        conf_grid = get_confidence(svm_grid, X)
        conf_ga   = get_confidence(svm_ga, X)

        # -------------------------------
        # gcForest requires PCA manually (only gcForest)
        # -------------------------------
        X_pca_gc = pca_gc.transform(X)
        pred_gc  = gc_forest.predict(X_pca_gc)[0]
        conf_gc  = get_confidence(gc_forest, X_pca_gc)

        return jsonify({
            "GridSVM": {
                "prediction": label_encoder.inverse_transform([pred_grid])[0],
                "confidence": conf_grid
            },
            "GASVM": {
                "prediction": label_encoder.inverse_transform([pred_ga])[0],
                "confidence": conf_ga
            },
            "gcForest": {
                "prediction": label_encoder.inverse_transform([pred_gc])[0],
                "confidence": conf_gc
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
