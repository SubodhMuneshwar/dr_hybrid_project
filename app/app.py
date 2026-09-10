import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

# --- make 'src' importable when app runs from app/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.infer import infer_image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dr-secret-dev-only")
# Limit upload size to 10MB to prevent abuse
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/scanner", methods=["GET", "POST"])
def scanner():
    context = {}
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file uploaded.")
            return redirect(url_for("scanner"))
        f = request.files["file"]
        if f.filename == "":
            flash("No selected file.")
            return redirect(url_for("scanner"))
        if not allowed_file(f.filename):
            flash("Please upload a PNG/JPG image.")
            return redirect(url_for("scanner"))
        
        filename = secure_filename(f.filename)
        save_path = os.path.join(config.UPLOADS_DIR, filename)
        f.save(save_path)

        try:
            pred, proba, heatmap_path = infer_image(save_path)

            # Calculate confidence as percentage of predicted class
            confidence = float(proba[pred]) * 100
            pred_class_name = config.CLASS_NAMES[int(pred)]
            pred_description = config.CLASS_DESCRIPTIONS[int(pred)]

            context.update({
                "pred": int(pred),
                "pred_name": pred_class_name,
                "confidence": f"{confidence:.1f}",
                "description": pred_description,
                "proba": proba.tolist(),
                "class_names": config.CLASS_NAMES,
                "overlay_url": url_for("outputs_file", filename=os.path.basename(heatmap_path)),
                "uploaded_name": filename,
            })
        except Exception as e:
            flash(f"Inference error: {e}")
            return redirect(url_for("scanner"))

    return render_template("scanner.html", **context)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    # Support both legacy names and new pipeline-generated names
    # Legacy expected: model_accuracy_bar_chart.png, normalized_cm_votingclassifier.png, model_radar_chart.png
    # Current pipeline generates: stacking_confusion_matrix.png, stacking_f1_scores.png
    def find_first(candidates):
        for name in candidates:
            p = os.path.join(config.OUTPUTS_DIR, name)
            if os.path.exists(p):
                return name
        return None

    cm_name = find_first(["stacking_confusion_matrix.png", "stacking_reval_confusion_matrix.png", "normalized_cm_votingclassifier.png", "model_accuracy_bar_chart.png"])
    f1_name = find_first(["stacking_f1_scores.png", "stacking_reval_f1_scores.png", "model_accuracy_bar_chart.png", "normalized_cm_votingclassifier.png"])
    radar_name = find_first(["model_radar_chart.png", "stacking_radar.png"])

    return render_template(
        "dashboard.html",
        cm_url=url_for("outputs_file", filename=cm_name) if cm_name else None,
        f1_url=url_for("outputs_file", filename=f1_name) if f1_name else None,
        radar_url=url_for("outputs_file", filename=radar_name) if radar_name else None,
    )


@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    # Serve anything from outputs (images, txt)
    return send_from_directory(config.OUTPUTS_DIR, filename)


if __name__ == "__main__":
    # For direct python app/app.py runs
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("1", "true", "yes")
    app.run(debug=debug, host="0.0.0.0", port=port)
