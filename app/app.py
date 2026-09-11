from datetime import datetime
import os
import sys
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, session
from werkzeug.utils import secure_filename
from flask import send_file
import json
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

def load_data():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

    if not os.path.exists(DATA_FILE):
        return {"patients": {}}

    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def format_datetime(iso_str):
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%d %b %Y, %I:%M %p")
    except:
        return iso_str

# --- make 'src' importable when app runs from app/ ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from dr_hybrid_project.src import data
from src import config
DATA_FILE = os.path.join(config.PROJECT_ROOT, "data", "patient_scans.json")
from src.infer import infer_image

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "dr-secret"  # set your own


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/scanner", methods=["GET", "POST"])
def scanner():
    context = {        
        "pred": None,
        "severity": None,
        "risk": None,
        "prediction_label": None,
        "confidence": None,}
    if request.method == "POST":
        
        patient_name = request.form.get("patient_name", "").strip()
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
        if not patient_name:
            flash("Patient name is required")
            return redirect(url_for("scanner"))
        
        filename = secure_filename(f.filename)
        save_path = os.path.join(config.UPLOADS_DIR, filename)
        f.save(save_path)

        try:
            pred, proba, heatmap_path, *_ = infer_image(save_path)

            import numpy as np
            
            proba = np.array(proba)
            
            pred_index = int(np.argmax(proba))
            confidence = float(proba[pred_index]) * 100
            
            #  SECOND BEST CLASS (VERY IMPORTANT)
            second_index = int(np.argsort(proba)[-2])
            second_conf = float(proba[second_index]) * 100
            
            #  DECISION CORRECTION LOGIC
            if abs(proba[pred_index] - proba[second_index]) < 0.05:
                # If close → pick higher severity (safer for medical use)
                # pred_index = max(pred_index, second_index)
                # mark as uncertain but keep original prediction
                confidence = float(proba[pred_index]) * 100 * 0.9

            # Standardized class names
            classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
            pred_name = classes[pred_index]

            # Description (keep yours if needed)
            pred_description = config.CLASS_DESCRIPTIONS[pred_index]

            # predicted_class = int(pred)
            #  Clean label mapping
            if pred_index == 0:
                prediction_label = "No Diabetic Retinopathy"
                severity_level = "None"
                # severity_level = "Healthy"
                risk = "Low"

            elif pred_index == 1:
                prediction_label = "Mild DR"
                severity_level = "Class 1"
                risk = "Moderate"

            elif pred_index == 2:
                prediction_label = "Moderate DR"
                severity_level = "Class 2"
                risk = "Moderate"

            elif pred_index == 3:
                prediction_label = "Severe DR"
                severity_level = "Severe"
                risk = "High"

            elif pred_index == 4:
                prediction_label = "Proliferative DR"
                severity_level = "Critical"
                risk = "High"
                
            # 🧠 Confidence threshold safeguard
            if confidence < 50:
                prediction_label += " (Low Confidence)"

            # ✅ Create scan entry

            from datetime import datetime

            scan_entry = {
                "patient_name": patient_name,   # 🔥 ADD THIS
                "prediction": prediction_label,
                "pred_name": pred_name,
                "confidence": round(confidence, 1),
                "severity": severity_level,
                "risk": risk,
                "timestamp": datetime.now().strftime("%d %b %H:%M")
            }

            # ✅ Initialize history if not exists
            timestamp = datetime.utcnow().isoformat()

            data = load_data()
            patients = data.setdefault("patients", {})

            if patient_name not in patients:
                patients[patient_name] = []

            patients[patient_name].append({
                "date": timestamp,
                "result": pred_name,
                "severity": severity_level,
                "risk": risk,
                "confidence": round(confidence, 1),
                "image_path": filename,
                "overlay_path": os.path.basename(heatmap_path)
            })

            save_data(data)

            context.update({
                "pred": pred_index,
                "severity": severity_level,
                "confidence": round(confidence, 1),
                "description": pred_description,
                "pred_name": pred_name,
                "prediction_label": prediction_label,
                "proba": proba.tolist(),
                "class_names": config.CLASS_NAMES,
                "overlay_url": url_for("outputs_file", filename=os.path.basename(heatmap_path)),
                "uploaded_url": url_for("uploads_file", filename=filename),
                "risk": risk,
                "scan_done": True,
            })
        except Exception as e:
            flash(f"Inference error: {e}")
            return redirect(url_for("scanner"))
            

    return render_template("scanner.html", **context)


@app.route("/")
def index():
    # session.clear()  # clears history + last result
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    if "initialized" not in session:
        session["initialized"] = True
    data = load_data()
    patients = data.get("patients", {})

    patient_summaries = []
    history = []

    for patient_name, scans in patients.items():
        if not scans:
            continue

        latest = scans[-1]

        patient_summaries.append({
            "name": patient_name,
            "scan_count": len(scans),
            "latest_timestamp": latest.get("date"),  # keep raw for sorting
            "display_time": format_datetime(latest.get("date")),
            "latest_prediction": latest.get("result"),
            "latest_confidence": latest.get("confidence"),
            "latest_severity": latest.get("severity"),
            "latest_risk": latest.get("risk"),
        })

        # ✅ THIS MUST BE INSIDE LOOP
        for scan in scans:
            history.append({
                "patient_name": patient_name,
                "prediction": scan.get("result"),
                "confidence": scan.get("confidence"),
                "severity": scan.get("severity"),
                "risk": scan.get("risk"),
                "timestamp": format_datetime(scan.get("date"))
            })

     # ✅ OUTSIDE LOOP
    history.sort(key=lambda x: x["timestamp"], reverse=True)
    patient_summaries.sort(key=lambda x: x["latest_timestamp"], reverse=True)
    total_scans = len(history)

    # 🔍 Search
    search_query = request.args.get("search", "").lower()

    # 🎯 Filter
    filter_risk = request.args.get("risk", "all")

    # 🔽 Sort
    sort_by = request.args.get("sort", "latest")

    filtered_history = history

    # Apply search
    if search_query:
        filtered_history = [
            h for h in filtered_history
            if search_query in h.get("prediction", "").lower()
        ]

    # Apply filter
    if filter_risk != "all":
        filtered_history = [
            h for h in filtered_history
            if h.get("risk") == filter_risk
        ]

    # Apply sorting
    if sort_by == "oldest":
        filtered_history = list(reversed(filtered_history))
    elif sort_by == "confidence":
        filtered_history = sorted(filtered_history, key=lambda x: x.get("confidence", 0), reverse=True)

    #  📄 Pagination (for Recent Scans patient cards)
    page = request.args.get("page", 1, type=int)
    per_page = 3

    total = len(patient_summaries)
    total_pages = (total + per_page - 1) // per_page if total > 0 else 1

    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages
    start = (page - 1) * per_page
    end = start + per_page

    paginated_patient_summaries = patient_summaries[start:end]
    # result = session.get("last_result", {}) or {}   
    # result = session.get("last_result") or {}

    # Chart files
    accuracy = "model_accuracy_bar_chart.png"
    cm = "normalized_cm_votingclassifier.png"
    radar = "model_radar_chart.png"
    f1 = "normalized_cm_votingclassifier.png"

    accuracy_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, accuracy))
    cm_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, cm))
    f1_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, f1))
    radar_exists = os.path.exists(os.path.join(config.OUTPUTS_DIR, radar))

    healthy_count = sum(
        1 for h in history 
        if h.get("risk") == "Low"
    )

    moderate_count  = sum(
        1 for h in history 
        if h.get("risk") in ["Moderate", "Mild"]
    )

    high_risk_count = sum(
    1 for h in history 
    if h.get("risk") in ["High", "Critical"]
    )

    total_scans = len(history)

    if total_scans == 0:
        healthy_percent = moderate_percent = high_percent = 0
    else:
        healthy_percent = round((healthy_count / total_scans) * 100, 1)
        moderate_percent = round((moderate_count / total_scans) * 100, 1)
        high_percent = round((high_risk_count / total_scans) * 100, 1)   

    if history:
        latest = history[0]   # already sorted
    else:
        latest = {}

    return render_template(
        "dashboard.html",
        patient_summaries=patient_summaries,
        paginated_patient_summaries=paginated_patient_summaries,
        prediction=latest.get("prediction"),
        confidence=latest.get("confidence"),
        severity=latest.get("severity"),
        risk=latest.get("risk"),
        history=history,
        total_scans=total_scans,
        search_query=search_query,
        filter_risk=filter_risk,
        sort_by=sort_by,
        page=page,
        total_pages=total_pages,
        healthy_count=healthy_count,
        moderate_count=moderate_count,
        high_risk_count=high_risk_count,
        healthy_percent=healthy_percent,
        moderate_percent=moderate_percent,
        high_percent=high_percent,
        accuracy_url=url_for("outputs_file", filename=accuracy) if accuracy_exists else None,
        cm_url=url_for("outputs_file", filename=cm) if cm_exists else None,
        radar_url=url_for("outputs_file", filename=radar) if radar_exists else None,
        f1_url=url_for("outputs_file", filename=f1) if f1_exists else None,
        )

@app.route("/download_report/<patient_name>")
def download_report(patient_name):
    data = load_data()
    patient_scans = data.get("patients", {}).get(patient_name, [])

    if not patient_scans:
        flash("No report found")
        return redirect(url_for("dashboard"))

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    y = height - 40

    # Title
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "RetinaScan AI - Patient Report")

    y -= 25
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, y, f"Patient Name: {patient_name}")

    y -= 15
    pdf.drawString(40, y, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

    # Latest scan
    latest = patient_scans[-1] if patient_scans else {}

    y -= 30
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Latest Scan")

    y -= 20
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, y, f"Date: {format_datetime(latest.get('date'))}")
    y -= 15
    pdf.drawString(40, y, f"Result: {latest.get('result')}")
    y -= 15
    pdf.drawString(40, y, f"Severity: {latest.get('severity')}")
    y -= 15
    pdf.drawString(40, y, f"Confidence: {latest.get('confidence')}%")

    # 🔥 IMAGES SECTION
    y -= 30
    pdf.drawString(40, y - 155, "Original")
    pdf.drawString(220, y - 155, "Segmented")
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Retinal Images")

    y -= 20

    # Original Image
    image_path = latest.get("image_path")
    if image_path:
        img_file = os.path.join(config.UPLOADS_DIR, image_path)
        if os.path.exists(img_file):
            try:
                pdf.drawImage(ImageReader(img_file), 40, y - 150, width=150, height=150)
            except:
                pass

    # Segmented Image
    overlay_path = latest.get("overlay_path")
    if overlay_path:
        overlay_file = os.path.join(config.OUTPUTS_DIR, overlay_path)
        if os.path.exists(overlay_file):
            try:
                pdf.drawImage(ImageReader(overlay_file), 220, y - 150, width=150, height=150)
            except:
                pass

    y -= 170

    # Scan History
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "Scan History")

    y -= 20
    pdf.setFont("Helvetica", 10)

    for i, scan in enumerate(reversed(patient_scans), 1):
        line = f"{i}. {format_datetime(scan.get('date'))} | Result: {scan.get('result')} | Severity: {scan.get('severity')} | Confidence: {scan.get('confidence')}%"
        pdf.drawString(40, y, line[:120])
        y -= 14

        if y < 60:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)

    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{patient_name}_report.pdf",
        mimetype="application/pdf"
    )

# @app.route("/reset")
# def reset():
#     session.clear()
#     return redirect(url_for("index"))

@app.route("/outputs/<path:filename>")
def outputs_file(filename):
    # Serve anything from outputs (images, txt)
    return send_from_directory(config.OUTPUTS_DIR, filename)

@app.route("/uploads/<path:filename>")
def uploads_file(filename):
    os.makedirs(config.UPLOADS_DIR, exist_ok=True)
    return send_from_directory(config.UPLOADS_DIR, filename)


if __name__ == "__main__":
    # For direct python app/app.py runs
    port = int(os.environ.get("PORT", 5001))
    app.run(debug=True, host="0.0.0.0", port=port)
