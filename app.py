import os
import io
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, flash
import pandas as pd
import numpy as np
from datetime import datetime

# Import modular components
import database
from prediction import FloodPredictor

# Initialize Flask
app = Flask(__name__)
app.secret_key = "rising_waters_secret_key_for_session_flash"

# Global Predictor reference (initialized on first request or main startup)
predictor = None

def get_predictor():
    """
    Lazy initialization of predictor to prevent crash if model is not yet trained
    when Flask starts up.
    """
    global predictor
    if predictor is None:
        try:
            predictor = FloodPredictor(
                model_path="models/model.pkl",
                scaler_path="models/scaler.pkl"
            )
        except FileNotFoundError:
            print("WARNING: Model/scaler files not found. Please run model_training.py to generate them.")
    return predictor

@app.before_request
def setup_app():
    """
    Ensures database is initialized.
    """
    database.init_db()

# --- ERROR HANDLERS ---
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template("error.html", error_message=str(e)), 500

# --- VIEW ROUTES ---
@app.route("/")
def index():
    """
    Landing / Home Page. Shows project hero and features.
    """
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Handles single prediction forms.
    """
    if request.method == "POST":
        # Extract inputs from form
        form_data = {
            "Annual Rainfall": request.form.get("annual_rainfall"),
            "Cloud Visibility": request.form.get("cloud_visibility"),
            "Seasonal Rainfall": request.form.get("seasonal_rainfall"),
            "Temperature": request.form.get("temperature"),
            "Humidity": request.form.get("humidity"),
            "Pressure": request.form.get("pressure"),
            "River Level": request.form.get("river_level"),
            "Wind Speed": request.form.get("wind_speed"),
            "Monsoon Intensity": request.form.get("monsoon_intensity"),
            "Average Rainfall": request.form.get("average_rainfall")
        }
        
        # Get predictor
        clf = get_predictor()
        if clf is None:
            flash("Prediction Engine is offline. Models have not been trained yet.", "danger")
            return redirect(url_for("predict"))
            
        try:
            # Perform inference
            pred_class, probability = clf.predict(form_data)
            prediction_label = "Flood Expected" if pred_class == 1 else "No Flood Expected"
            
            # Save to Database
            inputs_numeric = {k: float(v) for k, v in form_data.items()}
            row_id = database.add_prediction(inputs_numeric, prediction_label, probability)
            
            # Redirect to results page
            return redirect(url_for("result", record_id=row_id))
            
        except ValueError as e:
            flash(f"Validation Error: {str(e)}", "danger")
            return render_template("predict.html", form=request.form)
        except Exception as e:
            flash(f"System Error: {str(e)}", "danger")
            return render_template("predict.html", form=request.form)

    return render_template("predict.html")

@app.route("/result")
def result():
    """
    Result page showing prediction category, probability gauge, and PDF download options.
    """
    record_id = request.args.get("record_id")
    if not record_id:
        return redirect(url_for("predict"))
        
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        flash("Record not found.", "warning")
        return redirect(url_for("predict"))
        
    return render_template("result.html", record=dict(record))

@app.route("/dashboard")
def dashboard():
    """
    Renders the ML Model comparison, SQLite analytics, & EDA dashboard page.
    """
    # 1. Fetch dynamic stats from database
    conn = database.get_db_connection()
    cursor = conn.cursor()
    
    # Query totals
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_predictions = cursor.fetchone()[0]
    
    # Query warnings vs safe counts
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'Flood Expected'")
    total_warnings = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM predictions WHERE prediction_label = 'No Flood Expected'")
    total_safe = cursor.fetchone()[0]
    
    # Query average river level
    cursor.execute("SELECT AVG(river_level) FROM predictions")
    avg_river_level = cursor.fetchone()[0] or 0.0
    
    # Query recent 5 predictions for dashboard list
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5")
    recent_runs = [dict(row) for row in cursor.fetchall()]
    
    # Fetch all river levels & seasonal rainfall logs for Chart.js line plot
    cursor.execute("SELECT timestamp, river_level, seasonal_rainfall, probability FROM predictions ORDER BY id ASC")
    chart_rows = [dict(row) for row in cursor.fetchall()]
    
    conn.close()

    # 2. Load ML model results summary if available
    summary = None
    summary_path = "models/results_summary.pkl"
    if os.path.exists(summary_path):
        with open(summary_path, "rb") as f:
            import pickle
            summary = pickle.load(f)
            
    # Pack up all dashboard stats
    db_stats = {
        "total_predictions": total_predictions,
        "total_warnings": total_warnings,
        "total_safe": total_safe,
        "avg_river_level": round(avg_river_level, 2),
        "recent_runs": recent_runs,
        "chart_data": chart_rows
    }
            
    return render_template("dashboard.html", summary=summary, db_stats=db_stats)

@app.route("/history/export")
def export_history_csv():
    """
    Exports all predictions from the database as a CSV file.
    """
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        flash("No prediction data to export.", "warning")
        return redirect(url_for("history"))
        
    # Convert sqlite rows to pandas DataFrame
    data = [dict(row) for row in rows]
    df = pd.DataFrame(data)
    
    # Create string buffer
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Return as file download attachment
    mem_file = io.BytesIO()
    mem_file.write(output.getvalue().encode('utf-8'))
    mem_file.seek(0)
    
    return send_file(
        mem_file,
        as_attachment=True,
        download_name=f"flood_predictions_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype="text/csv"
    )

@app.route("/history")
def history():
    """
    Prediction log history.
    """
    records = database.get_all_predictions()
    return render_template("history.html", records=records)

@app.route("/delete_history/<int:record_id>", methods=["POST"])
def delete_history_item(record_id):
    """
    Deletes a specific prediction record.
    """
    deleted = database.delete_prediction(record_id)
    if deleted:
        flash("Record deleted from history.", "success")
    else:
        flash("Record not found or failed to delete.", "danger")
    return redirect(url_for("history"))

@app.route("/clear_history", methods=["POST"])
def clear_all_history():
    """
    Clears all prediction database history.
    """
    count = database.clear_history()
    flash(f"Successfully deleted {count} history record(s).", "success")
    return redirect(url_for("history"))

@app.route("/batch", methods=["GET", "POST"])
def batch_prediction():
    """
    Handles CSV file upload for batch predictions.
    """
    if request.method == "POST":
        file = request.files.get("csv_file")
        if not file or file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("batch_prediction"))
            
        if not file.filename.endswith(".csv"):
            flash("Only CSV files are supported.", "danger")
            return redirect(url_for("batch_prediction"))
            
        clf = get_predictor()
        if clf is None:
            flash("Prediction Engine is offline. Models have not been trained yet.", "danger")
            return redirect(url_for("batch_prediction"))
            
        try:
            df = pd.read_csv(file)
            result_df = clf.predict_batch(df)
            
            # Save all batch results to SQLite history
            for _, row in result_df.iterrows():
                inputs = {col: float(row[col]) for col in clf.feature_cols}
                database.add_prediction(
                    inputs, 
                    row["Prediction"], 
                    float(row["Flood Probability (%)"] / 100.0)
                )
                
            # Render predictions table
            records = result_df.to_dict(orient="records")
            return render_template("batch.html", records=records, headers=result_df.columns.tolist())
            
        except Exception as e:
            flash(f"Error processing CSV file: {str(e)}", "danger")
            return redirect(url_for("batch_prediction"))
            
    return render_template("batch.html")

# --- PDF REPORT GENERATION ROUTE ---
@app.route("/report/<int:record_id>")
def download_pdf_report(record_id):
    """
    Generates a professional PDF document for the prediction record.
    """
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions WHERE id = ?", (record_id,))
    record = cursor.fetchone()
    conn.close()
    
    if not record:
        return "Record not found", 404
        
    record_dict = dict(record)
    
    # Import reportlab components
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    story = []
    
    # Theme color definitions
    primary_color = colors.HexColor("#0B3C5D")
    secondary_color = colors.HexColor("#328CC1")
    text_dark = colors.HexColor("#1D2731")
    
    styles = getSampleStyleSheet()
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.gray,
        spaceAfter=25
    )
    
    section_title = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=secondary_color,
        spaceAfter=10,
        spaceBefore=15
    )
    
    normal_style = ParagraphStyle(
        'ReportNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=text_dark
    )
    
    bold_label_style = ParagraphStyle(
        'ReportBoldLabel',
        parent=normal_style,
        fontName='Helvetica-Bold'
    )
    
    alert_style = ParagraphStyle(
        'ReportAlert',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#D9534F") if "Expected" in record_dict["prediction_label"] and "No" not in record_dict["prediction_label"] else colors.HexColor("#5CB85C"),
        alignment=1 # Center
    )

    # 1. Document Header
    story.append(Paragraph("Rising Waters: Flood Prediction Report", title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Report Reference ID: #RW-{record_dict['id']:05d}", subtitle_style))
    story.append(Spacer(1, 10))
    
    # 2. Risk Evaluation Box
    story.append(Paragraph("Risk Evaluation Outcome", section_title))
    status_text = f"{record_dict['prediction_label'].upper()} ({record_dict['probability'] * 100:.1f}% Probability)"
    story.append(Paragraph(status_text, alert_style))
    story.append(Spacer(1, 15))
    
    # 3. Weather & Environmental Parameters Table
    story.append(Paragraph("Input Environmental Metrics", section_title))
    
    # Map raw DB keys to readable labels
    param_map = [
        ("Annual Rainfall", "mm"),
        ("Cloud Visibility", "%"),
        ("Seasonal Rainfall", "mm"),
        ("Temperature", "°C"),
        ("Humidity", "%"),
        ("Pressure", "hPa"),
        ("River Level", "m"),
        ("Wind Speed", "km/h"),
        ("Monsoon Intensity", "scale (1-10)"),
        ("Average Rainfall", "mm")
    ]
    
    table_data = []
    # Build a 2-column key-value format inside the PDF
    for label, unit in param_map:
        db_key = label.lower().replace(" ", "_")
        val = record_dict.get(db_key, "N/A")
        if isinstance(val, float):
            val_str = f"{val:.2f} {unit}" if "level" in db_key else f"{val:.1f} {unit}"
        else:
            val_str = f"{val} {unit}"
            
        table_data.append([
            Paragraph(label, bold_label_style),
            Paragraph(val_str, normal_style)
        ])
        
    t = Table(table_data, colWidths=[200, 250])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F9F9F9")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E2E2")),
        ('PADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    
    story.append(t)
    story.append(Spacer(1, 20))
    
    # 4. Disclaimer Footer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10,
        textColor=colors.gray,
        alignment=1 # Center
    )
    story.append(Spacer(1, 20))
    story.append(Paragraph("Disclaimer: This report is generated by a Machine Learning model using current inputs and does not represent official meteorological flood warnings. Please consult local emergency authorities for life-safety decisions.", disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"Flood_Report_RW_{record_dict['id']:05d}.pdf",
        mimetype="application/pdf"
    )

# --- REST API ENDPOINT ---
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for flood prediction.
    Accepts JSON body:
    {
        "Annual Rainfall": 2500,
        "Cloud Visibility": 50,
        "Seasonal Rainfall": 1200,
        "Temperature": 28.5,
        "Humidity": 75,
        "Pressure": 1005,
        "River Level": 6.5,
        "Wind Speed": 20,
        "Monsoon Intensity": 6,
        "Average Rainfall": 450
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "Missing JSON request body."}), 400
        
    clf = get_predictor()
    if clf is None:
        return jsonify({"status": "error", "message": "Prediction Engine is offline. Models not trained."}), 503
        
    try:
        # Perform prediction
        pred_class, probability = clf.predict(data)
        prediction_label = "Flood Expected" if pred_class == 1 else "No Flood Expected"
        
        # Save to database
        row_id = database.add_prediction(
            {k: float(v) for k, v in data.items() if k in clf.feature_cols}, 
            prediction_label, 
            probability
        )
        
        return jsonify({
            "status": "success",
            "prediction": prediction_label,
            "flood_probability": float(np.round(probability * 100, 2)),
            "record_id": row_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except ValueError as e:
        return jsonify({"status": "error", "message": f"Validation Error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"System Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
