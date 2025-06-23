from flask import Flask, request, jsonify, render_template, send_from_directory
import os, uuid, cv2
from werkzeug.utils import secure_filename
from core import process_query
from internal.brain_model import classify_image
from internal.brain_tumor_mask import predict_mask
import logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

app = Flask(__name__)

# Ensure upload folder exists
UPLOAD_FOLDER = "internal/storage/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")  # из templates/

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    txt = data.get("message", "").strip()
    if not txt:
        return jsonify({"error": "message is required"}), 400
    return jsonify(process_query(txt))

@app.route("/brain/segment", methods=["POST"])
def brain_segment():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image'"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    ext = os.path.splitext(f.filename)[1]
    fn = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_FOLDER, secure_filename(fn))
    f.save(path)
    try:
        pred = classify_image(path)
        mask = predict_mask(path)
        mask_fn = fn.replace(ext, "_mask.jpg")
        mask_path = os.path.join(UPLOAD_FOLDER, mask_fn)
        cv2.imwrite(mask_path, mask)
        return jsonify({
            "prediction": pred,
            "mask_image_url": f"/uploads/{mask_fn}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/uploads/<filename>")
def get_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run()
