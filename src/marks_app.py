from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "marks.csv"

app = Flask(__name__)


def ensure_csv_exists() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CSV_PATH.exists():
        return
    with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["timestamp", "student_name", "subject", "marks"]
        )
        writer.writeheader()


def append_marks(student_name: str, subject: str, marks: str) -> None:
    ensure_csv_exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file, fieldnames=["timestamp", "student_name", "subject", "marks"]
        )
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "student_name": student_name,
                "subject": subject,
                "marks": marks,
            }
        )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        student_name = request.form.get("student_name", "").strip()
        subject = request.form.get("subject", "").strip()
        marks = request.form.get("marks", "").strip()

        if not student_name or not subject or not marks:
            return render_template(
                "index.html",
                error="Please fill out all fields before submitting.",
                form_data={
                    "student_name": student_name,
                    "subject": subject,
                    "marks": marks,
                },
            )

        append_marks(student_name, subject, marks)
        return redirect(url_for("index", submitted="true"))

    submitted = request.args.get("submitted") == "true"
    return render_template("index.html", submitted=submitted)


if __name__ == "__main__":
    ensure_csv_exists()
    app.run(host="0.0.0.0", port=8000, debug=True)
