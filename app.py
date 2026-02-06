from __future__ import annotations


import csv
import calendar
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

try:
  from openpyxl import Workbook, load_workbook
  from openpyxl.utils import get_column_letter
except Exception:  # pragma: no cover - friendly failure handled at runtime
  Workbook = None  # type: ignore
  load_workbook = None  # type: ignore

from flask import Flask, redirect, request, url_for, render_template


# set base directory to the repository root (where app.py lives)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EXCEL_PATH = DATA_DIR / "marks.xlsx"
CSV_PATH = DATA_DIR / "marks.csv"

# templates are located in src/templates (we keep them there)
TEMPLATE_DIR = BASE_DIR / "src" / "templates"
app = Flask(__name__, template_folder=str(TEMPLATE_DIR))


# index template moved to src/templates/index.html


def ensure_excel_exists(path: Path, year: int) -> None:
  """Create an Excel workbook with one sheet per month for the given year if it doesn't exist.

  Sheets are named 'YYYY-MM'. Each sheet contains a header row.
  Requires openpyxl to be installed.
  """
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  if path.exists():
    return
  if Workbook is None:
    raise ImportError("openpyxl is required to create Excel workbook. Install with: python -m pip install openpyxl")
  wb = Workbook()
  # remove the default sheet
  default = wb.active
  wb.remove(default)
  for m in range(1, 13):
    key = f"{year}-{m:02}"
    ws = wb.create_sheet(title=key)
    ws.append([
      "timestamp",
      "student_name",
      "mobile",
      "address",
      "insurance_number",
      "insurance_expiry",
      "insurance_company",
    ])
    # widen columns a bit
    for i, _ in enumerate(range(1, 8), start=1):
      ws.column_dimensions[get_column_letter(i)].width = 20
  wb.save(path)


def sheet_name_for_month(month_key: str) -> str:
  # month_key expected as YYYY-MM; fall back to current month if invalid
  try:
    year, mon = month_key.split("-")
    ym = datetime(int(year), int(mon), 1)
  except Exception:
    now = datetime.now()
    ym = datetime(now.year, now.month, 1)
  return f"{ym.year}-{ym.month:02}"


def append_marks(
  excel_path: Path,
  sheet_name: str,
  student_name: str,
  mobile: str,
  address: str,
  insurance_number: str,
  insurance_expiry: str,
  insurance_company: str,
) -> None:
  """Append a row to the given sheet in the Excel workbook. Creates workbook/sheet with headers if missing."""
  if Workbook is None or load_workbook is None:
    raise ImportError("openpyxl is required to write to Excel. Install with: python -m pip install openpyxl")
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  if not excel_path.exists():
    # create workbook with the year sheets that include this sheet
    try:
      year = int(sheet_name.split("-")[0])
    except Exception:
      year = datetime.now().year
    ensure_excel_exists(excel_path, year)
  wb = load_workbook(filename=excel_path)
  if sheet_name not in wb.sheetnames:
    ws = wb.create_sheet(title=sheet_name)
    ws.append([
      "timestamp",
      "student_name",
      "mobile",
      "address",
      "insurance_number",
      "insurance_expiry",
      "insurance_company",
    ])
  ws = wb[sheet_name]
  ws.append(
    [
      datetime.now(timezone.utc).isoformat(timespec="seconds"),
      student_name,
      mobile,
      address,
      insurance_number,
      insurance_expiry,
      insurance_company,
    ]
  )
  wb.save(excel_path)


def ensure_csv_exists(path: Path) -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  if path.exists():
    return
  with path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
      csv_file,
      fieldnames=[
        "timestamp",
        "student_name",
        "mobile",
        "address",
        "insurance_number",
        "insurance_expiry",
        "insurance_company",
        "month",
      ],
    )
    writer.writeheader()


def append_to_csv(
  csv_path: Path,
  month_key: str,
  student_name: str,
  mobile: str,
  address: str,
  insurance_number: str,
  insurance_expiry: str,
  insurance_company: str,
) -> None:
  ensure_csv_exists(csv_path)
  with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(
      csv_file,
      fieldnames=[
        "timestamp",
        "student_name",
        "mobile",
        "address",
        "insurance_number",
        "insurance_expiry",
        "insurance_company",
        "month",
      ],
    )
    writer.writerow(
      {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "student_name": student_name,
        "mobile": mobile,
        "address": address,
        "insurance_number": insurance_number,
        "insurance_expiry": insurance_expiry,
        "insurance_company": insurance_company,
        "month": month_key,
      }
    )


def read_entries_from_csv(csv_path: Path, month_key: str) -> List[Dict[str, str]]:
  ensure_csv_exists(csv_path)
  entries: List[Dict[str, str]] = []
  with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
      if row.get("month") == month_key:
        entries.append(row)
  return list(reversed(entries))


def read_entries(excel_path: Path, sheet_name: str) -> List[Dict[str, str]]:
  """Read rows from a sheet in the workbook and return list of dicts (newest first)."""
  if load_workbook is None:
    raise ImportError("openpyxl is required to read Excel. Install with: python -m pip install openpyxl")
  if not excel_path.exists():
    return []
  wb = load_workbook(filename=excel_path, read_only=True)
  if sheet_name not in wb.sheetnames:
    return []
  ws = wb[sheet_name]
  rows = list(ws.rows)
  if not rows or len(rows) < 2:
    return []
  headers = [c.value for c in rows[0]]
  entries: List[Dict[str, str]] = []
  for row in rows[1:]:
    entry = {headers[i]: (row[i].value if row[i].value is not None else "") for i in range(len(headers))}
    entries.append(entry)
  # newest last -> show newest first
  return list(reversed(entries))


def list_month_tabs(year: int) -> List[Tuple[str, str]]:
  # return list of (key, display_name) for months in the given year
  months: List[Tuple[str, str]] = []
  for m in range(1, 13):
    key = f"{year}-{m:02}"
    name = f"{calendar.month_name[m]} {year}"
    months.append((key, name))
  return months


def read_all_entries_from_csv(csv_path: Path) -> List[Dict[str, str]]:
  ensure_csv_exists(csv_path)
  entries: List[Dict[str, str]] = []
  with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
      entries.append(row)
  return list(reversed(entries))


def filter_entries_by_date(entries: List[Dict[str, str]], start_date: str, end_date: str) -> List[Dict[str, str]]:
  """Filter entries whose timestamp date lies between start_date and end_date (inclusive).

  start_date and end_date are strings in YYYY-MM-DD.
  """
  try:
    sd = datetime.fromisoformat(start_date).date()
    ed = datetime.fromisoformat(end_date).date()
  except Exception:
    raise ValueError("Invalid date format; expected YYYY-MM-DD")
  out: List[Dict[str, str]] = []
  for e in entries:
    ts = e.get("timestamp")
    if not ts:
      continue
    try:
      dt = datetime.fromisoformat(ts)
    except Exception:
      # skip malformed timestamps
      continue
    d = dt.date()
    if sd <= d <= ed:
      out.append(e)
  return out


def delete_entry_from_csv(csv_path: Path, timestamp: str) -> bool:
  """Delete rows with matching timestamp from CSV. Returns True if any row was removed."""
  ensure_csv_exists(csv_path)
  removed = False
  rows: List[Dict[str, str]] = []
  with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
      if row.get("timestamp") == timestamp:
        removed = True
        continue
      rows.append(row)

  if removed:
    # write back
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
      fieldnames = [
        "timestamp",
        "student_name",
        "mobile",
        "address",
        "insurance_number",
        "insurance_expiry",
        "insurance_company",
        "month",
      ]
      writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
      writer.writeheader()
      for r in rows:
        writer.writerow(r)

  return removed


@app.route("/", methods=["GET", "POST"])
def index():
  error = None
  success = None
  # determine active month from querystring, default to current year-month
  q_month = request.args.get("month")
  now = datetime.now()
  default_month_key = f"{now.year}-{now.month:02}"
  active_month = q_month or default_month_key
  # form defaults
  form = {
    "student_name": "",
    "mobile": "",
    "address": "",
    "insurance_number": "",
    "insurance_expiry": "",
    "insurance_company": "",
  }

  if request.method == "POST":
    student_name = request.form.get("student_name", "").strip()
    mobile = request.form.get("mobile", "").strip()
    address = request.form.get("address", "").strip()
    insurance_number = request.form.get("insurance_number", "").strip()
    insurance_expiry = request.form.get("insurance_expiry", "").strip()
    insurance_company = request.form.get("insurance_company", "").strip()
    form = {
      "student_name": student_name,
      "mobile": mobile,
      "address": address,
      "insurance_number": insurance_number,
      "insurance_expiry": insurance_expiry,
      "insurance_company": insurance_company,
    }

    # Basic validations
    if not student_name:
      error = "Name is required."
    elif insurance_expiry:
      # if provided, ensure date format OK
      try:
        datetime.fromisoformat(insurance_expiry)
      except Exception:
        error = "Insurance expiry must be a valid date (YYYY-MM-DD)."

    if not error:
      # save to CSV (primary source)
      month_key = sheet_name_for_month(active_month)
      append_to_csv(
        CSV_PATH,
        month_key,
        student_name,
        mobile,
        address,
        insurance_number,
        insurance_expiry,
        insurance_company,
      )
      success = "Entry saved."
      form = {k: "" for k in form}

  # prepare UI data
  months = list_month_tabs(now.year)
  sheet = sheet_name_for_month(active_month)
  entries = read_entries_from_csv(CSV_PATH, sheet)
  # check for deletion flag from redirect
  dq = request.args.get("deleted")
  if dq == "1":
    success = "Entry deleted."
  elif dq == "0":
    error = "Failed to delete entry."
  return render_template(
    "index.html", error=error, success=success, entries=entries, form=form, months=months, active_month=active_month
  )


@app.route("/records", methods=["GET", "POST"])
def records():
  error = None
  results: List[Dict[str, str]] = []
  start_date = request.values.get("start_date", "")
  end_date = request.values.get("end_date", "")

  if request.method == "POST":
    if not start_date or not end_date:
      error = "Please provide both start and end dates."
    else:
      try:
        all_entries = read_all_entries_from_csv(CSV_PATH)
        results = filter_entries_by_date(all_entries, start_date, end_date)
      except ValueError as exc:
        error = str(exc)

  return render_template("records.html", error=error, entries=results, start_date=start_date, end_date=end_date)


@app.route("/delete", methods=["POST"])
def delete():
  ts = request.form.get("timestamp")
  next_url = request.form.get("next") or request.referrer or url_for("index")
  if not ts:
    return redirect(next_url)
  try:
    deleted = delete_entry_from_csv(CSV_PATH, ts)
  except Exception:
    deleted = False

  sep = "&" if "?" in next_url else "?"
  if deleted:
    return redirect(f"{next_url}{sep}deleted=1")
  return redirect(f"{next_url}{sep}deleted=0")


if __name__ == "__main__":
  # ensure workbook exists for current year
  now = datetime.now()
  ensure_excel_exists(EXCEL_PATH, now.year)
  # Bind to localhost for development (0.0.0.0 also works if you need remote access)
  app.run(host="0.0.0.0", port=8000, debug=True)
