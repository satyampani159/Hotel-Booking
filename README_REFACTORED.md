# Hotel Booking Prediction — Refactored

Overview
-
This project refactors a notebook-based hotel booking cancellation prediction into a modular Python package. The pipeline ingests data, preprocesses it, trains a logistic regression model and saves the trained model to `artifacts/models/`.

Setup
-
1. Create and activate a virtual environment (recommended).

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run pipeline
-
From the repository root run:

```bash
python "Hotel Booking/Hotel Booking/pipeline/run_pipeline.py"
```

Notes
-
- Dataset should be at `Hotel Booking_DATA/hotel_bookings.csv` (already present).
- Configs and constants are in `Hotel Booking/constants/`.
- Models are saved under `artifacts/models/`.
