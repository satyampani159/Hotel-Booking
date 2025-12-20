import os

# This module exposes configurable paths used by the project.
# It uses the current working directory as the project root when the pipeline
# is executed from repository root. Adjust if running from other locations.

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "Hotel Booking_DATA")
DATA_FILE = "hotel_bookings.csv"
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MODEL_DIR = os.path.join(ARTIFACTS_DIR, "models")
MODEL_FILE = "logistic_model.joblib"
