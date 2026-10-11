"""CSV schema definitions shared by every part of the package.

Two CSV files describe a dataset (see docs/DATA_FORMAT.md for the full spec):

* ``stations.csv``     -- one row per sensing station (static metadata, graph nodes)
* ``observations.csv`` -- one row per station per timestamp (time series + labels)
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Severity classes (ordered)
# --------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "Warning", "Evacuate", "Landslide"]
NUM_CLASSES = len(CLASS_NAMES)
UNLABELED = -1

# --------------------------------------------------------------------------
# stations.csv
# --------------------------------------------------------------------------
STATION_REQUIRED = ["station_id", "x_m", "y_m", "z_m"]
STATION_OPTIONAL = {
    "site_id": "site_0",          # several monitored slopes can share one file
    "geo_unit": None,             # geological unit label -> edge similarity
    "downslope_az_deg": None,     # steepest-descent azimuth (deg, clockwise from north)
}

# --------------------------------------------------------------------------
# observations.csv
# --------------------------------------------------------------------------
KEY_COLUMNS = ["event_id", "timestamp", "station_id"]

# The 15 per-station features of Eq. (5) in the paper, in this exact order.
FEATURE_COLUMNS = [
    # IMU orientation (degrees)
    "roll_deg", "pitch_deg", "yaw_deg",
    # IMU change within the reporting interval (degrees)
    "droll_deg", "dpitch_deg", "dyaw_deg",
    # GNSS displacement w.r.t. installation epoch (metres, local ENU)
    "gps_dx_m", "gps_dy_m", "gps_dz_m",
    # GNSS displacement speed (millimetres per hour)
    "gps_vel_mm_h",
    # Rainfall (mm/h and accumulated mm)
    "rain_rate_mm_h", "rain_24h_mm", "rain_72h_mm",
    # Atmosphere
    "temp_c", "rh_pct",
]

LABEL_COLUMN = "label"
OBS_OPTIONAL = {
    "site_id": "site_0",
    "label": UNLABELED,   # empty / -1 = unknown (ignored by the loss and metrics)
    "split": None,        # train | val | test, constant within an event
    "source": None,       # free text, e.g. sim | lab | field (informational)
}

VALID_SPLITS = {"train", "val", "test"}
