# Dataset format

A dataset consists of **two CSV files**:

| File | One row per | Purpose |
|---|---|---|
| `stations.csv` | sensing station | static metadata; the graph nodes and edges are built from it |
| `observations.csv` | station × timestamp | the 15 input features, the severity label, and the split |

Both files are UTF-8, comma-separated, with a header row. Column order does not
matter; column names do (case-sensitive). Extra columns are ignored, so you can
keep raw fields (battery voltage, RSSI, ...) in the same file.

Check a dataset before training:

```bash
python -m landslide_gnn validate --stations data/stations.csv --observations data/observations.csv
```

---

## 1. `stations.csv`

| Column | Type | Required | Description |
|---|---|---|---|
| `station_id` | string | **yes** | Station identifier, unique within a site. Must match `observations.csv`. |
| `x_m` | float | **yes** | Easting in a local metric frame (m). |
| `y_m` | float | **yes** | Northing in the same frame (m). |
| `z_m` | float | **yes** | Elevation (m). Drives the direction of the flow edges. |
| `site_id` | string | no | Monitored slope. Default `site_0`. Stations of different sites are never connected. |
| `geo_unit` | string | no | Geological unit at the station. Edge similarity σ = 1 if both ends share the unit, else 0. Empty everywhere → σ = 1. |
| `downslope_az_deg` | float | no | Steepest-descent azimuth at the station, degrees clockwise from north (e.g. from a DEM). Empty → estimated from a plane fit to nearby stations. |

Coordinates must be metric (UTM or a local ENU frame), not latitude/longitude.

Example:

```csv
site_id,station_id,x_m,y_m,z_m,geo_unit,downslope_az_deg
cisarua,ST01,712345.2,9234567.8,1210.4,andesite_breccia,185
cisarua,ST02,712370.9,9234561.0,1209.7,andesite_breccia,
cisarua,ST03,712350.1,9234540.3,1196.2,colluvium,190
```

---

## 2. `observations.csv`

Long format: **one row per station per reporting time**. Stations do not need to
report at exactly the same timestamps, but within an event the code builds a
common time grid from the union of all timestamps, so the data should be
**resampled to a regular interval** (e.g. 1 min or 10 min) before export.

### 2.1 Key columns

| Column | Type | Required | Description |
|---|---|---|---|
| `event_id` | string | **yes** | A continuous recording segment: a simulation scenario, a lab test, or a field period (e.g. one month, or one rainy season). Windows never cross events, and train/val/test splits are made **per event** to avoid leakage between overlapping windows. |
| `timestamp` | ISO-8601 or number | **yes** | `2026-01-24T02:30:00Z` (UTC recommended) or seconds as a plain number. |
| `station_id` | string | **yes** | Must exist in `stations.csv` for the same `site_id`. |
| `site_id` | string | no | Default `site_0`. |

### 2.2 Feature columns (all required, Eq. (5) of the paper)

Values may be empty (NaN) where a sensor failed; whole missing rows are allowed
(packet loss). Short gaps (≤ `data.max_gap` samples) are forward-filled; a
station with too little data in its window is treated as offline.

| Column | Unit | Description |
|---|---|---|
| `roll_deg` | deg | Fused roll (complementary filter) |
| `pitch_deg` | deg | Fused pitch |
| `yaw_deg` | deg | Integrated yaw (drifts; only its change matters) |
| `droll_deg` | deg | Change of roll since the previous report |
| `dpitch_deg` | deg | Change of pitch since the previous report |
| `dyaw_deg` | deg | Change of yaw since the previous report |
| `gps_dx_m` | m | GNSS displacement east w.r.t. installation epoch |
| `gps_dy_m` | m | GNSS displacement north |
| `gps_dz_m` | m | GNSS displacement up |
| `gps_vel_mm_h` | mm/h | Smoothed GNSS displacement speed |
| `rain_rate_mm_h` | mm/h | Rainfall rate at the assigned weather station |
| `rain_24h_mm` | mm | Accumulated rainfall over the previous 24 h |
| `rain_72h_mm` | mm | Accumulated rainfall over the previous 72 h |
| `temp_c` | °C | Air temperature |
| `rh_pct` | % | Relative humidity |

Weather columns are repeated on every station row (copy the reading of the
weather station assigned to that station). Features are standardised with the
mean/std of the **training** split, which are stored in the checkpoint.

### 2.3 Label and split

| Column | Type | Required | Description |
|---|---|---|---|
| `label` | int | for `train`/`evaluate` | Severity at this station and time: `0` Normal, `1` Warning, `2` Evacuate, `3` Landslide. Empty or `-1` = unknown (ignored by loss and metrics). Not needed for `predict`. |
| `split` | string | no | `train`, `val` or `test`. Must be constant within an event. If absent, events are split randomly 70/15/15 (`data.split_ratios`). |
| `source` | string | no | Free text, e.g. `sim`, `lab`, `field`. Kept for your own bookkeeping. |

The label is the class **at the decision time `timestamp`**. The model sees the
preceding `data.window` samples (default 60) and predicts the label of the last
one. The labelling rule recommended in the paper (Section VI-C):

* `3` Landslide — at or after the failure onset of the station's block
* `2` Evacuate — within `H_E` before onset
* `1` Warning — within `H_W` before onset, or staged criteria (tilt rate, rainfall) met
* `0` Normal — otherwise

Example:

```csv
site_id,event_id,timestamp,station_id,roll_deg,pitch_deg,yaw_deg,droll_deg,dpitch_deg,dyaw_deg,gps_dx_m,gps_dy_m,gps_dz_m,gps_vel_mm_h,rain_rate_mm_h,rain_24h_mm,rain_72h_mm,temp_c,rh_pct,label,split,source
cisarua,F2026-01,2026-01-23T22:00:00Z,ST01,1.21,-0.84,37.2,0.01,0.00,0.02,0.003,-0.004,-0.001,0.4,12.6,88.2,141.0,19.8,97,1,train,field
cisarua,F2026-01,2026-01-23T22:00:00Z,ST02,0.47,2.13,-101.5,0.00,0.02,-0.01,0.001,-0.002,0.000,0.2,12.6,88.2,141.0,19.8,97,0,train,field
```

Empty templates: [`examples/stations_template.csv`](../examples/stations_template.csv),
[`examples/observations_template.csv`](../examples/observations_template.csv).
A small filled example (synthetic): `examples/sample/`.

---

## 3. Practical notes

* **Window length vs. sampling.** With 10-min data and `data.window=60` the model
  sees 10 h of history. Choose the window in samples to cover the precursor time
  scale you care about.
* **Event length.** An event shorter than `data.window` samples produces no windows.
* **Class balance.** Report the label counts printed by `validate`. If Evacuate
  or Landslide samples are very rare, see the note on `loss.beta` in the README.
* **Several sites.** Put all sites in the same two files with different `site_id`
  values; each snapshot uses its own site graph. Cross-site transfer: put one
  site's events in `train`/`val` and the other's in `test`.
* **Multiple data sources.** Simulation, laboratory and field data can live in
  the same files, separated by `event_id` (and `source`). To pretrain on
  simulation and test on field data, set `split` accordingly.
