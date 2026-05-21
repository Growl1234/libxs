# Predict Samples

Four executables demonstrating fingerprint-guided prediction:

- **predict_params** -- Parameter prediction from structured CSV
  (GPU kernel tuning, configuration databases).
- **predict_sunspots** -- Timeseries forecasting via sliding-window kNN
  (monthly sunspot numbers, 1749-present).
- **predict_earthquakes** -- Spatial prediction of earthquake magnitude
  from location and depth (USGS catalog).
- **predict_discharge** -- River discharge forecasting via sliding-window
  kNN with day-of-year seasonality (USGS NWIS daily streamflow).
## Build

    make

Or from the LIBXS root:

    make GNU=1 samples/predict
## predict_params

Train a prediction model from a CSV file and save it for later use.
Finds the optimal training fraction and polynomial order automatically.
Reports validation quality on a held-out subset.

### Usage

    ./predict_params.x [fraction] [auto|cat|interp] [-N] <csvfile> [modelfile]

    fraction   Validation split 0..1 for quality report (default: 0.8).
               The full model always trains on all entries.
    auto       Auto-detect mode per output (default).
    cat        Force categorical (kNN) for all outputs.
    interp     Force interpolation for all outputs.
    -N         Max polynomial order for final build (default: 0 = auto).
    csvfile    Delimited text file (semicolons, commas, or tabs).
               The first line may be a header (auto-skipped if non-numeric).
    modelfile  Output path for the binary model.
               Default: derived from CSV basename (e.g., data.csv -> data.bin).

### Example

    ./predict_params.x ../../samples/smm/params/tune_multiply_PVC.csv
## predict_sunspots

Timeseries forecasting using sliding-window nearest-neighbor prediction.
The recent history (window of W values) serves as input; the next H values
are predicted as output. The kNN confidence indicates whether similar
patterns were seen in training.

### Usage

    ./predict_sunspots.x <csvfile> [train_fraction]

    csvfile         Semicolon-delimited timeseries (SILSO sunspot format).
    train_fraction  Fraction of data used for training (default: 0.8).

### Example

    ./predict_sunspots.x predict_sunspots.csv 0.8

    Loaded 3328 monthly sunspot values from predict_sunspots.csv
    Window=12, Horizon=6, Train=2650, Test=666
    Built: 51 clusters, 32.0x compression, order=2
    Forecast quality (661 test windows):
      step   avg-err   max-err
      t+1      17.58     88.10
      t+2      19.48    115.00
      t+3      21.01    107.10
      t+4      21.76    114.50
      t+5      22.84    118.40
      t+6      24.26    153.20
      avg confidence: 1.000

### Data Source

Monthly mean total sunspot number from
[SILSO](https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv)
(World Data Center, Royal Observatory of Belgium).
Semicolon-delimited: year, month, decimal_year, sunspot_number,
std_dev, obs_count, marker.
"Source: WDC-SILSO, Royal Observatory of Belgium, Brussels"
## predict_earthquakes

Predict earthquake magnitude from geographic location and depth.
This is a spatial prediction problem (not timeseries): given where
an earthquake occurs, what magnitude is expected based on historical
patterns at nearby locations?

### Usage

    ./predict_earthquakes.x <usgs_csv> [train_fraction]

    usgs_csv        USGS earthquake catalog CSV (comma-delimited).
    train_fraction  Fraction of data for training (default: 0.8).

### Example

    ./predict_earthquakes.x predict_earthquakes.csv

    Loaded 19619 earthquake events from predict_earthquakes.csv
    Inputs: latitude, longitude, depth -> Output: magnitude
    Train=15695, Test=3924
    Built: 125 clusters, 83.9x compression, order=2
    Prediction quality (3924 test events):
      avg magnitude error: 0.272
      max magnitude error: 2.700
      avg confidence: 0.649

### Data Source

[USGS Earthquake Hazards Program](https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2022-04-01&endtime=2025-01-01&minmagnitude=4.5&limit=20000)
(public domain, US Government).
Comma-delimited: time, latitude, longitude, depth, mag, magType, ...
## predict_discharge

River discharge (streamflow) forecasting using sliding-window kNN
with day-of-year as an additional input dimension to capture
seasonality. Predicts the next 7 days from the previous 14 days.

### Usage

    ./predict_discharge.x <discharge_tsv> [train_fraction]

    discharge_tsv   USGS NWIS daily discharge (tab-delimited RDB format).
    train_fraction  Fraction of data for training (default: 0.8).

### Example

    ./predict_discharge.x predict_discharge.tsv

    Loaded 9135 daily discharge values from predict_discharge.tsv
    Window=14 (+day-of-year), Horizon=7, Train=7294, Test=1827
    Built: 85 clusters, 58.9x compression, order=2
    Forecast quality (1821 test windows):
      step   avg-err   max-err
      t+1      763.6   22600.0
      t+2      895.3   28800.0
      t+3     1015.9   29800.0
      t+4     1091.7   27100.0
      t+5     1162.7   27200.0
      t+6     1249.7   27100.0
      t+7     1353.2   27200.0
      avg confidence: 1.000

### Data Source

[USGS National Water Information System](https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=09380000&parameterCd=00060&startDT=2000-01-01&endDT=2025-01-01)
(public domain, US Government). Colorado River at Lees Ferry, site 09380000.
Tab-delimited RDB, comment lines start with #, data columns:
agency_cd, site_no, datetime, discharge_value, qualification_code.
## How It Works

All samples share the same prediction library (libxs_predict):

1. **predict_params**: Each CSV row is an independent (inputs, outputs)
   pair. The model learns spatial relationships in the input space
   and predicts outputs for unseen input combinations.

2. **predict_sunspots**: Each sliding window of W consecutive values
   becomes an input vector; the next H values become outputs. The
   model finds historically similar windows and predicts the
   continuation. The kNN confidence reflects how well the current
   pattern matches training history.

3. **predict_earthquakes**: Each earthquake event provides
   (lat, lon, depth) as inputs and magnitude as output. The model
   finds geographically similar past events and predicts expected
   magnitude for new locations.

4. **predict_discharge**: Combines temporal sliding-window (14 days)
   with day-of-year seasonality as an extra input dimension. The
   model captures both short-term flow dynamics and annual patterns.

The fingerprint automatically determines per-output whether polynomial
interpolation or distance-weighted kNN voting is more appropriate.
Per-output confidence scores enable the caller to gate predictions
and fall back to safe defaults when the model is uncertain.
