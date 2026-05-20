# Predict Samples

Two executables demonstrating fingerprint-guided prediction:

- **predict_params** -- Parameter prediction from structured CSV
  (GPU kernel tuning, configuration databases).
- **predict_sunspots** -- Timeseries forecasting via sliding-window kNN
  (sunspot numbers, sensor data, any univariate series).
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

Monthly mean total sunspot number from SILSO (World Data Center,
Royal Observatory of Belgium):

    https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv

Format: semicolon-delimited, columns: year, month, decimal_year,
sunspot_number, std_dev, obs_count, marker.

Data freely available for research with attribution:
"Source: WDC-SILSO, Royal Observatory of Belgium, Brussels"
## How It Works

Both samples share the same prediction library (libxs_predict):

1. **predict_params**: Each CSV row is an independent (inputs, outputs)
   pair. The model learns spatial relationships in the input space
   and predicts outputs for unseen input combinations.

2. **predict_sunspots**: Each sliding window of W consecutive values
   becomes an input vector; the next H values become outputs. The
   model finds historically similar windows and predicts the
   continuation. The kNN confidence reflects how well the current
   pattern matches training history.

The fingerprint automatically determines per-output whether polynomial
interpolation or distance-weighted kNN voting is more appropriate.
Per-output confidence scores enable the caller to gate predictions
and fall back to safe defaults when the model is uncertain.
