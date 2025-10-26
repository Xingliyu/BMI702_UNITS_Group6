import wfdb
import numpy as np
import os

input_path = "dataset/physionet.org/files/apnea-ecg/1.0.0/"
output_path = "dataset/ApneaECG/"
forecast_output_path = "dataset/ApneaECG_Forecast/"

############# Classification #############
# learning set
learning_set = (
    [f'a{i:02d}' for i in range(1, 21)] + 
    [f'b{i:02d}' for i in range(1, 6)] + 
    [f'c{i:02d}' for i in range(1, 11)]
)

# check if output directory exists, if not create it
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Open the output .ts file
with open(output_path + 'ApneaECG_TRAIN.ts', 'w') as f:
    f.write('@problemName ApneaECG\n')
    f.write('@timeStamps false\n')
    f.write('@missing false\n')
    f.write('@univariate true\n')
    f.write('@equalLength true\n')
    f.write('@seriesLength 6000\n')
    f.write('@classLabel true A N\n')
    f.write('@data\n')

    # Process each record in the learning set
    for record_name in learning_set:
        # Read the ECG signal
        record = wfdb.rdrecord(input_path + record_name)
        # Read the apnea annotations
        annotation = wfdb.rdann(input_path + record_name, 'apn')

        # Extract the ECG signal (single-channel)
        signal = record.p_signal[:, 0]

        # Get annotation symbols ('A' or 'N')
        ann_symbols = annotation.symbol

        # Samples per minute (60 seconds * 100 Hz)
        samples_per_minute = 6000

        # Extract segments based on annotation positions
        for i, symbol in enumerate(ann_symbols):
            start_sample = i * samples_per_minute
            end_sample = start_sample + samples_per_minute
            # Skip incomplete segments at the end
            if end_sample > len(signal):
                break
            segment = signal[start_sample:end_sample]
            # Convert segment to comma-separated string
            segment_str = ','.join(map(str, segment))
            # Write the segment and its label
            f.write(f'{segment_str}:{symbol}\n')

print("'ApneaECG_TRAIN.ts' has been generated.")


# Test Set
learning_set = (
    [f'x{i:02d}' for i in range(1, 36)]
)

# Open the output .ts file
with open(output_path + 'ApneaECG_TEST.ts', 'w') as f:
    # Write the header
    f.write('@problemName ApneaECG\n')
    f.write('@timeStamps false\n')
    f.write('@missing false\n')
    f.write('@univariate true\n')
    f.write('@equalLength true\n')
    f.write('@seriesLength 6000\n')
    f.write('@classLabel true A N\n')
    f.write('@data\n')

    # Process each record in the learning set
    for record_name in learning_set:
        # Read the ECG signal
        record = wfdb.rdrecord(input_path + record_name)
        # Read the apnea annotations
        annotation = wfdb.rdann(input_path + record_name, 'apn')

        # Extract the ECG signal
        signal = record.p_signal[:, 0]

        # Get annotation symbols ('A' or 'N')
        ann_symbols = annotation.symbol

        # Samples per minute (60 seconds * 100 Hz)
        samples_per_minute = 6000

        # Extract segments based on annotation positions
        for i, symbol in enumerate(ann_symbols):
            start_sample = i * samples_per_minute
            end_sample = start_sample + samples_per_minute
            # Skip incomplete segments at the end
            if end_sample > len(signal):
                break
            segment = signal[start_sample:end_sample]
            # Convert segment to comma-separated string
            segment_str = ','.join(map(str, segment))
            # Write the segment and its label
            f.write(f'{segment_str}:{symbol}\n')

print("'ApneaECG_TEST.ts' has been generated.")



############# Forecasting and Imputation #############
from datetime import timedelta
import pandas as pd

# check if output directory exists, if not create it
if not os.path.exists(forecast_output_path):
    os.makedirs(forecast_output_path)

def convert_to_csv(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Find all record names
    record_names = [f[:-4] for f in os.listdir(input_dir) if f.endswith('.dat')]
    for rec in record_names:
        path = os.path.join(input_dir, rec)
        record = wfdb.rdrecord(path)
        signal = record.p_signal[:, 0]
        fs = int(record.fs)
        n_samples = len(signal)
        # Create datetime index starting at a fixed epoch
        start = pd.Timestamp('2000-01-01')
        interval = int(1000 / fs) 
        times = pd.date_range(start=start, periods=n_samples, freq=f"{interval}ms")
        df = pd.DataFrame({'datetime': times, 'ECG': signal})
        out_file = os.path.join(output_dir, f"{rec}.csv")
        df.to_csv(out_file, index=False)

convert_to_csv(input_path, forecast_output_path)