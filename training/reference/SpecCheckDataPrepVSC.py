# Import statements
import pandas as pd
import os
import glob
from argparse import ArgumentParser
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
import joblib
import matplotlib.pyplot as plt
import gc
import regex as re
from pyteomics import mzml
from tqdm import tqdm


tf.keras.backend.clear_session()
gc.collect()

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.optimizer.set_jit(False)
tf.config.experimental.set_synchronous_execution(True)


gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth set for GPUs")
    except RuntimeError as e:
        print(e)


def unlabeled_dataprep(input_folder, output_folder):
    # Function call
    print("Function called")

    # Step 1: Fixing output
    os.makedirs(output_folder, exist_ok=True)

    # Step 2: Read and concatenate all valid parquet files
    parquet_files = glob.glob(os.path.join(input_folder, "*.parquet"))
    print(f"Found {len(parquet_files)} parquet files.")

    df_list = []  # To store valid DataFrames
    invalid_files = []  # To keep track of invalid files

    for file in parquet_files:
        try:
            # Attempt to read each file
            temp_df = pd.read_parquet(file)
            df_list.append(temp_df)
            print(f"Successfully read: {file}")
        except Exception as e:
            # Log invalid files
            print(f"Error reading {file}: {e}")
            invalid_files.append(file)

    # Combine all valid DataFrames into one
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total rows after concatenation: {len(df)}")
    else:
        print("No valid parquet files found. Exiting.")
        exit()

    # Step 3: Drop unnecessary columns
    columns_to_drop = ["ms_level", "precursor_mz", "precursor_charge"]  # Update with actual column names to drop
    df = df.drop(columns=columns_to_drop, errors="ignore")  # Ignore missing columns

    # Step 4: Shuffle the DataFrame rows
    df = df.sample(n=750000, random_state=42, replace=False).reset_index(drop=True)
    df["Outcome"] = 0
    print("Shuffled and sampled the DataFrame rows.")

    # Step 5: Saving to unlabeled dataset
    output_path = os.path.join(output_folder, f"unlabeled.parquet")
    df.to_parquet(output_path, index=False)

    # Log invalid files
    if invalid_files:
        print("The following files were invalid and skipped:")
        for invalid_file in invalid_files:
            print(invalid_file)

    print("Finished processing all files.")
    return


def scan_extractor_psm(spectrum_id):
    identifier_cleaned = re.findall(pattern=r'\d+(?=.)', string=spectrum_id)[-1].lstrip('0')
    return identifier_cleaned


def scan_extractor_raw_data(spectrum_id):
    match = re.search(r'\.(\d+)\.\d+\.\d+$', spectrum_id)
    if match:
        return int(match.group(1))
    return None


def dataset_prep_HLA():
    # Function call
    print("Function called")
    # Get Parent directory
    directory = '/lustre1/scratch/351/vsc35132/Kuster/Analysis/Fragpipe'
    folders = os.listdir(directory)
    for folder in folders:
        HLA_analysis_folders = os.listdir(f"{directory}/{folder}")
        for HLA_analysis_folder in HLA_analysis_folders:
            tmp = os.listdir(f"{directory}/{folder}/{HLA_analysis_folder}")
            mzML_regex = re.compile(r'\.mzML$')
            psm_regex = re.compile(r'psm.tsv')

            mzML_filename = next((s for s in tmp if mzML_regex.search(s)), None)
            psm_filename = next((s for s in tmp if psm_regex.search(s)), None)

            if psm_filename is not None and mzML_filename is not None:
                # --- Read search results ---
                search_results = pd.read_csv(f"{directory}/{folder}/{HLA_analysis_folder}/{psm_filename}", sep='\t')

                # --- Filter for HLA proteins ---
                search_results_filtered = search_results[search_results.Protein.astype(str).str.startswith("TUM_HLA")]

                # --- Extract scan IDs from PSMs ---
                search_results_filtered["scanID"] = search_results_filtered["Spectrum"].apply(scan_extractor_psm)

                # --- Read mzML manually (instead of DepthCharge) ---
                mzml_file_path = f"{directory}/{folder}/{HLA_analysis_folder}/{mzML_filename}"
                headers = []
                ms_levels = []
                mz_arrays = []
                intensity_arrays = []

                with mzml.MzML(mzml_file_path) as reader:
                    for spectrum in reader:
                        headers.append(spectrum['spectrum title'])
                        ms_levels.append(spectrum['ms level'])
                        mz_arrays.append(spectrum['m/z array'])
                        intensity_arrays.append(spectrum['intensity array'])

                raw_data = pd.DataFrame({
                    'spectrum_title': headers,
                    'ms_level': ms_levels,
                    'mz_array': mz_arrays,
                    'intensity_array': intensity_arrays
                })

                # --- Extract scan IDs from raw data titles ---
                raw_data["scanID"] = raw_data["spectrum_title"].apply(scan_extractor_raw_data)

                # --- Match spectra with filtered search results ---
                scan_ids_search_results = search_results_filtered["scanID"].dropna().astype(int).tolist()
                raw_data_filtered = raw_data[raw_data["scanID"].isin(scan_ids_search_results)]
                raw_data_negatives = raw_data[~raw_data["scanID"].isin(scan_ids_search_results)]

                raw_data_filtered.to_parquet(
                    f"/lustre1/scratch/351/vsc35132/Kuster/Analysis/Positives/{HLA_analysis_folder}.parquet"
                )
                raw_data_negatives.to_parquet(
                    f"/lustre1/scratch/351/vsc35132/Kuster/Analysis/Negatives/{HLA_analysis_folder}.parquet"
                )
                print(f"Processed {HLA_analysis_folder}: {len(raw_data_filtered)} positives, {len(raw_data_negatives)} negatives.")
            else:
                print(f"{HLA_analysis_folder} has no search results or mzML file.")


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(mz, intensity, p_mz, p_charge, label):
    feature = {
        'mz': _float_feature(mz),
        'intensity': _float_feature(intensity),
        'precursor_mz': _float_feature([p_mz]),
        'precursor_charge': _float_feature([p_charge]),
        'label': _float_feature([label])
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def convert_to_tfrecords(input_dir, output_dir, shards=32):
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    
    print(f"Found {len(parquet_files)} parquet files.")
    
    # We will write to N shard files
    writers = [tf.io.TFRecordWriter(f"{output_dir}/train_{i:03d}.tfrec") for i in range(shards)]
    
    counter = 0
    for file in tqdm(parquet_files):
        try:
            df = pd.read_parquet(file)
            # Ensure columns exist
            if 'Outcome' not in df.columns: df['Outcome'] = 0.0
            
            for _, row in df.iterrows():
                # Round-robin writing to shards
                writer_idx = counter % shards
                
                # Handle arrays
                mz = row['mz_array']
                ints = row['intensity_array']
                
                # Serialize
                example = serialize_example(
                    mz, 
                    ints, 
                    float(row.get('precursor_mz', 0.0)), 
                    float(row.get('precursor_charge', 2.0)), 
                    float(row['Outcome'])
                )
                
                writers[writer_idx].write(example)
                counter += 1
                
        except Exception as e:
            print(f"Error reading {file}: {e}")

    for w in writers:
        w.close()
    print(f"Finished. Converted {counter} spectra to {shards} TFRecord shards.")

# Function
def main(command_line=None):
    # add main parser object
    parser = ArgumentParser(description="SpecCheck")
    # add sub parser object
    subparsers = parser.add_subparsers(dest="mode")
    ####################################################################################################################
    # Dataprepping
    prepping_unlabeled = subparsers.add_parser("prepping", help="Prepping the unlabeled data")
    # Adding arguments
    ## Input
    prepping_unlabeled.add_argument("-i", "--input", dest="input_directory", required=True, help='Give input folder')
    ## Model
    prepping_unlabeled.add_argument("-o", "--output", dest="output_directory", required=True, help="Give output folder")
    ####################################################################################################################
    # Subparser for fitting the model
    dataset_prep_parser = subparsers.add_parser("dataset_prep_HLA", help="prepping the HLA dataset")
    ####################################################################################################################
    tf_data_conversion = subparsers.add_parser("tf_data_conversion", help="Convert parquet files to TFRecords")
    tf_data_conversion.add_argument("-i", "--input", dest="input_directory", required=True, help='Give input folder')
    tf_data_conversion.add_argument("-o", "--output", dest="output_directory", required=True, help="Give output folder")

    # Argument parsing
    args = parser.parse_args(command_line)
    if args.mode == "prepping":
        print("Args loaded okay")
        unlabeled_dataprep(input_folder=args.input_directory,
                           output_folder=args.output_directory)
    elif args.mode == "dataset_prep_HLA":
        print("Args loaded okay")
        dataset_prep_HLA()
    elif args.mode == "tf_data_conversion":
        print("Args loaded okay")
        convert_to_tfrecords(input_dir=args.input_directory,
                             output_dir=args.output_directory,
                             shards=32)

if __name__ == "__main__":
    main()
