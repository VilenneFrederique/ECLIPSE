"""
Consensus Spectrum Generator for Cluster Analysis
==================================================

Generates consensus (averaged) spectra from clustered MS/MS data.
Outputs in mzML format for downstream de novo sequencing.

Usage:
    python generate_consensus_spectra.py \
        --parquet /path/to/parquet/dir \
        --clustering /path/to/clustering/dir \
        --output /path/to/output.mzML \
        --min-size 5 \
        --max-mz-std 0.5 \
        --min-files 2

Author: Generated for Frédérique's PhD project
Date: January 2026
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import xml.etree.ElementTree as ET
from datetime import datetime

# pyteomics is optional - we write mzML manually
try:
    from pyteomics import mzml
    HAS_PYTEOMICS = True
except ImportError:
    HAS_PYTEOMICS = False


def load_cluster_data(clustering_dir: str) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load clustering results."""
    labels = np.load(os.path.join(clustering_dir, 'cluster_labels.npy'))
    metadata = pd.read_parquet(os.path.join(clustering_dir, 'metadata.parquet'))
    
    scores_file = os.path.join(clustering_dir, 'cluster_scores.csv')
    if os.path.exists(scores_file):
        scores = pd.read_csv(scores_file)
    else:
        scores = None
    
    return labels, metadata, scores


def filter_clusters(scores: pd.DataFrame, 
                    min_size: int = 5,
                    max_mz_std: float = 0.5,
                    min_files: int = 2,
                    exclude_validated: bool = True,
                    validation_file: Optional[str] = None) -> List[int]:
    """
    Filter clusters based on quality criteria.
    
    Args:
        scores: Cluster scores DataFrame
        min_size: Minimum cluster size
        max_mz_std: Maximum precursor m/z standard deviation (Da)
        min_files: Minimum number of files cluster appears in
        exclude_validated: If True, exclude clusters that have peptide labels
        validation_file: Path to validation_results.csv to identify validated clusters
    
    Returns:
        List of cluster IDs passing filters
    """
    filtered = scores[
        (scores['size'] >= min_size) &
        (scores['mz_std'] <= max_mz_std) &
        (scores['n_files'] >= min_files)
    ].copy()
    
    if exclude_validated and validation_file and os.path.exists(validation_file):
        validation = pd.read_csv(validation_file)
        validated_ids = set(validation['cluster_id'])
        filtered = filtered[~filtered['cluster_id'].isin(validated_ids)]
        print(f"  Excluded {len(validated_ids)} validated clusters")
    
    return filtered['cluster_id'].tolist()


def load_spectra_for_cluster(parquet_files: List[str], 
                              metadata: pd.DataFrame,
                              labels: np.ndarray,
                              cluster_id: int) -> List[Dict]:
    """
    Load all spectra belonging to a specific cluster.
    
    Returns list of dicts with mz_array, intensity_array, precursor_mz, charge, etc.
    """
    # Get indices for this cluster
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    cluster_meta = metadata.iloc[cluster_indices]
    
    # Group by file
    spectra_by_file = defaultdict(list)
    for idx, row in cluster_meta.iterrows():
        spectra_by_file[row['file']].append({
            'scan': row.get('scan'),
            'precursor_mz': row['precursor_mz'],
            'charge': row['charge'],
            'ion_mobility': row.get('ion_mobility', 0),
        })
    
    # Load spectra from parquet files
    spectra = []
    for pf in parquet_files:
        filename = os.path.basename(pf)
        if filename not in spectra_by_file:
            continue
        
        df = pd.read_parquet(pf)
        
        for spec_info in spectra_by_file[filename]:
            # Find matching spectrum by scan or precursor_mz
            if spec_info['scan'] is not None:
                match = df[df['scanID'] == spec_info['scan']]
            else:
                # Fallback to precursor_mz matching
                match = df[np.abs(df['precursor_mz'] - spec_info['precursor_mz']) < 0.01]
            
            if len(match) > 0:
                row = match.iloc[0]
                spectra.append({
                    'mz_array': np.array(row['mz_array'], dtype=np.float64),
                    'intensity_array': np.array(row['intensity_array'], dtype=np.float64),
                    'precursor_mz': float(row['precursor_mz']),
                    'charge': int(row['precursor_charge']) if pd.notna(row.get('precursor_charge')) else int(spec_info['charge']),
                    'ion_mobility': float(row.get('IM', spec_info['ion_mobility'])),
                    'file': filename,
                    'scan': row.get('scanID'),
                })
    
    return spectra


def generate_consensus_spectrum(spectra: List[Dict], 
                                 mz_tolerance: float = 0.01,
                                 min_peak_occurrence: float = 0.3) -> Dict:
    """
    Generate a consensus spectrum from multiple spectra.
    
    Algorithm:
    1. Bin all peaks across spectra by m/z (with tolerance)
    2. For each bin, compute average m/z and average intensity
    3. Keep peaks that appear in >= min_peak_occurrence fraction of spectra
    4. Normalize intensities
    
    Args:
        spectra: List of spectrum dicts
        mz_tolerance: Tolerance for binning peaks (Da)
        min_peak_occurrence: Minimum fraction of spectra a peak must appear in
    
    Returns:
        Consensus spectrum dict
    """
    if len(spectra) == 0:
        return None
    
    n_spectra = len(spectra)
    
    # Collect all peaks
    all_peaks = []
    for spec in spectra:
        mz = spec['mz_array']
        intensity = spec['intensity_array']
        
        # Normalize intensity within spectrum
        if len(intensity) > 0 and intensity.max() > 0:
            intensity = intensity / intensity.max()
        
        for m, i in zip(mz, intensity):
            all_peaks.append((m, i))
    
    if len(all_peaks) == 0:
        return None
    
    # Sort by m/z
    all_peaks.sort(key=lambda x: x[0])
    
    # Bin peaks by m/z
    bins = []
    current_bin = [all_peaks[0]]
    
    for peak in all_peaks[1:]:
        if peak[0] - current_bin[0][0] <= mz_tolerance:
            current_bin.append(peak)
        else:
            bins.append(current_bin)
            current_bin = [peak]
    bins.append(current_bin)
    
    # Compute consensus peaks
    consensus_mz = []
    consensus_intensity = []
    
    for bin_peaks in bins:
        # Count how many spectra contributed to this bin
        # (approximate: assume each spectrum contributes at most once per bin)
        occurrence = min(len(bin_peaks), n_spectra) / n_spectra
        
        if occurrence >= min_peak_occurrence:
            # Average m/z and intensity
            avg_mz = np.mean([p[0] for p in bin_peaks])
            avg_intensity = np.mean([p[1] for p in bin_peaks])
            
            consensus_mz.append(avg_mz)
            consensus_intensity.append(avg_intensity)
    
    if len(consensus_mz) == 0:
        return None
    
    consensus_mz = np.array(consensus_mz)
    consensus_intensity = np.array(consensus_intensity)
    
    # Normalize to max = 10000 (standard for mzML)
    if consensus_intensity.max() > 0:
        consensus_intensity = consensus_intensity / consensus_intensity.max() * 10000
    
    # Sort by m/z
    sort_idx = np.argsort(consensus_mz)
    consensus_mz = consensus_mz[sort_idx]
    consensus_intensity = consensus_intensity[sort_idx]
    
    # Compute consensus precursor info
    precursor_mz = np.mean([s['precursor_mz'] for s in spectra])
    # charge = int(np.median([s['charge'] for s in spectra]))
    ion_mobility = np.mean([s['ion_mobility'] for s in spectra if s['ion_mobility'] > 0])

    charges = [s['charge'] for s in spectra if s.get('charge') and not np.isnan(s['charge'])]
    if charges:
        charge = int(np.median(charges))
    else:
        charge = 2
    
    return {
        'mz_array': consensus_mz,
        'intensity_array': consensus_intensity,
        'precursor_mz': precursor_mz,
        'charge': charge,
        'ion_mobility': ion_mobility,
        'n_spectra': n_spectra,
        'n_peaks': len(consensus_mz),
    }


def write_mzml(consensus_spectra: List[Dict], 
               cluster_ids: List[int],
               output_path: str,
               cluster_scores: pd.DataFrame = None):
    """
    Write consensus spectra to mzML format.
    
    Uses pyteomics for writing mzML.
    """
    print(f"\nWriting {len(consensus_spectra)} consensus spectra to {output_path}")
    
    # Build spectrum list for mzML
    spectra_for_mzml = []
    
    for i, (consensus, cluster_id) in enumerate(zip(consensus_spectra, cluster_ids)):
        if consensus is None:
            continue
        
        # Get cluster info if available
        cluster_info = ""
        if cluster_scores is not None:
            row = cluster_scores[cluster_scores['cluster_id'] == cluster_id]
            if len(row) > 0:
                row = row.iloc[0]
                cluster_info = f"mz_std={row['mz_std']:.4f}_im_std={row['im_std']:.4f}_files={row['n_files']}"
        
        spectrum = {
            'id': f"cluster_{cluster_id}",
            'index': i,
            'ms level': 2,
            'm/z array': consensus['mz_array'],
            'intensity array': consensus['intensity_array'],
            'precursorList': {
                'count': 1,
                'precursor': [{
                    'selectedIonList': {
                        'count': 1,
                        'selectedIon': [{
                            'selected ion m/z': consensus['precursor_mz'],
                            'charge state': consensus['charge'],
                        }]
                    },
                    'activation': {
                        'collision energy': 25.0,  # Placeholder
                    }
                }]
            },
            'scanList': {
                'count': 1,
                'scan': [{
                    'scan start time': float(i),  # Placeholder
                    'ion injection time': 0.0,
                }]
            },
            # Custom params for cluster info
            'userParam': [
                {'name': 'cluster_id', 'value': str(cluster_id)},
                {'name': 'n_spectra', 'value': str(consensus['n_spectra'])},
                {'name': 'n_peaks', 'value': str(consensus['n_peaks'])},
                {'name': 'ion_mobility', 'value': f"{consensus['ion_mobility']:.4f}"},
            ]
        }
        
        if cluster_info:
            spectrum['userParam'].append({'name': 'cluster_info', 'value': cluster_info})
        
        spectra_for_mzml.append(spectrum)
    
    # Write using pyteomics
    # Note: pyteomics mzml.write requires specific format
    # We'll write a simplified mzML manually for better control
    
    write_mzml_manual(spectra_for_mzml, output_path)
    
    print(f"  Written {len(spectra_for_mzml)} spectra")


def write_mzml_manual(spectra: List[Dict], output_path: str):
    """
    Write mzML file manually for full control over format.
    
    Creates a valid mzML 1.1.0 file compatible with most tools.
    """
    import zlib
    import base64
    import struct
    
    def encode_array(arr: np.ndarray, precision: int = 64) -> str:
        """Encode numpy array to base64 compressed string."""
        if precision == 64:
            packed = struct.pack(f'<{len(arr)}d', *arr)
        else:
            packed = struct.pack(f'<{len(arr)}f', *arr.astype(np.float32))
        
        compressed = zlib.compress(packed)
        return base64.b64encode(compressed).decode('ascii')
    
    # XML namespaces
    ns = {
        'default': 'http://psi.hupo.org/ms/mzml',
        'xsi': 'http://www.w3.org/2001/XMLSchema-instance',
    }
    
    # Build XML
    root = ET.Element('mzML', {
        'xmlns': ns['default'],
        'xmlns:xsi': ns['xsi'],
        'xsi:schemaLocation': 'http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd',
        'version': '1.1.0',
    })
    
    # cvList
    cv_list = ET.SubElement(root, 'cvList', {'count': '2'})
    ET.SubElement(cv_list, 'cv', {
        'id': 'MS',
        'fullName': 'Proteomics Standards Initiative Mass Spectrometry Ontology',
        'version': '4.1.0',
        'URI': 'https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo'
    })
    ET.SubElement(cv_list, 'cv', {
        'id': 'UO',
        'fullName': 'Unit Ontology',
        'version': 'releases/2020-03-10',
        'URI': 'http://obo.cvs.sourceforge.net/*checkout*/obo/obo/ontology/phenotype/unit.obo'
    })
    
    # fileDescription
    file_desc = ET.SubElement(root, 'fileDescription')
    file_content = ET.SubElement(file_desc, 'fileContent')
    ET.SubElement(file_content, 'cvParam', {
        'cvRef': 'MS', 'accession': 'MS:1000580', 'name': 'MSn spectrum', 'value': ''
    })
    
    # softwareList
    soft_list = ET.SubElement(root, 'softwareList', {'count': '1'})
    software = ET.SubElement(soft_list, 'software', {'id': 'consensus_generator', 'version': '1.0'})
    ET.SubElement(software, 'cvParam', {
        'cvRef': 'MS', 'accession': 'MS:1000799', 'name': 'custom unreleased software tool', 'value': ''
    })
    
    # instrumentConfigurationList
    inst_list = ET.SubElement(root, 'instrumentConfigurationList', {'count': '1'})
    inst_config = ET.SubElement(inst_list, 'instrumentConfiguration', {'id': 'IC1'})
    ET.SubElement(inst_config, 'cvParam', {
        'cvRef': 'MS', 'accession': 'MS:1000031', 'name': 'instrument model', 'value': ''
    })
    
    # dataProcessingList
    dp_list = ET.SubElement(root, 'dataProcessingList', {'count': '1'})
    dp = ET.SubElement(dp_list, 'dataProcessing', {'id': 'consensus_processing'})
    pm = ET.SubElement(dp, 'processingMethod', {'order': '1', 'softwareRef': 'consensus_generator'})
    ET.SubElement(pm, 'cvParam', {
        'cvRef': 'MS', 'accession': 'MS:1000544', 'name': 'Conversion to mzML', 'value': ''
    })
    
    # run
    run = ET.SubElement(root, 'run', {
        'id': 'consensus_spectra',
        'defaultInstrumentConfigurationRef': 'IC1',
    })
    
    # spectrumList
    spec_list = ET.SubElement(run, 'spectrumList', {
        'count': str(len(spectra)),
        'defaultDataProcessingRef': 'consensus_processing'
    })
    
    for spec in spectra:
        spectrum = ET.SubElement(spec_list, 'spectrum', {
            'id': spec['id'],
            'index': str(spec['index']),
            'defaultArrayLength': str(len(spec['m/z array'])),
        })
        
        # CV params
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000511', 'name': 'ms level', 'value': '2'
        })
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000580', 'name': 'MSn spectrum', 'value': ''
        })
        ET.SubElement(spectrum, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000130', 'name': 'positive scan', 'value': ''
        })
        
        # User params for cluster info
        for param in spec.get('userParam', []):
            ET.SubElement(spectrum, 'userParam', {
                'name': param['name'],
                'value': param['value']
            })
        
        # scanList
        scan_list = ET.SubElement(spectrum, 'scanList', {'count': '1'})
        ET.SubElement(scan_list, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000795', 'name': 'no combination', 'value': ''
        })
        scan = ET.SubElement(scan_list, 'scan')
        ET.SubElement(scan, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000016', 'name': 'scan start time',
            'value': str(spec['scanList']['scan'][0]['scan start time']),
            'unitCvRef': 'UO', 'unitAccession': 'UO:0000010', 'unitName': 'second'
        })
        
        # precursorList
        prec_list = ET.SubElement(spectrum, 'precursorList', {'count': '1'})
        precursor = ET.SubElement(prec_list, 'precursor')
        
        sel_ion_list = ET.SubElement(precursor, 'selectedIonList', {'count': '1'})
        sel_ion = ET.SubElement(sel_ion_list, 'selectedIon')
        
        prec_mz = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['selected ion m/z']
        charge = spec['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]['charge state']
        
        ET.SubElement(sel_ion, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000744', 'name': 'selected ion m/z',
            'value': f'{prec_mz:.6f}',
            'unitCvRef': 'MS', 'unitAccession': 'MS:1000040', 'unitName': 'm/z'
        })
        ET.SubElement(sel_ion, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000041', 'name': 'charge state',
            'value': str(charge)
        })
        
        activation = ET.SubElement(precursor, 'activation')
        ET.SubElement(activation, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000422', 'name': 'beam-type collision-induced dissociation', 'value': ''
        })
        
        # binaryDataArrayList
        bin_list = ET.SubElement(spectrum, 'binaryDataArrayList', {'count': '2'})
        
        # m/z array
        mz_arr = ET.SubElement(bin_list, 'binaryDataArray', {
            'encodedLength': str(len(encode_array(spec['m/z array'])))
        })
        ET.SubElement(mz_arr, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000523', 'name': '64-bit float', 'value': ''
        })
        ET.SubElement(mz_arr, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000574', 'name': 'zlib compression', 'value': ''
        })
        ET.SubElement(mz_arr, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000514', 'name': 'm/z array',
            'unitCvRef': 'MS', 'unitAccession': 'MS:1000040', 'unitName': 'm/z', 'value': ''
        })
        mz_binary = ET.SubElement(mz_arr, 'binary')
        mz_binary.text = encode_array(spec['m/z array'])
        
        # intensity array
        int_arr = ET.SubElement(bin_list, 'binaryDataArray', {
            'encodedLength': str(len(encode_array(spec['intensity array'])))
        })
        ET.SubElement(int_arr, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000523', 'name': '64-bit float', 'value': ''
        })
        ET.SubElement(int_arr, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000574', 'name': 'zlib compression', 'value': ''
        })
        ET.SubElement(int_arr, 'cvParam', {
            'cvRef': 'MS', 'accession': 'MS:1000515', 'name': 'intensity array',
            'unitCvRef': 'MS', 'unitAccession': 'MS:1000131', 'unitName': 'number of detector counts', 'value': ''
        })
        int_binary = ET.SubElement(int_arr, 'binary')
        int_binary.text = encode_array(spec['intensity array'])
    
    # Write to file
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    
    with open(output_path, 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        tree.write(f, encoding='unicode' if isinstance(f, type(None)) else None)
    
    # Re-write with proper encoding
    with open(output_path, 'r') as f:
        content = f.read()
    with open(output_path, 'w', encoding='utf-8') as f:
        if not content.startswith('<?xml'):
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description='Generate consensus spectra from clustered MS/MS data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate consensus for all high-confidence novel candidates
    python generate_consensus_spectra.py \\
        --parquet /path/to/parquet \\
        --clustering /path/to/clustering \\
        --output consensus_novel.mzML \\
        --max-mz-std 0.5 \\
        --novel-only --validation /path/to/validation_results.csv
    
    # Generate consensus for specific clusters
    python generate_consensus_spectra.py \\
        --parquet /path/to/parquet \\
        --clustering /path/to/clustering \\
        --output consensus_selected.mzML \\
        --clusters 3251 2072 2027 1511
        """
    )
    
    parser.add_argument('--parquet', required=True, help='Directory containing parquet files')
    parser.add_argument('--clustering', required=True, help='Clustering output directory')
    parser.add_argument('--output', required=True, help='Output mzML file path')
    
    # Filtering options
    parser.add_argument('--min-size', type=int, default=5, help='Minimum cluster size (default: 5)')
    parser.add_argument('--max-mz-std', type=float, default=0.5, help='Maximum m/z std dev in Da (default: 0.5)')
    parser.add_argument('--min-files', type=int, default=1, help='Minimum number of files (default: 2)')
    parser.add_argument('--novel-only', action='store_true', help='Only include clusters without peptide labels')
    parser.add_argument('--validation', type=str, help='Path to validation_results.csv (for --novel-only)')
    parser.add_argument('--clusters', nargs='+', type=int, help='Specific cluster IDs to process')
    
    # Consensus options
    parser.add_argument('--mz-tolerance', type=float, default=0.01, help='m/z tolerance for peak binning (default: 0.01)')
    parser.add_argument('--min-peak-occurrence', type=float, default=0.3, help='Min fraction of spectra for peak inclusion (default: 0.3)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Consensus Spectrum Generator")
    print("=" * 60)
    
    # Load clustering data
    print("\nLoading clustering data...")
    labels, metadata, scores = load_cluster_data(args.clustering)
    print(f"  Total spectra: {len(labels):,}")
    print(f"  Clusters: {len(set(labels)) - 1}")
    
    # Determine which clusters to process
    if args.clusters:
        cluster_ids = args.clusters
        print(f"\nProcessing {len(cluster_ids)} specified clusters")
    else:
        print("\nFiltering clusters...")
        print(f"  min_size: {args.min_size}")
        print(f"  max_mz_std: {args.max_mz_std}")
        print(f"  min_files: {args.min_files}")
        print(f"  novel_only: {args.novel_only}")
        
        cluster_ids = filter_clusters(
            scores,
            min_size=args.min_size,
            max_mz_std=args.max_mz_std,
            min_files=args.min_files,
            exclude_validated=args.novel_only,
            validation_file=args.validation
        )
        print(f"  Clusters passing filters: {len(cluster_ids)}")
    
    if len(cluster_ids) == 0:
        print("\nNo clusters to process!")
        return
    
    # Find parquet files
    parquet_files = sorted(glob.glob(os.path.join(args.parquet, "*.parquet")))
    print(f"\nFound {len(parquet_files)} parquet files")
    
    # Generate consensus spectra
    print(f"\nGenerating consensus spectra for {len(cluster_ids)} clusters...")
    consensus_spectra = []
    processed_ids = []
    
    for i, cluster_id in enumerate(cluster_ids):
        if (i + 1) % 50 == 0:
            print(f"  Processing cluster {i+1}/{len(cluster_ids)}...")
        
        # Load spectra for this cluster
        spectra = load_spectra_for_cluster(parquet_files, metadata, labels, cluster_id)
        
        if len(spectra) < 2:
            continue
        
        # Generate consensus
        consensus = generate_consensus_spectrum(
            spectra,
            mz_tolerance=args.mz_tolerance,
            min_peak_occurrence=args.min_peak_occurrence
        )
        
        if consensus is not None and consensus['n_peaks'] >= 5:
            consensus_spectra.append(consensus)
            processed_ids.append(cluster_id)
        else:
            print(f"  Skipping cluster {cluster_id} (n_spectra={len(spectra)}, n_peaks={consensus['n_peaks'] if consensus else 0})")
    
    print(f"\nGenerated {len(consensus_spectra)} consensus spectra")
    
    # Write mzML
    write_mzml(consensus_spectra, processed_ids, args.output, scores)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Input clusters: {len(cluster_ids)}")
    print(f"Output spectra: {len(consensus_spectra)}")
    print(f"Output file: {args.output}")
    
    # Save cluster mapping
    mapping_file = args.output.replace('.mzML', '_cluster_mapping.csv')
    mapping_df = pd.DataFrame({
        'spectrum_index': range(len(processed_ids)),
        'cluster_id': processed_ids,
    })
    
    if scores is not None:
        mapping_df = mapping_df.merge(
            scores[['cluster_id', 'size', 'mz_mean', 'mz_std', 'im_mean', 'n_files']],
            on='cluster_id',
            how='left'
        )
    
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Cluster mapping: {mapping_file}")


if __name__ == '__main__':
    main()