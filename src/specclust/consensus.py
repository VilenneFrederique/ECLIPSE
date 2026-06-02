"""Consensus (averaged) spectrum generation for clusters.

The consensus algorithm is ported from the project's consensus script; the full
original (with the richer mzML writer and FragPipe-specific helpers) is kept in
``training/consensus_reference.py`` for reference.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def generate_consensus_spectrum(
    spectra: List[Dict],
    mz_tolerance: float = 0.01,
    min_peak_occurrence: float = 0.3,
) -> Optional[Dict]:
    """Average multiple spectra into a single consensus spectrum.

    Peaks within ``mz_tolerance`` (Da) are binned; a bin is kept if it occurs in
    at least ``min_peak_occurrence`` of the spectra. Each input dict must have
    ``mz_array``, ``intensity_array``, ``precursor_mz``, ``charge``,
    ``ion_mobility``.

    Returns a consensus dict, or ``None`` if nothing passes the filters.
    """
    if not spectra:
        return None
    n_spectra = len(spectra)

    all_peaks = []
    for spec in spectra:
        mz = np.asarray(spec["mz_array"], dtype=float)
        intensity = np.asarray(spec["intensity_array"], dtype=float)
        if len(intensity) > 0 and intensity.max() > 0:
            intensity = intensity / intensity.max()
        all_peaks.extend(zip(mz, intensity))

    if not all_peaks:
        return None
    all_peaks.sort(key=lambda x: x[0])

    bins: List[List] = []
    current = [all_peaks[0]]
    for peak in all_peaks[1:]:
        if peak[0] - current[0][0] <= mz_tolerance:
            current.append(peak)
        else:
            bins.append(current)
            current = [peak]
    bins.append(current)

    consensus_mz, consensus_intensity = [], []
    for bin_peaks in bins:
        occurrence = min(len(bin_peaks), n_spectra) / n_spectra
        if occurrence >= min_peak_occurrence:
            consensus_mz.append(np.mean([p[0] for p in bin_peaks]))
            consensus_intensity.append(np.mean([p[1] for p in bin_peaks]))

    if not consensus_mz:
        return None

    consensus_mz = np.array(consensus_mz)
    consensus_intensity = np.array(consensus_intensity)
    if consensus_intensity.max() > 0:
        consensus_intensity = consensus_intensity / consensus_intensity.max() * 10000

    order = np.argsort(consensus_mz)
    consensus_mz = consensus_mz[order]
    consensus_intensity = consensus_intensity[order]

    precursor_mz = float(np.mean([s["precursor_mz"] for s in spectra]))
    ims = [s["ion_mobility"] for s in spectra if s.get("ion_mobility", 0) > 0]
    ion_mobility = float(np.mean(ims)) if ims else 0.0
    charges = [
        s["charge"] for s in spectra if s.get("charge") and not np.isnan(s["charge"])
    ]
    charge = int(np.median(charges)) if charges else 2

    return {
        "mz_array": consensus_mz,
        "intensity_array": consensus_intensity,
        "precursor_mz": precursor_mz,
        "charge": charge,
        "ion_mobility": ion_mobility,
        "n_spectra": n_spectra,
        "n_peaks": len(consensus_mz),
    }


def write_mzml(consensus_spectra: List[Dict], output_path: str) -> None:
    """Write consensus spectra to a minimal, valid mzML file.

    A compact writer covering MS2 centroid spectra with a single precursor.
    For the full-featured writer (instrument metadata, etc.), see
    ``training/consensus_reference.py``.
    """
    import base64
    import struct
    from datetime import datetime, timezone

    def _b64(arr, dtype):
        packed = struct.pack(f"<{len(arr)}{'d' if dtype == 64 else 'f'}", *arr)
        return base64.b64encode(packed).decode("ascii")

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">',
        "  <run id=\"specclust_consensus\" "
        f'startTimeStamp="{datetime.now(timezone.utc).isoformat()}">',
        f'    <spectrumList count="{len(consensus_spectra)}">',
    ]
    for idx, s in enumerate(consensus_spectra):
        mz = np.asarray(s["mz_array"], dtype=np.float64)
        inten = np.asarray(s["intensity_array"], dtype=np.float32)
        mz_b64 = _b64(mz, 64)
        in_b64 = _b64(inten, 32)
        lines += [
            f'      <spectrum index="{idx}" id="scan={idx + 1}" '
            f'defaultArrayLength="{len(mz)}">',
            '        <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum"/>',
            '        <cvParam cvRef="MS" accession="MS:1000511" name="ms level" '
            'value="2"/>',
            "        <precursorList count=\"1\"><precursor><selectedIonList count=\"1\">"
            "<selectedIon>",
            '          <cvParam cvRef="MS" accession="MS:1000744" '
            f'name="selected ion m/z" value="{s["precursor_mz"]}"/>',
            '          <cvParam cvRef="MS" accession="MS:1000041" '
            f'name="charge state" value="{s["charge"]}"/>',
            "        </selectedIon></selectedIonList></precursor></precursorList>",
            '        <binaryDataArrayList count="2">',
            f'          <binaryDataArray encodedLength="{len(mz_b64)}">'
            '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000514" name="m/z array"/>'
            f"<binary>{mz_b64}</binary></binaryDataArray>",
            f'          <binaryDataArray encodedLength="{len(in_b64)}">'
            '<cvParam cvRef="MS" accession="MS:1000521" name="32-bit float"/>'
            '<cvParam cvRef="MS" accession="MS:1000515" name="intensity array"/>'
            f"<binary>{in_b64}</binary></binaryDataArray>",
            "        </binaryDataArrayList>",
            "      </spectrum>",
        ]
    lines += ["    </spectrumList>", "  </run>", "</mzML>"]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(consensus_spectra)} consensus spectra to {output_path}")
