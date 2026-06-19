import numpy as np

from eclipse_ms import (
    COND_DIM,
    Config,
    bin_spectrum_numpy,
    build_cond_vector,
    preprocess,
)


def test_config_dimensions():
    assert Config.N_BINS == 3200
    assert COND_DIM == Config.MAX_CHARGE + 2 == 8


def test_bin_spectrum_shape_and_normalisation():
    mz = np.array([200.0, 500.3, 999.9, 1500.0])
    inten = np.array([10.0, 100.0, 50.0, 5.0])
    binned = bin_spectrum_numpy(mz, inten)
    assert binned.shape == (Config.N_BINS,)
    assert binned.dtype == np.float32
    assert np.isclose(binned.max(), 1.0)
    assert binned.min() >= 0.0


def test_bin_spectrum_drops_out_of_range_and_empty():
    # all peaks outside [MZ_MIN, MZ_MAX) -> all zeros
    mz = np.array([10.0, 5000.0])
    inten = np.array([1.0, 1.0])
    binned = bin_spectrum_numpy(mz, inten)
    assert binned.shape == (Config.N_BINS,)
    assert binned.sum() == 0.0


def test_bin_spectrum_top_n_peaks():
    rng = np.random.default_rng(0)
    mz = rng.uniform(Config.MZ_MIN, Config.MZ_MAX, size=500)
    inten = rng.uniform(0.1, 1.0, size=500)
    binned = bin_spectrum_numpy(mz, inten)
    # at most TOP_N_PEAKS non-zero bins (often fewer due to collisions)
    assert (binned > 0).sum() <= Config.TOP_N_PEAKS


def test_build_cond_vector():
    cond = build_cond_vector(precursor_mz=850.0, charge=2, ion_mobility=1.1)
    assert cond.shape == (COND_DIM,)
    # one-hot charge in first MAX_CHARGE entries
    assert cond[1] == 1.0 and cond[:Config.MAX_CHARGE].sum() == 1.0
    # m/z and IM normalised into [0, 1]
    assert 0.0 <= cond[-2] <= 1.0
    assert 0.0 <= cond[-1] <= 1.0


def test_build_cond_vector_clamps_charge():
    hi = build_cond_vector(800.0, charge=99, ion_mobility=1.0)
    lo = build_cond_vector(800.0, charge=0, ion_mobility=1.0)
    assert hi[Config.MAX_CHARGE - 1] == 1.0   # clamped to max
    assert lo[0] == 1.0                        # clamped to 1


def test_preprocess_returns_pair():
    mz = np.array([300.0, 600.0, 900.0])
    inten = np.array([1.0, 0.5, 0.2])
    binned, cond = preprocess(mz, inten, 700.0, 3, 1.0)
    assert binned.shape == (Config.N_BINS,)
    assert cond.shape == (COND_DIM,)
