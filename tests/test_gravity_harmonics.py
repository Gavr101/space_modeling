from pathlib import Path
import shutil
import zipfile

import numpy as np

import dynamics.gravity_harmonics as gh
from dynamics.gravity_harmonics import (
    GravityHarmonicCoefficients,
    download_icgem_gfc,
    harmonic_perturbing_acceleration,
    read_icgem_gfc,
)
from dynamics.propagator import J2_EARTH, MU_EARTH, R_EARTH, _j2_acceleration_fixed_axis


def _j2_coefficients() -> GravityHarmonicCoefficients:
    c = np.zeros((3, 3), dtype=float)
    s = np.zeros_like(c)
    c[2, 0] = -J2_EARTH / np.sqrt(5.0)
    return GravityHarmonicCoefficients(mu_m3_s2=MU_EARTH, radius_m=R_EARTH, c=c, s=s)


def test_degree2_harmonic_matches_analytic_j2_acceleration() -> None:
    r_itrs = np.array([6_900_000.0, 1_100_000.0, 1_700_000.0])

    harmonic = harmonic_perturbing_acceleration(
        r_itrs,
        _j2_coefficients(),
        max_degree=2,
        max_order=0,
        finite_difference_step_m=1.0,
    )
    analytic = _j2_acceleration_fixed_axis(r_itrs)

    np.testing.assert_allclose(harmonic, analytic, rtol=5e-5, atol=1e-9)


def test_read_icgem_gfc_parses_metadata_and_degree_filter() -> None:
    path = Path("tests") / "_sample_gravity.gfc"
    path.write_text(
        "\n".join(
            [
                "earth_gravity_constant 398600441800000.0",
                "radius 6378136.3",
                "norm fully_normalized",
                "gfc 2 0 -4.84165143790815e-04 0.0 0.0 0.0",
                "gfc 3 0 9.57161207093490e-07 0.0 0.0 0.0",
            ]
        ),
        encoding="utf-8",
    )
    try:
        coefficients = read_icgem_gfc(path, max_degree=2)
    finally:
        path.unlink()

    assert coefficients.max_degree == 2
    assert coefficients.mu_m3_s2 == MU_EARTH
    assert coefficients.radius_m == R_EARTH
    assert coefficients.c[2, 0] < 0.0


def test_download_icgem_gfc_extracts_gfc_from_zip(monkeypatch) -> None:
    scratch = Path("tests") / "_gravity_download_scratch"
    scratch.mkdir(exist_ok=True)
    zip_path = scratch / "model.zip"
    try:
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.writestr("nested/sample.gfc", "earth_gravity_constant 1\nradius 1\n")

        def fake_urlretrieve(url: str, filename: str | Path):
            shutil.copyfile(zip_path, filename)
            return filename, None

        monkeypatch.setattr(gh, "urlretrieve", fake_urlretrieve)
        target = download_icgem_gfc("https://example.test/model.zip", scratch / "out.gfc")

        assert target.read_text(encoding="utf-8") == "earth_gravity_constant 1\nradius 1\n"
    finally:
        for path in scratch.glob("*"):
            path.unlink()
        scratch.rmdir()
