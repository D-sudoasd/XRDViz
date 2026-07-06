from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from xrdviz.calibration import auto_calibrate_phases, build_publication_state
from xrdviz.plot.renderer import export_project
from xrdviz.publication import export_publication_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export the previous four XRD samples with the current publication palette.")
    parser.add_argument("--data-root", default="F:/3_data_analysis", help="Directory containing CIF files and Integrated_profiles.")
    parser.add_argument("--out", default=str(ROOT / "demo_outputs" / "four_samples_new_palette"), help="Output directory.")
    parser.add_argument("--check", action="store_true", help="Only check that all required input files exist.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out)
    jobs = _jobs(data_root)
    missing = _missing_paths(jobs)
    if missing:
        print("Missing required files:")
        for path in missing:
            print(f"  {path}")
        return 2

    if args.check:
        print("All required files are available.")
        return 0

    for title, spectra, phases, output_name in jobs:
        state = build_publication_state(title=title, spectra=spectra, phase_paths=phases)
        auto_calibrate_phases(state)
        target = out_root / output_name
        target.mkdir(parents=True, exist_ok=True)
        for suffix in ("png", "svg", "pdf"):
            export_project(state, target / f"{output_name}.{suffix}")
        export_publication_bundle(state, target, figure_name=f"{output_name}_bundle.pdf")
        print(f"Exported {title}: {target}")
    return 0


def _jobs(data_root: Path):
    integrated = data_root / "Integrated_profiles"
    nb_doped_spectra = [
        ("H-free", integrated / "Nb_doped_H_free_Tensile_to_fracture" / "Full" / "Az_Full_000000.txt"),
        ("H-charged", integrated / "Nb_doped_H_doped_Tensile_to_fracture" / "Full" / "Az_Full_000000.txt"),
    ]
    nb_free_spectra = [
        ("H-free", integrated / "Nb_free_H_free_Tensile_to_fracture" / "Full" / "Az_Full_000000.txt"),
        ("H-charged", integrated / "Nb_free_H_doped_Tensile_to_fracture" / "Full" / "Az_Full_000000.txt"),
    ]
    b2 = ("B2", data_root / "B2.cif")
    fcc = ("FCC", data_root / "FCC.cif")
    laves = ("Laves", data_root / "Laves.cif")
    return [
        ("Nb-doped", nb_doped_spectra, [b2, fcc, laves], "nb_doped_three_phase_new_palette"),
        ("Nb-free", nb_free_spectra, [b2, fcc], "nb_free_two_phase_new_palette"),
    ]


def _missing_paths(jobs) -> list[Path]:
    paths: list[Path] = []
    for _title, spectra, phases, _output_name in jobs:
        paths.extend(Path(path) for _label, path in spectra)
        paths.extend(Path(path) for _label, path in phases)
    return sorted({path for path in paths if not path.exists()})


if __name__ == "__main__":
    raise SystemExit(main())
