import errno
from utils import retrieve_bvalue

STUDY_UID_COL = "StudyInstanceUID"
SERIES_NO_COL = "SeriesNumber"
SERIES_SELECTION = {
    "T2": "manual_sele_t2",
    "HBV": "manual_sele_hi_b",
    "ADC": "manual_sele_ADC",
}

diffusion_bvalue = (0x0018, 0x9087)
diffusion_bvalue_ge = (0x0043, 0x1039)
diffusion_bvalue_siemens = (0x0019, 0x100C)

if __name__ == "__main__":
    import argparse
    import os
    import pandas as pd
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to DICOM files",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to metadata file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output directory",
    )

    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    data_dir = Path(args.data_dir)
    all_series = {}
    for i, row in metadata.iterrows():
        study_uid = row[STUDY_UID_COL]
        series_no = row[SERIES_NO_COL]
        for k in SERIES_SELECTION:
            if row[SERIES_SELECTION[k]] == "yes":
                series_type = k
                if study_uid not in all_series:
                    all_series[study_uid] = {}
                all_series[study_uid][series_type] = {
                    "series_no": series_no,
                    "series_type": series_type,
                }

    for study_uid in all_series:
        if len(all_series[study_uid]) == len(SERIES_SELECTION):
            study_path = next(data_dir.rglob(f"{study_uid}"))
            for k in SERIES_SELECTION:
                series_no = all_series[study_uid][k]["series_no"]
                series_type = all_series[study_uid][k]["series_type"]
                dicom_files = list(study_path.rglob(f"{series_no}/*dcm"))
                paths = [str(x) for x in dicom_files]
                if k == "HBV":
                    all_bvalues = [int(retrieve_bvalue(x)) for x in paths]
                    unique_bvalues = set(all_bvalues)
                    max_bvalue = sorted(unique_bvalues)[-1]
                    paths = [
                        x
                        for x, b in zip(paths, all_bvalues)
                        if b == max_bvalue
                    ]
                for path in paths:
                    link_path = os.path.join(
                        args.output_dir,
                        study_uid,
                        series_type,
                        os.path.basename(path),
                    )
                    Path(link_path).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.symlink(path, link_path)
                    except FileExistsError:
                        print(f"File {link_path} already exists, continuing")
