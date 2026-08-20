from pathlib import Path

import kagglehub

download_path = kagglehub.dataset_download(
    "ravaghi/wellbore-geology-prediction-artifacts"
)
for f in Path(download_path).rglob("*.csv"):
    print(f.relative_to(download_path), f.stat().st_size / 1e6, "MB")
