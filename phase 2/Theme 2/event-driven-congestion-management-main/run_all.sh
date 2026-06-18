#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python scripts/01_prepare_data.py --input data/cleaned_gridlock.csv
python scripts/02_build_network.py
python scripts/03_train_duration_model.py
python scripts/04_predict_impact.py
python scripts/05_manpower_optimizer.py
python scripts/06_barricade_simulator.py
python scripts/07_diversion_routes.py
python scripts/08_generate_dashboard.py
python scripts/09_mlflow_logger.py --mode log-latest

echo "Pipeline complete. Open output/dashboards/dashboard.html or run: streamlit run app/main.py"
