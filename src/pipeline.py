import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


# configuration

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

# False = use existing Raw CSV
# True = run live SBP ingestion
USE_LIVE_INGESTION = os.getenv("USE_LIVE_INGESTION", "false").lower() == "true"

# Local execution is the default. Set ENABLE_AZURE_UPLOADS=true only
# when Azure credentials and the cloud environment are available.
ENABLE_AZURE_UPLOADS = os.getenv("ENABLE_AZURE_UPLOADS", "false").lower() == "true"

# off by default so nothing breaks without ollama installed. also used
# by generate_insights.py directly, just shown here for the status line.
ENABLE_AI_SUMMARY = os.getenv("ENABLE_AI_SUMMARY", "false").lower() == "true"


# logging

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "pipeline.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# run pipeline step

def run_step(name, script):

    logger.info(f"STARTED: {name}")

    print("\n" + "=" * 60, flush=True)
    print(f"RUNNING: {name}", flush=True)
    print("=" * 60, flush=True)

    result = subprocess.run(
        [PYTHON, str(PROJECT_ROOT / script)],
        cwd=PROJECT_ROOT,
        text=True
    )

    # Stop the pipeline if the step failed
    if result.returncode != 0:

        logger.error(
            f"FAILED: {name} | "
            f"Return code: {result.returncode}"
        )

        print(
            f"\nFAILED: {name} "
            f"(return code: {result.returncode})",
            flush=True
        )

        sys.exit(result.returncode)

    logger.info(f"COMPLETED: {name}")

    print(
        f"\nCOMPLETED: {name}",
        flush=True
    )


# clean silver data

def clean_silver_data():

    silver_path = (
        PROJECT_ROOT
        / "Data"
        / "Silver"
        / "electricity_generation_silver.csv"
    )

    clean_path = (
        PROJECT_ROOT
        / "Data"
        / "Silver"
        / "electricity_generation_silver_clean.csv"
    )

    print("\n" + "=" * 60, flush=True)
    print("CLEANING SILVER DATA", flush=True)
    print("=" * 60, flush=True)

    df = pd.read_csv(silver_path)

    original_rows = len(df)

    # Remove negative generation values
    df = df[
        df["generation_gwh"] >= 0
    ].copy()

    removed_rows = original_rows - len(df)

    df.to_csv(
        clean_path,
        index=False
    )

    logger.info(
        f"DATA CLEANING | "
        f"Original rows: {original_rows} | "
        f"Clean rows: {len(df)} | "
        f"Removed: {removed_rows}"
    )

    print(
        f"Original rows: {original_rows}",
        flush=True
    )

    print(
        f"Clean rows: {len(df)}",
        flush=True
    )

    print(
        f"Removed rows: {removed_rows}",
        flush=True
    )

    print(
        f"Clean dataset: {clean_path}",
        flush=True
    )


# main pipeline

def main():

    logger.info(
        "========== PIPELINE STARTED =========="
    )

    print("\n", flush=True)
    print("=" * 60, flush=True)
    print(
        "POWER ANALYTICS PIPELINE",
        flush=True
    )
    print(
        f"Live ingestion: {'enabled' if USE_LIVE_INGESTION else 'disabled'} | "
        f"Azure publishing: {'enabled' if ENABLE_AZURE_UPLOADS else 'disabled'} | "
        f"AI summary: {'enabled' if ENABLE_AI_SUMMARY else 'disabled'}",
        flush=True
    )
    print("=" * 60, flush=True)

    try:

        # 1. ingestion

        if USE_LIVE_INGESTION:

            run_step(
                "Download Latest SBP Electricity Data",
                "src/ingestion/download_data.py"
            )

        else:

            print(
                "\n" + "=" * 60,
                flush=True
            )

            print(
                "SKIPPING LIVE SBP INGESTION",
                flush=True
            )

            print(
                "Using existing Raw dataset for pipeline test",
                flush=True
            )

            print(
                "=" * 60,
                flush=True
            )

        # 2. transformation

        run_step(
            "Transform Raw → Silver",
            "src/transformation/transform_electricity.py"
        )

        # 3. validation

        run_step(
            "Validate Silver Data",
            "src/validation/validate_data.py"
        )

        # 4. cleaning

        clean_silver_data()

        # 5. gold tables

        run_step(
            "Create Monthly Gold",
            "src/transformation/create_gold.py"
        )

        run_step(
            "Create Source Gold",
            "src/transformation/create_source_gold.py"
        )

        # 6. forecasting

        run_step(
            "Generate Energy Forecast",
            "src/forecasting/forecast.py"
        )

        # 7. intelligence

        run_step(
            "Generate Intelligence Insights",
            "src/intelligence/generate_insights.py"
        )

        # 8. optional azure publishing

        if ENABLE_AZURE_UPLOADS:

            run_step(
                "Upload Raw Data to Azure Bronze",
                "src/azure/upload_to_bronze.py"
            )

            run_step(
                "Upload Silver Data to Azure",
                "src/azure/upload_silver.py"
            )

            run_step(
                "Upload Gold Data to Azure",
                "src/azure/upload_gold.py"
            )

            run_step(
                "Upload Source Gold Data to Azure",
                "src/azure/upload_source_gold.py"
            )

        else:

            print(
                "\nAzure publishing is disabled; outputs remain local.",
                flush=True
            )

        # pipeline complete

        logger.info(
            "========== PIPELINE COMPLETED SUCCESSFULLY =========="
        )

        print(
            "\n" + "=" * 60,
            flush=True
        )

        print(
            "PIPELINE COMPLETED SUCCESSFULLY",
            flush=True
        )

        print(
            "=" * 60,
            flush=True
        )

    except Exception as e:

        logger.exception(
            f"PIPELINE FAILED: {e}"
        )

        print(
            f"\nPIPELINE FAILED: {e}",
            flush=True
        )

        raise


# entry point

if __name__ == "__main__":
    main()