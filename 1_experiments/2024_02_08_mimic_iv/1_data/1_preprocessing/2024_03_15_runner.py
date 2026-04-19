import subprocess
import sys
from pathlib import Path


if __name__ == '__main__':

    current_dir = Path(__file__).resolve().parent
    stats_script = current_dir / "2024_02_01_overall_stats_generation.py"
    filter_script = current_dir / "2024_03_13_filter_columns.py"

    # Run stats generation
    subprocess.run([sys.executable, str(stats_script)], check=True)

    # Run actual filtering and processing
    subprocess.run([sys.executable, str(filter_script)], check=True)




