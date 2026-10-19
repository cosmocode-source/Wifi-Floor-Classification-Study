"""Run Random_Forest independently, from any working directory."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run import main

if __name__ == '__main__':
    try: main(only='Random_Forest')
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        print(f'ERROR: {e}', file=sys.stderr); sys.exit(1)
