import os
import sys

spot_path = os.getenv('spot_path')
source_dir = f"{spot_path}/source/"

notebooks_dir = f"{spot_path}/notebooks/work1/"

if source_dir not in sys.path:
    sys.path.append(source_dir)

assert spot_path is not None, " Set Spot Path before invocation"
print(f"Adding {source_dir} to sys.path")
