#!/bin/bash
# The directory where this file and all the plotting files are located
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# Loop through every file and run those with the py extension in python
for file in $SCRIPT_DIR/*; do
  if [ ${file##*.} = "py" ]; then
    python3 ${file}
  fi
done;
