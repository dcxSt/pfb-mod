#!/bin/bash
# Loop through every file and run those with the py extension in python
for file in *; do
  if [ ${file##*.} = "py" ]; then
    python ${file}
  fi
done;
