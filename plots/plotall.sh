#!/bin/bash
for file in *; do
  if [ ${file##*.} = "py" ]; then
    python ${file}
  fi
done;
