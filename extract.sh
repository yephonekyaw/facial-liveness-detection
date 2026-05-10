#!/usr/bin/env bash

set -uo pipefail
shopt -s nullglob

STRICT=0
LOG_FILE=""

# --- Parse args ---
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strict)
      STRICT=1
      shift
      ;;
    --log)
      LOG_FILE="$2"
      shift 2
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL[@]}"

# --- Input validation ---
if [ $# -ne 1 ]; then
  echo "Usage: $0 [--strict] [--log <file>] <relative-path>"
  exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
  echo "Error: Directory '$TARGET_DIR' does not exist."
  exit 1
fi

# --- Logging setup ---
if [ -n "$LOG_FILE" ]; then
  exec > >(tee -a "$LOG_FILE") 2>&1
  echo "Logging to $LOG_FILE"
fi

cd "$TARGET_DIR"

files=(*.tar.gz)
total=${#files[@]}
count=0

if [ $total -eq 0 ]; then
  echo "No .tar.gz files found in $TARGET_DIR"
  exit 0
fi

echo "Found $total archives in $TARGET_DIR"
echo

# --- Main loop ---
for f in "${files[@]}"; do
  ((count++))
  dir="${f%.tar.gz}"

  echo "[$count/$total] Extracting: $f -> $dir/"

  mkdir -p "$dir"

  if command -v pv >/dev/null 2>&1; then
    pv "$f" | tar -xzf - -C "$dir"
  else
    tar -xzf "$f" -C "$dir"
  fi

  status=$?

  if [ $status -eq 0 ]; then
    rm -f "$f"
    echo "Done and removed: $f"
  else
    echo "Failed: $f (not removed)"
    if [ $STRICT -eq 1 ]; then
      echo "Stopping due to --strict"
      exit 1
    fi
  fi

  echo "-----------------------------"
done

echo "All done."
