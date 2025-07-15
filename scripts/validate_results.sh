#!/bin/bash
#
# validate_results.sh - A script to run experiments and verify plot outputs.
#

set -e

# --- Configuration ---
CANONICAL_CHECKSUMS="results/canonical_checksums.sha256"
GENERATED_PLOTS_DIR="data/evaluation/plots"
BASELINE_PLOT_SCRIPT="scripts/baseline_comparison.py"

# --- Helper Functions ---
function info {
    echo "[INFO] $1"
}

function error {
    echo "[ERROR] $1" >&2
    exit 1
}

# --- Main Logic ---

# 1. Generate the plots by running the comparison script
info "Running the baseline comparison to generate evaluation plots..."
python -m ${BASELINE_PLOT_SCRIPT}

if [ $? -ne 0 ]; then
    error "Failed to run the baseline comparison script. Aborting."
fi

info "Plot generation complete."

# 2. Check if the checksums file exists
if [ ! -f "${CANONICAL_CHECKSUMS}" ]; then
    error "Canonical checksums file not found at ${CANONICAL_CHECKSUMS}. Cannot verify results."
fi

# 3. Verify the checksums of the generated plots
info "Verifying checksums of generated plots against ${CANONICAL_CHECKSUMS}..."

# Change to the plots directory to ensure sha256sum uses relative paths
cd ${GENERATED_PLOTS_DIR}

# Use the canonical checksums file to check the generated files
# The `--strict` flag ensures we exit with an error if any file fails the check
sha256sum -c --strict ../../../${CANONICAL_CHECKSUMS}

if [ $? -ne 0 ]; then
    error "Checksum validation failed. One or more plots do not match the canonical version."
fi

info "SUCCESS: All generated plots match the canonical checksums."

exit 0 