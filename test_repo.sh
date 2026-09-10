#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-check}"

if [[ "${MODE}" == "--help" || "${MODE}" == "-h" ]]; then
  cat <<'USAGE'
Usage:
  ./test_repo.sh [check|--check-only]
  ./test_repo.sh --fix
  ./test_repo.sh --clean
  ./test_repo.sh --ci

The gate uses geoPFA's dev-gblk Pixi environment and validates lint, byte
compilation, and the complete pytest suite.
USAGE
  exit 0
fi

if [[ "${MODE}" == "--clean" ]]; then
  echo ">>> Cleaning generated validation artifacts"
  rm -rf .coverage coverage.xml htmlcov .pytest_cache .ruff_cache
  find geopfa tests -type d -name __pycache__ -prune -exec rm -rf {} +
  MODE="check"
fi

if [[ -z "${GEOPFA_GATE_UNDER_PIXI:-}" ]]; then
  if ! command -v pixi >/dev/null 2>&1; then
    echo "pixi is required. Install pixi, then rerun ./test_repo.sh." >&2
    exit 127
  fi
  exec pixi run -e dev-gblk env GEOPFA_GATE_UNDER_PIXI=1 bash test_repo.sh "${MODE}"
fi

if [[ "${MODE}" == "--ci" || "${MODE}" == "--check-only" ]]; then
  MODE="check"
fi

if [[ "${MODE}" == "--fix" ]]; then
  echo ">>> Ruff check --fix"
  ruff check --fix geopfa tests
  MODE="check"
fi

if [[ "${MODE}" != "check" ]]; then
  echo "Unknown mode: ${MODE}" >&2
  exit 2
fi

echo ">>> Python version"
python -V

echo ">>> Tracked artifact size"
artifact_limit_bytes=$((10 * 1024 * 1024))
artifact_report=$(mktemp "${TMPDIR:-/tmp}/geopfa-artifacts.XXXXXX")
trap 'rm -f "${artifact_report}"' EXIT
while IFS= read -r tracked_path; do
  case "${tracked_path}" in
    data/raw/* | data/evidence_layers/* | data/labeled_wells/* | \
      examples/*/*/data/* | examples/*/*/outputs/*)
      printf '%s\n' "${tracked_path}" >> "${artifact_report}"
      ;;
  esac
done < <(git ls-files)
if [[ -s "${artifact_report}" ]]; then
  echo "Generated study data and outputs must not be tracked:" >&2
  sort "${artifact_report}" >&2
  exit 1
fi
while IFS= read -r -d '' artifact_path; do
  [[ -f "${artifact_path}" ]] || continue
  artifact_bytes=$(wc -c < "${artifact_path}")
  if ((artifact_bytes >= artifact_limit_bytes)); then
    printf '%s\t%s\n' "${artifact_bytes}" "${artifact_path}" \
      >> "${artifact_report}"
  fi
done < <(git ls-files -z)
if [[ -s "${artifact_report}" ]]; then
  echo "Tracked files must be smaller than 10 MiB:" >&2
  sort -nr "${artifact_report}" >&2
  exit 1
fi
rm -f "${artifact_report}"
trap - EXIT

echo ">>> Ruff check"
ruff check geopfa tests

echo ">>> Compile source and tests"
python -m compileall -q geopfa tests

echo ">>> Pytest"
python -m pytest \
  --durations=20 \
  -rapP \
  --cov=geopfa \
  --cov-report=html \
  --cov-branch \
  --cov-report=xml:coverage.xml \
  --cov-fail-under=20 \
  -n auto \
  tests

echo ">>> geoPFA validation complete"
