#!/usr/bin/env bash

set -euo pipefail

if [[ -f /codex/use-codex.sh ]]; then
    # shellcheck disable=SC1091
    source /codex/use-codex.sh
fi

usage() {
    cat <<'EOF'
Usage: scripts/te_cluster_setup.sh [OPTIONS]

Install image-specific Transformer Engine dependencies, sync submodules, and
build an editable TE install for a JAX or PyTorch image.

Options:
  --framework auto|jax|pytorch  Framework/image to build for (default: auto).
  --arch ARCH                   CUDA arch to build, e.g. 90, 12.0, 120, sm_90.
                                Defaults to NVTE_CUDA_ARCHS or inferred GPU arch.
  --python PYTHON               Python executable to use (default: python3).
  --skip-deps                   Do not install image-specific pip dependencies.
  --skip-submodules             Do not sync/update git submodules.
  --skip-build                  Do not run the editable TE build.
  --verbose                     Add verbose pip output to the TE build.
  -h, --help                    Show this help.

Examples:
  scripts/te_cluster_setup.sh --framework jax
  scripts/te_cluster_setup.sh --framework pytorch --arch 90
  scripts/te_cluster_setup.sh --framework jax --arch 120 --verbose
EOF
}

die() {
    echo "Error: $*" >&2
    exit 1
}

log() {
    echo "==> $*"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "${script_dir}/.." rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${repo_root}" ]]; then
    repo_root="$(cd -- "${script_dir}/.." && pwd)"
fi

framework="auto"
arch_arg=""
python_bin="${PYTHON:-python3}"
install_deps=1
sync_submodules=1
build_te=1
verbose=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --framework)
            [[ $# -ge 2 ]] || die "--framework requires a value"
            framework="$2"
            shift 2
            ;;
        --framework=*)
            framework="${1#*=}"
            shift
            ;;
        --arch)
            [[ $# -ge 2 ]] || die "--arch requires a value"
            arch_arg="$2"
            shift 2
            ;;
        --arch=*)
            arch_arg="${1#*=}"
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || die "--python requires a value"
            python_bin="$2"
            shift 2
            ;;
        --python=*)
            python_bin="${1#*=}"
            shift
            ;;
        --skip-deps)
            install_deps=0
            shift
            ;;
        --skip-submodules)
            sync_submodules=0
            shift
            ;;
        --skip-build)
            build_te=0
            shift
            ;;
        --verbose)
            verbose=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

normalize_framework_name() {
    local name="${1,,}"
    case "${name}" in
        auto|jax|pytorch)
            echo "${name}"
            ;;
        torch)
            echo "pytorch"
            ;;
        *)
            die "unsupported framework '${1}'. Use auto, jax, or pytorch."
            ;;
    esac
}

detect_framework() {
    local requested
    requested="$(normalize_framework_name "${framework}")"
    if [[ "${requested}" != "auto" ]]; then
        echo "${requested}"
        return
    fi

    if [[ -n "${NVTE_FRAMEWORK:-}" ]]; then
        local env_framework="${NVTE_FRAMEWORK,,}"
        if [[ "${env_framework}" != *","* ]]; then
            case "${env_framework}" in
                jax|pytorch|torch)
                    normalize_framework_name "${env_framework}"
                    return
                    ;;
            esac
        fi
    fi

    local detected
    if ! detected="$("${python_bin}" - <<'PY'
import importlib.util

has_jax = importlib.util.find_spec("jax") is not None
has_torch = importlib.util.find_spec("torch") is not None

if has_jax and not has_torch:
    print("jax")
elif has_torch and not has_jax:
    print("pytorch")
elif has_jax and has_torch:
    print("both")
else:
    print("none")
PY
)"; then
        die "failed to run '${python_bin}' while detecting the framework"
    fi

    case "${detected}" in
        jax|pytorch)
            echo "${detected}"
            ;;
        both)
            die "both JAX and PyTorch are installed; pass --framework jax or --framework pytorch"
            ;;
        none)
            die "could not detect JAX or PyTorch; pass --framework jax or --framework pytorch"
            ;;
        *)
            die "unexpected framework detection result: ${detected}"
            ;;
    esac
}

normalize_single_arch() {
    local arch="${1,,}"
    arch="${arch// /}"
    arch="${arch#sm_}"
    arch="${arch#compute_}"

    case "${arch}" in
        7|8|9)
            arch="${arch}0"
            ;;
        10|12)
            arch="${arch}0"
            ;;
        *.*)
            arch="${arch/./}"
            ;;
    esac

    if [[ ! "${arch}" =~ ^[0-9]{2,3}$ ]]; then
        die "invalid CUDA arch '${1}'"
    fi

    echo "${arch}"
}

normalize_archs() {
    local raw="${1//,/;}"
    local result=""
    local old_ifs="${IFS}"
    local part
    local parts=()

    IFS=";"
    read -r -a parts <<< "${raw}"
    IFS="${old_ifs}"

    append_arch() {
        local item="$1"
        case ";${result};" in
            *";${item};"*) ;;
            *) result="${result:+${result};}${item}" ;;
        esac
    }

    for part in "${parts[@]}"; do
        [[ -n "${part// /}" ]] || continue
        local normalized
        normalized="$(normalize_single_arch "${part}")"
        if [[ "${normalized}" == "120" ]]; then
            append_arch "75"
            append_arch "120"
        else
            append_arch "${normalized}"
        fi
    done

    [[ -n "${result}" ]] || die "no CUDA arch value was provided"
    echo "${result}"
}

detect_arch_with_nvidia_smi() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0); if ($0 ~ /^[0-9]+(\.[0-9]+)?$/) {print; exit}}'
}

detect_arch_with_pytorch() {
    [[ "${effective_framework}" == "pytorch" ]] || return 1
    "${python_bin}" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print(f"{major}{minor}")
except Exception:
    pass
PY
}

detect_arch() {
    if [[ -n "${arch_arg}" ]]; then
        normalize_archs "${arch_arg}"
        return
    fi

    if [[ -n "${NVTE_CUDA_ARCHS:-}" ]]; then
        normalize_archs "${NVTE_CUDA_ARCHS}"
        return
    fi

    local detected=""
    detected="$(detect_arch_with_nvidia_smi || true)"
    if [[ -z "${detected}" ]]; then
        detected="$(detect_arch_with_pytorch || true)"
    fi

    [[ -n "${detected}" ]] || die "could not infer CUDA arch; pass --arch, e.g. --arch 90"
    normalize_archs "${detected}"
}

install_framework_deps() {
    local deps=()
    case "${effective_framework}" in
        jax)
            deps=(pybind11 pytest cmake==3.21.0)
            ;;
        pytorch)
            deps=(pytest)
            ;;
        *)
            die "unexpected framework '${effective_framework}'"
            ;;
    esac

    log "Installing ${effective_framework} image dependencies: ${deps[*]}"
    "${python_bin}" -m pip install "${deps[@]}"
}

sync_git_submodules() {
    log "Syncing git submodules"
    git submodule sync --recursive
    git submodule update --init --recursive
}

build_editable_te() {
    local pip_args=(install --no-build-isolation -e .)
    if [[ "${verbose}" -eq 1 ]]; then
        pip_args=(install -vvv --no-build-isolation -e .)
    fi

    log "Building editable TE install"
    log "NVTE_FRAMEWORK=${effective_framework}"
    log "NVTE_CUDA_ARCHS=${effective_archs}"
    env \
        NVTE_FRAMEWORK="${effective_framework}" \
        NVTE_CUDA_ARCHS="${effective_archs}" \
        "${python_bin}" -m pip "${pip_args[@]}"
}

cd "${repo_root}"

effective_framework="$(detect_framework)"
effective_archs="$(detect_arch)"

if [[ "${effective_archs}" == *"120"* && "${effective_archs}" == *"75"* ]]; then
    log "Using NVTE_CUDA_ARCHS=${effective_archs} for SM120 compatibility"
fi

if [[ "${install_deps}" -eq 1 ]]; then
    install_framework_deps
fi

if [[ "${sync_submodules}" -eq 1 ]]; then
    sync_git_submodules
fi

if [[ "${build_te}" -eq 1 ]]; then
    build_editable_te
fi

log "Done"
