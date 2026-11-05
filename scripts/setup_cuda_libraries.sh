#!/usr/bin/env bash

setup_cuda_libraries() {
  local python_path="$1"
  local site_packages
  local nvidia_libraries

  site_packages="$("$python_path" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  nvidia_libraries="$(find "$site_packages/nvidia" -mindepth 2 -maxdepth 2 \
    -type d -name lib -print 2>/dev/null | sort | tr '\n' ':')"
  [[ -n "$nvidia_libraries" ]] || {
    echo "Missing NVIDIA runtime libraries under $site_packages/nvidia" >&2
    return 1
  }
  [[ -d "$site_packages/torch/lib" ]] || {
    echo "Missing PyTorch libraries under $site_packages/torch/lib" >&2
    return 1
  }
  export LD_LIBRARY_PATH="${nvidia_libraries}${site_packages}/torch/lib:${LD_LIBRARY_PATH:-}"
}
