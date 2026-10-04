#!/usr/bin/env bash
# Build the tested System One revision for the requested Tauri target.
set -euo pipefail
repo_root=$(cd "$(dirname "$0")/.." && pwd)
target=${1:-$(rustc -vV | sed -n 's/^host: //p')}
revision=46847e61582097979f539595d893d83d8e1d1af1
source_dir="$repo_root/.clef-local/sidecar-source"
build_dir="$repo_root/.clef-local/sidecar-build-$target"
cmake_args=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=ON -DOPENSSL_USE_STATIC_LIBS=TRUE)
case "$target" in
  aarch64-apple-darwin|x86_64-apple-darwin)
    [[ $(uname -s) == Darwin ]] || { echo 'macOS sidecars must be built on macOS' >&2; exit 1; }
    arch=arm64; [[ "$target" == x86_64-* ]] && arch=x86_64
    [[ $(uname -m) == "$arch" ]] || { echo "Build $target on a matching native Mac (OpenSSL must also match)" >&2; exit 1; }
    cmake_args+=(-DOPENSSL_ROOT_DIR="$(brew --prefix openssl@3)" -DCMAKE_OSX_ARCHITECTURES="$arch" -DCMAKE_OSX_DEPLOYMENT_TARGET=10.15 -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON)
    ;;
  x86_64-unknown-linux-gnu)
    [[ $(uname -s) == Linux ]] || { echo 'Linux sidecars must be built on Linux' >&2; exit 1; }
    cmake_args+=(-DGGML_CUDA="${ELDERWISE_BUILD_CUDA:-OFF}")
    ;;
  *) echo "Unsupported sidecar target: $target" >&2; exit 1 ;;
esac
if [[ ! -d "$source_dir/.git" ]]; then git init "$source_dir"; git -C "$source_dir" remote add origin https://github.com/ggml-org/llama.cpp.git; fi
git -C "$source_dir" fetch --depth 1 origin "$revision"
git -C "$source_dir" checkout --detach FETCH_HEAD
cmake -S "$source_dir" -B "$build_dir" "${cmake_args[@]}"
cmake --build "$build_dir" --config Release --target llama-server -j "${ELDERWISE_BUILD_JOBS:-4}"
mkdir -p "$repo_root/src-tauri/binaries"
install -m 755 "$build_dir/bin/llama-server" "$repo_root/src-tauri/binaries/llama-server-$target"
# Show linkage for review: macOS should depend only on system libraries/frameworks.
if [[ $(uname -s) == Darwin ]]; then
  linkage=$(otool -L "$repo_root/src-tauri/binaries/llama-server-$target")
  echo "$linkage"
  if echo "$linkage" | tail -n +2 | awk '{print $1}' | rg -v '^(/usr/lib/|/System/Library/)'; then
    echo 'Unexpected non-system dynamic dependency; do not package this sidecar' >&2
    exit 1
  fi
fi
