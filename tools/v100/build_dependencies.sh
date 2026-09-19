#!/usr/bin/env bash
set -euo pipefail

# Private dependencies for CUDA 12.x hosts whose distribution predates FFmpeg 6.
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
deps_dir=${NINFER_DEPS_DIR:-"${repo_dir}/build/_deps"}
prefix="${deps_dir}/install"
sources="${deps_dir}/sources"
mkdir -p "${prefix}" "${sources}"

fetch_source() {
    local name=$1
    local url=$2
    if [[ ! -d "${sources}/${name}" ]]; then
        curl --fail --location --retry 3 --output "${sources}/${name}.tar.xz" "${url}"
        tar -xf "${sources}/${name}.tar.xz" -C "${sources}"
    fi
}

fetch_source nasm-2.16.03 https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/nasm-2.16.03.tar.xz
if [[ ! -x "${prefix}/bin/nasm" ]]; then
    (
        cd "${sources}/nasm-2.16.03"
        ./configure --prefix="${prefix}"
        make -j
        make install
    )
fi
export PATH="${prefix}/bin:${PATH}"

fetch_source ffmpeg-6.1.2 https://ffmpeg.org/releases/ffmpeg-6.1.2.tar.xz
if [[ ! -f "${prefix}/lib/pkgconfig/libavformat.pc" ]]; then
    (
        cd "${sources}/ffmpeg-6.1.2"
        ./configure --prefix="${prefix}" --enable-shared --disable-static \
            --disable-programs --disable-doc --disable-autodetect \
            --disable-encoders --disable-muxers --disable-filters \
            --disable-avdevice --disable-avfilter --disable-postproc \
            --disable-network --enable-zlib
        make -j
        make install
    )
fi

fetch_source openssl-3.5.4 https://www.openssl.org/source/openssl-3.5.4.tar.gz
if [[ ! -f "${prefix}/lib/pkgconfig/openssl.pc" ]]; then
    (
        cd "${sources}/openssl-3.5.4"
        ./Configure --prefix="${prefix}" --libdir=lib shared no-tests
        make -j
        make install_sw
    )
fi

fetch_source curl-8.10.1 https://curl.se/download/curl-8.10.1.tar.xz
if [[ ! -f "${prefix}/lib/pkgconfig/libcurl.pc" ]]; then
    cmake -S "${sources}/curl-8.10.1" -B "${deps_dir}/curl-build" -G Ninja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${prefix}" \
        -DCMAKE_PREFIX_PATH="${prefix}" -DOPENSSL_ROOT_DIR="${prefix}" \
        -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF \
        -DBUILD_CURL_EXE=OFF -DBUILD_TESTING=OFF -DCURL_USE_OPENSSL=ON \
        -DCURL_USE_LIBPSL=OFF -DCURL_DISABLE_LDAP=ON -DCURL_DISABLE_LDAPS=ON
    cmake --build "${deps_dir}/curl-build" -j
    cmake --install "${deps_dir}/curl-build"
fi

PKG_CONFIG_PATH="${prefix}/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
    pkg-config --modversion libavformat libavcodec libavutil libswscale openssl libcurl
printf '\nDependency prefix: %s\n' "${prefix}"
printf 'Configure with PKG_CONFIG_PATH=%s/lib/pkgconfig\n' "${prefix}"
