# cmake/CPM.cmake — CPM.cmake bootstrap shim (architecture.md §6, §17 step 2).
#
# The architecture names `cmake/CPM.cmake` as the single home of the dependency
# fetcher (architecture.md:384 `include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)`).
# This is the upstream-recommended downloader bootstrap: it does NOT vendor CPM;
# it downloads the PINNED, SHA256-VERIFIED CPM.cmake release into the build/cache
# tree on first use and `include()`s it, so `CPMAddPackage(...)` becomes available.
# On a box that already has a CPM copy (CPM_SOURCE_CACHE / a prior fetch) it reuses
# it — re-checking the pinned SHA256 — and makes NO network call.
#
# This file is included only from the two CUDA-free leaf subtrees that need a fetched
# dependency: `src/app/CMakeLists.txt` (behind STEPPE_BUILD_CLI, for CLI11) and
# `bindings/CMakeLists.txt` (behind STEPPE_BUILD_PYTHON, for nanobind). The CPM
# machinery — and the deps it fetches — never reach the core/device build; CLI11 and
# nanobind each stay PRIVATE to their subtree (architecture.md §4 layering: the
# CUDA-free `app`/`bindings` layers are the only places that may pull these deps).

# Pinned CPM bootstrap: the release tag fixes the version, and CPM_EXPECTED_SHA256 is
# the SHA256 of that immutable GitHub release asset — the exact bytes this shim
# `include()`s and EXECUTES as CMake, so they are integrity-verified before they run.
# Recompute after a version bump:
#   curl -fsSL https://github.com/cpm-cmake/CPM.cmake/releases/download/v<ver>/CPM.cmake | sha256sum
set(CPM_DOWNLOAD_VERSION 0.42.3)
set(CPM_EXPECTED_SHA256 "a609e875fd532b067174250f6abbc3dac22fe2d64869783fb1e80bda1625c844")

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Fetch the pinned CPM.cmake release on first use and verify it against
# CPM_EXPECTED_SHA256; reuse the cached copy after. `file(DOWNLOAD ... EXPECTED_HASH)`
# skips the network when a cached file already matches the hash and HARD-FAILS on a
# mismatch (CMake file() docs), so a tampered or re-pointed asset is rejected at
# configure time before any of its CMake runs. The reuse branch (a pre-placed
# CPM_SOURCE_CACHE copy) is integrity-checked against the same pin via file(SHA256),
# so a cached entry can never be include()'d unverified.
function(steppe_cpm_download)
  if(NOT (EXISTS "${CPM_DOWNLOAD_LOCATION}"))
    message(STATUS "steppe: downloading CPM.cmake v${CPM_DOWNLOAD_VERSION} -> ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
         "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
         "${CPM_DOWNLOAD_LOCATION}"
         EXPECTED_HASH SHA256=${CPM_EXPECTED_SHA256}
         STATUS _cpm_dl_status)
    list(GET _cpm_dl_status 0 _cpm_dl_code)
    if(NOT _cpm_dl_code EQUAL 0)
      list(GET _cpm_dl_status 1 _cpm_dl_msg)
      message(FATAL_ERROR
              "steppe: failed to fetch/verify CPM.cmake v${CPM_DOWNLOAD_VERSION}: ${_cpm_dl_msg}.\n"
              "(A SHA256 mismatch against the pinned hash also reports through this path.)\n"
              "Set CPM_SOURCE_CACHE to a populated cache, or pre-place the matching CPM.cmake at "
              "${CPM_DOWNLOAD_LOCATION}, to build the CLI/bindings offline (architecture.md §6).")
    endif()
  else()
    # Reuse branch: integrity-check the pre-placed / previously-fetched copy against the
    # same pin, so a stale or tampered cache entry can never be include()'d unverified.
    file(SHA256 "${CPM_DOWNLOAD_LOCATION}" _cpm_cached_sha256)
    if(NOT _cpm_cached_sha256 STREQUAL "${CPM_EXPECTED_SHA256}")
      message(FATAL_ERROR
              "steppe: cached CPM.cmake at ${CPM_DOWNLOAD_LOCATION} fails the pinned SHA256 "
              "(expected ${CPM_EXPECTED_SHA256}, got ${_cpm_cached_sha256}). Delete it (or fix "
              "CPM_SOURCE_CACHE) and reconfigure to re-fetch the verified v${CPM_DOWNLOAD_VERSION} asset.")
    endif()
  endif()
endfunction()

steppe_cpm_download()
include("${CPM_DOWNLOAD_LOCATION}")
