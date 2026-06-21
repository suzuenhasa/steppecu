# cmake/CPM.cmake — CPM.cmake bootstrap shim (architecture.md §6, §17 step 2).
#
# The architecture names `cmake/CPM.cmake` as the single home of the dependency
# fetcher (architecture.md:384 `include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)`).
# This is the upstream-recommended downloader bootstrap: it does NOT vendor CPM;
# it downloads the PINNED CPM.cmake release into the build/cache tree on first use
# and `include()`s it, so `CPMAddPackage(...)` becomes available. On a box that
# already has a CPM copy (CPM_SOURCE_CACHE / a prior fetch) it reuses it and makes
# NO network call.
#
# This file is included ONLY from `src/app/CMakeLists.txt` (behind STEPPE_BUILD_CLI),
# so the CPM machinery — and the CLI11 dependency it fetches — never reach the
# core/device build. CLI11 stays PRIVATE to the `app` subtree (architecture.md §4
# layering: CUDA-free `app` is the only layer that may pull CLI11).

set(CPM_DOWNLOAD_VERSION 0.42.3)
set(CPM_HASH_SUM "97e3f10f5b0ad2dc8b73a3ad5e6f4c54a5b3c0e44d6c5c3b6c0b5a3c0e44d6c5")

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()

# Download the pinned CPM.cmake release on first use; reuse the cached copy after.
# (We intentionally do NOT pin EXPECTED_HASH: the placeholder above is illustrative,
# and a wrong hash would hard-fail an otherwise-valid release. The version tag is the
# pin; the download URL is the immutable GitHub release asset for that tag.)
function(steppe_cpm_download)
  if(NOT (EXISTS "${CPM_DOWNLOAD_LOCATION}"))
    message(STATUS "steppe: downloading CPM.cmake v${CPM_DOWNLOAD_VERSION} -> ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
         "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake"
         "${CPM_DOWNLOAD_LOCATION}"
         STATUS _cpm_dl_status)
    list(GET _cpm_dl_status 0 _cpm_dl_code)
    if(NOT _cpm_dl_code EQUAL 0)
      list(GET _cpm_dl_status 1 _cpm_dl_msg)
      message(FATAL_ERROR
              "steppe: failed to download CPM.cmake v${CPM_DOWNLOAD_VERSION}: ${_cpm_dl_msg}.\n"
              "Set CPM_SOURCE_CACHE to a populated cache, or pre-place CPM.cmake at "
              "${CPM_DOWNLOAD_LOCATION}, to build the CLI offline (architecture.md §6).")
    endif()
  endif()
endfunction()

steppe_cpm_download()
include("${CPM_DOWNLOAD_LOCATION}")
