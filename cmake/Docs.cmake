# cmake/Docs.cmake
#
# Opt-in GENERATED API REFERENCE (kimiactions C5), included from the top-level
# CMakeLists.txt only when STEPPE_BUILD_DOCS is ON. Two independent, OPTIONAL custom
# targets, each a SOFT NO-OP when its tool is absent so a normal build never breaks:
#
#   docs         Doxygen over include/steppe -> ${STEPPE_DOXYGEN_OUTPUT_DIR}/html
#   docs-python  pdoc over bindings/steppe   -> ${CMAKE_BINARY_DIR}/docs/api/python
#
# DOC-ONLY: defines NO library/object targets and is NOT on the compute/precision path
# (architecture.md §4 "NO targets defined in cmake/" applies to first-party libraries;
# these are doc-generation custom targets, build only when explicitly requested).
#
# KEY PROPERTY: Doxygen parses headers as TEXT — no compiler, no CUDA — so the `docs`
# target builds on the LOCAL non-Blackwell box (unlike every other target). The pdoc
# `docs-python` target DOES import the facade, which imports the compiled `_core`
# extension, so it needs a built `_core` on PYTHONPATH (run it on the box / post-wheel);
# see docs/archive/api/README.md.

include_guard(GLOBAL)

# --- C++ public headers -> Doxygen HTML (the `docs` target) ------------------
find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  # @-substituted into docs/Doxyfile.in by configure_file below.
  set(STEPPE_DOXYGEN_INPUT "${PROJECT_SOURCE_DIR}/include/steppe")
  set(STEPPE_DOXYGEN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/docs/api" CACHE PATH
      "Doxygen OUTPUT_DIRECTORY for the `docs` target (HTML lands in <dir>/html)")

  set(_steppe_doxyfile "${CMAKE_BINARY_DIR}/docs/Doxyfile")
  configure_file("${PROJECT_SOURCE_DIR}/docs/archive/Doxyfile.in" "${_steppe_doxyfile}" @ONLY)

  # Doxygen::doxygen is the FindDoxygen imported executable target (CMake >= 3.9),
  # preferred over the deprecated DOXYGEN_EXECUTABLE variable.
  add_custom_target(docs
    COMMAND Doxygen::doxygen "${_steppe_doxyfile}"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "steppe: Doxygen C++ API reference -> ${STEPPE_DOXYGEN_OUTPUT_DIR}/html"
    VERBATIM)
  message(STATUS "steppe: docs target enabled (Doxygen ${DOXYGEN_VERSION})")
else()
  message(STATUS "steppe: Doxygen not found - `docs` target disabled (soft no-op)")
endif()

# --- Python facade -> pdoc HTML (the `docs-python` target) -------------------
# Modern pdoc CLI: `pdoc -o <outdir> <package>` (reads the existing __init__.py
# docstrings; no new content). Soft no-op when pdoc is not on PATH.
find_program(STEPPE_PDOC_EXECUTABLE NAMES pdoc)
if(STEPPE_PDOC_EXECUTABLE)
  add_custom_target(docs-python
    COMMAND "${STEPPE_PDOC_EXECUTABLE}"
            -o "${CMAKE_BINARY_DIR}/docs/api/python"
            "${PROJECT_SOURCE_DIR}/bindings/steppe"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "steppe: pdoc Python API reference -> ${CMAKE_BINARY_DIR}/docs/api/python"
    VERBATIM)
  message(STATUS "steppe: docs-python target enabled (${STEPPE_PDOC_EXECUTABLE})")
else()
  message(STATUS "steppe: pdoc not found - `docs-python` target disabled (soft no-op)")
endif()
