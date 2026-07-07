// src/app/pca_html_writer.hpp
//
// write_pca_html — serialize a PcaResult into ONE self-contained interactive scatter HTML
// file (the `steppe pca --emit-html` artifact). Everything is inline (CSS in <style>, JS in
// <script>, the PC coordinates in a JSON data literal); there is NO external reference —
// no CDN, no <link>, no remote font, no network fetch — so the file opens offline. The
// renderer is a dependency-free canvas-2D scatter (PC-axis selectors, pan/zoom, hover
// labels, a click-to-toggle population legend, a scree strip). App-only, CUDA-free host
// stream I/O. The data schema (axisNames/coords) is generic so a later UMAP embedding
// slots in as two more axis entries with no template change.
#ifndef STEPPE_APP_PCA_HTML_WRITER_HPP
#define STEPPE_APP_PCA_HTML_WRITER_HPP

#include <string>

#include "steppe/pca.hpp"

namespace steppe::app {

// Write the self-contained interactive PCA scatter HTML to `path`. Returns true on success;
// on an open/write failure it prints a `steppe <prefix>:` diagnostic to stderr and returns
// false. `prefix` is the CLI command name for that diagnostic (e.g. "pca").
[[nodiscard]] bool write_pca_html(const std::string& path, const PcaResult& result,
                                  const char* prefix);

}  // namespace steppe::app

#endif  // STEPPE_APP_PCA_HTML_WRITER_HPP
