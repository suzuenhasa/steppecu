// src/io/beagle_reader.hpp
//
// Beagle genotype-likelihood ingest — the staging arm that feeds the shipped
// LikelihoodTile -> device LikelihoodTensor path for `steppe pcangsd`. A beagle
// GL file (gzip or plain) is whitespace text:
//
//   marker  allele1 allele2  Ind0 Ind0 Ind0  Ind1 Ind1 Ind1  ...
//
// with 3 GL columns per individual per data row = (P(A1A1), P(A1A2), P(A2A2)),
// already linear and (nominally) summing to 1, where allele1 is the reference
// (major) allele. This reader is deliberately THIN: no panel join, no lift, no
// polarity vs a target panel — PCAngsd is self-contained on the beagle markers.
//
// The parsed triplet is stored REVERSED into the LikelihoodTile g-axis (g = copies
// of A1): beagle gives (P(A1A1), P(A1A2), P(A2A2)) = (2,1,0 copies of A1), so
// l[base+0]=P(A2A2), l[base+1]=P(A1A2), l[base+2]=P(A1A1). This keeps the tile
// contract (g = copies of A1) intact; the PCAngsd EM then counts copies of A2 to
// match pcangsd's individual-allele-frequency sign. Uniform per-triplet
// renormalization (normalize_gp) does not change any PCAngsd output (E_dosage is a
// ratio of like-scaled terms), so it is safe for the tile contract.
//
// Pure host C++20 io-leaf (links zlib via GzipLineReader only); failures surface as
// std::runtime_error.
#ifndef STEPPE_IO_BEAGLE_READER_HPP
#define STEPPE_IO_BEAGLE_READER_HPP

#include <string>

#include "io/likelihood_tile.hpp"

namespace steppe::io {

struct BeagleReadResult {
    LikelihoodTile tile;
    long n_site = 0;
    int n_sample = 0;
};

// Parse a beagle GL file (gzip or plain) into a site-major LikelihoodTile.
// n_sample = (header_cols - 3)/3; every data row must have exactly 3*n_sample+3
// whitespace fields or the read throws. Sample IDs are the distinct header tokens
// at columns 3,6,9,... (beagle repeats each id 3x; the first of each triple is
// taken, and if the repeats are not distinct, synthetic Ind{k} ids are used).
[[nodiscard]] BeagleReadResult read_beagle_gl(const std::string& path);

}  // namespace steppe::io

#endif  // STEPPE_IO_BEAGLE_READER_HPP
