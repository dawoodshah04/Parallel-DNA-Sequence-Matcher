#pragma once

#include <string>
#include <vector>
#include <chrono>

// ── Core data types ────────────────────────────────────────────────────────

struct DNASequence {
    std::string id;        // FASTA header without '>'
    std::string sequence;  // uppercase ACGT only
};

struct AlignmentResult {
    int         score         = 0;
    std::string aligned_query;
    std::string aligned_ref;
    int         query_start   = 0;  // 0-indexed start in original query
    int         ref_start     = 0;  // 0-indexed start in original ref
};

// ── High-resolution benchmark timer ───────────────────────────────────────

struct BenchmarkTimer {
    void   start();
    void   stop();
    double elapsed_ms() const;  // milliseconds as double

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start_{};
    std::chrono::time_point<std::chrono::high_resolution_clock> t_end_{};
};

// ── FASTA I/O ──────────────────────────────────────────────────────────────

// Parse a FASTA file; throws std::runtime_error if file cannot be opened.
// Non-ACGT characters are silently skipped with a warning to stderr.
std::vector<DNASequence> parse_fasta(const std::string& filepath);

// ── Random sequence generation ─────────────────────────────────────────────

// Generate a random ACGT sequence of the given length using the given seed.
std::string generate_random_sequence(int length, unsigned int seed);
