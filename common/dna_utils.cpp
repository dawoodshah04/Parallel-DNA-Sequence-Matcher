#include "dna_utils.h"

#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

// ── BenchmarkTimer ─────────────────────────────────────────────────────────

void BenchmarkTimer::start() {
    t_start_ = std::chrono::high_resolution_clock::now();
}

void BenchmarkTimer::stop() {
    t_end_ = std::chrono::high_resolution_clock::now();
}

double BenchmarkTimer::elapsed_ms() const {
    return std::chrono::duration<double, std::milli>(t_end_ - t_start_).count();
}

// ── FASTA parser ───────────────────────────────────────────────────────────

std::vector<DNASequence> parse_fasta(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open FASTA file: " + filepath);
    }

    std::vector<DNASequence> sequences;
    DNASequence current;
    std::string line;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (line[0] == '>') {
            if (!current.id.empty()) {
                sequences.push_back(std::move(current));
                current = {};
            }
            current.id = line.substr(1);
            // Trim trailing whitespace from ID
            while (!current.id.empty() &&
                   std::isspace(static_cast<unsigned char>(current.id.back()))) {
                current.id.pop_back();
            }
        } else {
            for (char c : line) {
                char uc = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                if (uc == 'A' || uc == 'C' || uc == 'G' || uc == 'T') {
                    current.sequence += uc;
                } else if (!std::isspace(static_cast<unsigned char>(c))) {
                    std::cerr << "Warning: skipping non-ACGT character '"
                              << c << "' in sequence '" << current.id << "'\n";
                }
            }
        }
    }

    if (!current.id.empty()) {
        sequences.push_back(std::move(current));
    }

    return sequences;
}

// ── Random sequence generator ──────────────────────────────────────────────

std::string generate_random_sequence(int length, unsigned int seed) {
    static constexpr char kBases[] = {'A', 'C', 'G', 'T'};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 3);

    std::string seq;
    seq.reserve(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) {
        seq += kBases[dist(rng)];
    }
    return seq;
}
