#include <cstdio>
#include <iostream>
#include <vector>
#include <omp.h>
#include "dna_utils.h"
#include "smith_waterman.h"
#include "kmp.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: dna_openmp <fasta_file>\n";
        return 1;
    }

    std::vector<DNASequence> sequences;
    try {
        sequences = parse_fasta(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    if (sequences.size() < 2) {
        std::cerr << "Error: FASTA file must contain at least 2 sequences.\n";
        return 1;
    }

    const std::string& query = sequences[0].sequence;
    int db_count = static_cast<int>(sequences.size()) - 1;

    std::cout << "Query   : " << sequences[0].id
              << " (" << query.size() << " bp)\n";
    std::cout << "DB      : " << db_count << " sequences\n";
    std::cout << "Threads : " << omp_get_max_threads() << "\n\n";

    // ── Smith-Waterman — parallel over DB sequences ───────────────────────
    std::vector<AlignmentResult> results(db_count);

    double t_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < db_count; ++i) {
        results[i] = smith_waterman(query, sequences[i + 1].sequence);
    }

    double t_end = omp_get_wtime();

    // Find best result (serial — negligible)
    int best_idx = 0;
    for (int i = 1; i < db_count; ++i) {
        if (results[i].score > results[best_idx].score) best_idx = i;
    }

    std::cout << "=== Smith-Waterman (OpenMP) ===\n";
    std::cout << "Best match : " << sequences[best_idx + 1].id
              << "  score=" << results[best_idx].score << '\n';
    std::cout << "Query  : " << results[best_idx].aligned_query << '\n';
    std::cout << "Ref    : " << results[best_idx].aligned_ref   << '\n';
    std::printf("[OPENMP] SW Time: %.3f ms\n\n", (t_end - t_start) * 1000.0);

    // ── KMP — parallel over DB sequences ─────────────────────────────────
    const std::string pattern =
        query.substr(0, std::min<std::size_t>(20, query.size()));

    std::vector<std::vector<int>> kmp_results(db_count);

    t_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < db_count; ++i) {
        kmp_results[i] = kmp_search(sequences[i + 1].sequence, pattern);
    }

    t_end = omp_get_wtime();

    std::size_t total_hits = 0;
    for (auto& v : kmp_results) total_hits += v.size();

    std::cout << "=== KMP Pattern Search (OpenMP) ===\n";
    std::cout << "Pattern          : " << pattern << '\n';
    std::cout << "Total hits across " << db_count << " sequences: " << total_hits << '\n';
    std::printf("[OPENMP] KMP Time: %.3f ms\n\n", (t_end - t_start) * 1000.0);

    // ── Boyer-Moore — parallel over DB sequences ──────────────────────────
    std::vector<std::vector<int>> bm_results(db_count);

    t_start = omp_get_wtime();

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < db_count; ++i) {
        bm_results[i] = boyer_moore_search(sequences[i + 1].sequence, pattern);
    }

    t_end = omp_get_wtime();

    total_hits = 0;
    for (auto& v : bm_results) total_hits += v.size();

    std::cout << "=== Boyer-Moore Pattern Search (OpenMP) ===\n";
    std::cout << "Total hits across " << db_count << " sequences: " << total_hits << '\n';
    std::printf("[OPENMP] BM Time: %.3f ms\n", (t_end - t_start) * 1000.0);

    return 0;
}
