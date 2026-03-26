#include <cstdio>
#include <iostream>
#include "dna_utils.h"
#include "smith_waterman.h"
#include "kmp.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: dna_sequential <fasta_file>\n";
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

    std::cout << "Query  : " << sequences[0].id
              << " (" << query.size() << " bp)\n";
    std::cout << "DB     : " << db_count << " sequences\n\n";

    // ── Smith-Waterman over all DB sequences ──────────────────────────────
    BenchmarkTimer timer;
    timer.start();

    AlignmentResult best;
    int best_db = -1;
    for (int i = 0; i < db_count; ++i) {
        AlignmentResult r = smith_waterman(query, sequences[i + 1].sequence);
        if (r.score > best.score) {
            best    = r;
            best_db = i;
        }
    }

    timer.stop();

    std::cout << "=== Smith-Waterman ===\n";
    if (best_db >= 0) {
        std::cout << "Best match : " << sequences[best_db + 1].id
                  << "  score=" << best.score << '\n';
        std::cout << "Query  : " << best.aligned_query << '\n';
        std::cout << "Ref    : " << best.aligned_ref   << '\n';
    }
    std::printf("[SEQUENTIAL] SW Time: %.3f ms\n\n", timer.elapsed_ms());

    // ── KMP pattern search ────────────────────────────────────────────────
    const std::string pattern =
        query.substr(0, std::min<std::size_t>(20, query.size()));
    const std::string& ref = sequences[1].sequence;

    timer.start();
    auto kmp_hits = kmp_search(ref, pattern);
    timer.stop();

    std::cout << "=== KMP Pattern Search ===\n";
    std::cout << "Pattern (first 20 bp of query): " << pattern << '\n';
    std::cout << "Hits in '" << sequences[1].id << "': " << kmp_hits.size() << '\n';
    std::printf("[SEQUENTIAL] KMP Time: %.3f ms\n\n", timer.elapsed_ms());

    // ── Boyer-Moore pattern search ────────────────────────────────────────
    timer.start();
    auto bm_hits = boyer_moore_search(ref, pattern);
    timer.stop();

    std::cout << "=== Boyer-Moore Pattern Search ===\n";
    std::cout << "Hits in '" << sequences[1].id << "': " << bm_hits.size() << '\n';
    std::printf("[SEQUENTIAL] BM Time: %.3f ms\n", timer.elapsed_ms());

    return 0;
}
