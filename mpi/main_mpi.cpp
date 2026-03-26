#include <mpi.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include "dna_utils.h"
#include "smith_waterman.h"
#include "kmp.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) std::cerr << "Usage: dna_mpi <fasta_file>\n";
        MPI_Finalize();
        return 1;
    }

    // All ranks load the same FASTA — file is small relative to sequences
    std::vector<DNASequence> sequences;
    try {
        sequences = parse_fasta(argv[1]);
    } catch (const std::exception& e) {
        std::cerr << "Rank " << rank << " error: " << e.what() << '\n';
        MPI_Finalize();
        return 1;
    }

    if (sequences.size() < 2) {
        if (rank == 0)
            std::cerr << "Error: FASTA file must contain at least 2 sequences.\n";
        MPI_Finalize();
        return 1;
    }

    const std::string& query = sequences[0].sequence;
    int db_count = static_cast<int>(sequences.size()) - 1;

    if (rank == 0) {
        std::cout << "Query  : " << sequences[0].id
                  << " (" << query.size() << " bp)\n";
        std::cout << "DB     : " << db_count << " sequences\n";
        std::cout << "Ranks  : " << size << "\n\n";
    }

    // ── Smith-Waterman — round-robin partition across ranks ───────────────
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    int local_best_score = 0;
    int local_best_idx   = -1;
    AlignmentResult local_best;

    for (int i = rank; i < db_count; i += size) {
        AlignmentResult r = smith_waterman(query, sequences[i + 1].sequence);
        if (r.score > local_best_score) {
            local_best_score = r.score;
            local_best_idx   = i;
            local_best       = r;
        }
    }

    // Reduce: find global best score at rank 0
    int global_best_score = 0;
    MPI_Reduce(&local_best_score, &global_best_score, 1,
               MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=== Smith-Waterman (MPI) ===\n";
        std::cout << "Global best score: " << global_best_score << '\n';
        std::printf("[MPI] SW Time: %.3f ms\n\n", (t_end - t_start) * 1000.0);
    }

    // ── KMP — parallel over DB sequences ─────────────────────────────────
    const std::string pattern =
        query.substr(0, std::min<std::size_t>(20, query.size()));

    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    int local_hits = 0;
    for (int i = rank; i < db_count; i += size) {
        auto hits = kmp_search(sequences[i + 1].sequence, pattern);
        local_hits += static_cast<int>(hits.size());
    }

    int global_hits = 0;
    MPI_Reduce(&local_hits, &global_hits, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=== KMP Pattern Search (MPI) ===\n";
        std::cout << "Pattern      : " << pattern << '\n';
        std::cout << "Total hits   : " << global_hits << '\n';
        std::printf("[MPI] KMP Time: %.3f ms\n", (t_end - t_start) * 1000.0);
    }

    MPI_Finalize();
    return 0;
}
