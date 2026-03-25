#!/usr/bin/env python3
"""
gen_sequences.py — Random FASTA sequence generator for benchmarking.

Usage
-----
Generate 100 sequences of 1000 bp each:
    python3 gen_sequences.py --count 100 --length 1000 --output test_large.fasta

Generate with a mutated copy of the first sequence as the second entry:
    python3 gen_sequences.py --count 50 --length 500 --mutate 0.05
"""

import argparse
import random
import sys

BASES = ['A', 'C', 'G', 'T']


def generate_sequence(length: int, rng: random.Random) -> str:
    return ''.join(rng.choice(BASES) for _ in range(length))


def mutate_sequence(seq: str, rate: float, rng: random.Random) -> str:
    """Return a copy of seq with ~rate fraction of bases randomly substituted."""
    result = list(seq)
    for i in range(len(result)):
        if rng.random() < rate:
            result[i] = rng.choice(BASES)
    return ''.join(result)


def write_fasta(filepath: str, sequences: list[tuple[str, str]]) -> None:
    with open(filepath, 'w') as f:
        for seq_id, seq in sequences:
            f.write(f'>{seq_id}\n')
            # Wrap at 70 characters per line (standard FASTA)
            for i in range(0, len(seq), 70):
                f.write(seq[i:i + 70] + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate random DNA sequences in FASTA format.')
    parser.add_argument('--count',  type=int,   default=10,
                        help='Number of sequences to generate (default: 10)')
    parser.add_argument('--length', type=int,   default=200,
                        help='Length of each sequence in bp (default: 200)')
    parser.add_argument('--output', type=str,   default='sequences.fasta',
                        help='Output FASTA filepath (default: sequences.fasta)')
    parser.add_argument('--seed',   type=int,   default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--mutate', type=float, default=0.0,
                        help='If > 0, the second sequence is a mutated copy '
                             'of the first with this mutation rate (default: 0)')
    args = parser.parse_args()

    if args.count < 1:
        sys.exit('Error: --count must be >= 1')
    if args.length < 1:
        sys.exit('Error: --length must be >= 1')
    if not (0.0 <= args.mutate <= 1.0):
        sys.exit('Error: --mutate must be in [0, 1]')

    rng = random.Random(args.seed)
    sequences: list[tuple[str, str]] = []

    # First sequence — the query
    first_seq = generate_sequence(args.length, rng)
    sequences.append(('query_seq1', first_seq))

    # Second sequence — mutated copy of first (if --mutate > 0)
    if args.mutate > 0.0 and args.count >= 2:
        mutated = mutate_sequence(first_seq, args.mutate, rng)
        sequences.append((f'db_seq2_mutated_{int(args.mutate * 100)}pct', mutated))
        start = 2
    else:
        start = 1

    # Remaining random sequences
    for i in range(start, args.count):
        seq = generate_sequence(args.length, rng)
        sequences.append((f'db_seq{i + 1}', seq))

    write_fasta(args.output, sequences)
    print(f'Generated {len(sequences)} sequences of {args.length} bp → {args.output}')


if __name__ == '__main__':
    main()
