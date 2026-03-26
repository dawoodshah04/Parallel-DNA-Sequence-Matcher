/*
 * Parallel exact pattern matching kernel.
 *
 * Each work-item i checks whether pattern appears at text[i].
 * This is the naive parallel search: O(m) work per work-item, O(n) work-items.
 * Suitable for short patterns relative to a large reference sequence.
 *
 * Arguments:
 *   text        — reference sequence, length text_len
 *   text_len    — length of text
 *   pattern     — search pattern, length pattern_len
 *   pattern_len — length of pattern
 *   matches     — output: matches[i] = 1 if pattern starts at text[i], else 0
 */
__kernel void naive_search(
    __global const char* text,
    int text_len,
    __global const char* pattern,
    int pattern_len,
    __global int* matches)
{
    int i = get_global_id(0);

    if (i + pattern_len > text_len) {
        matches[i] = 0;
        return;
    }

    int found = 1;
    for (int k = 0; k < pattern_len; ++k) {
        if (text[i + k] != pattern[k]) {
            found = 0;
            break;
        }
    }
    matches[i] = found;
}

/*
 * Count the total number of matches across the matches[] array.
 * Launched with global_size = 1 after naive_search completes.
 */
__kernel void count_matches(
    __global const int* matches,
    int n,
    __global int* total)
{
    int count = 0;
    for (int i = 0; i < n; ++i) count += matches[i];
    *total = count;
}
