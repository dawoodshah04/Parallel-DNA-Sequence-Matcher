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
 * Count the total number of matches using parallel tree-based reduction.
 * Work-group size should be a power of 2 (e.g., 256).
 * Global size should be a multiple of work-group size.
 */
__kernel void count_matches(
    __global const int* matches,
    int n,
    __global int* total,
    __local int* scratch)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    int group_id = get_group_id(0);
    int num_groups = get_num_groups(0);
    
    // Each work-item loads one element (or 0 if out of bounds)
    int local_sum = (gid < n) ? matches[gid] : 0;
    scratch[lid] = local_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree-based reduction within work-group
    for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            scratch[lid] += scratch[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // First work-item in each group writes partial sum to global memory
    if (lid == 0) {
        // Use atomic add to accumulate across all work-groups
        atomic_add(total, scratch[0]);
    }
}
