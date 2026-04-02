/*
 * Smith-Waterman optimized single-kernel wavefront with GPU-side barriers.
 *
 * All cells on a given anti-diagonal d (i + j == d) are data-independent
 * and can be computed in parallel. This version uses a single kernel launch
 * with internal diagonal iteration, eliminating host-device synchronization overhead.
 *
 * Arguments:
 *   H           — DP matrix, row-major, size (m+1)*(n+1), cl_int
 *   query       — query sequence characters, length m
 *   ref         — reference sequence characters, length n
 *   m           — query length
 *   n           — reference length
 *   match_score — score for matching bases (SW_MATCH)
 *   mismatch    — score for mismatching bases (SW_MISMATCH)
 *   gap         — gap penalty (SW_GAP, negative)
 *
 * NOTE: This kernel requires workgroup size that can handle max diagonal length.
 * For very large sequences, may need to fall back to multi-kernel approach.
 */
__kernel void sw_wavefront_optimized(
    __global int*        H,
    __global const char* query,
    __global const char* ref,
    int m,
    int n,
    int match_score,
    int mismatch,
    int gap)
{
    int gid = get_global_id(0);
    int max_diag_len = (m < n) ? m : n;
    
    if (gid >= max_diag_len) return;
    
    int stride = n + 1;
    
    // Iterate through all diagonals
    for (int diag = 2; diag <= m + n; ++diag) {
        int i_min = max(1, diag - n);
        int i_max = min(m, diag - 1);
        
        if (i_min > i_max) continue;
        
        int diag_len = i_max - i_min + 1;
        
        // Only active threads participate
        if (gid < diag_len) {
            int i = i_min + gid;
            int j = diag - i;
            
            char q = query[i - 1];
            char r = ref[j - 1];
            int  s = (q == r) ? match_score : mismatch;
            
            int diag_val = H[(i - 1) * stride + (j - 1)] + s;
            int up_val   = H[(i - 1) * stride +  j     ] + gap;
            int left_val = H[ i      * stride + (j - 1)] + gap;
            
            int val = max(0, max(diag_val, max(up_val, left_val)));
            H[i * stride + j] = val;
        }
        
        // Synchronize all threads before next diagonal
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

/*
 * Original Smith-Waterman anti-diagonal wavefront kernel (multi-launch version).
 * Used when sequences are too large for single-kernel approach.
 *
 * All cells on a given anti-diagonal d (i + j == d) are data-independent
 * and can be computed in parallel.  The host launches one NDRange per
 * diagonal, stepping from d=2 to d=(m+n).
 *
 * Arguments:
 *   H           — DP matrix, row-major, size (m+1)*(n+1), cl_int
 *   query       — query sequence characters, length m
 *   ref         — reference sequence characters, length n
 *   m           — query length
 *   n           — reference length
 *   diag        — current anti-diagonal index (i+j == diag)
 *   match_score — score for matching bases (SW_MATCH)
 *   mismatch    — score for mismatching bases (SW_MISMATCH)
 *   gap         — gap penalty (SW_GAP, negative)
 */
__kernel void sw_wavefront(
    __global int*        H,
    __global const char* query,
    __global const char* ref,
    int m,
    int n,
    int diag,
    int match_score,
    int mismatch,
    int gap)
{
    int idx = get_global_id(0);

    /* Map linear index to (i, j) on this diagonal.
     * i ranges from max(1, diag-n) to min(m, diag-1). */
    int i_min = max(1, diag - n);
    int i     = i_min + idx;
    int j     = diag - i;

    if (i < 1 || i > m || j < 1 || j > n) return;

    int stride = n + 1;

    char q = query[i - 1];
    char r = ref[j - 1];
    int  s = (q == r) ? match_score : mismatch;

    int diag_val = H[(i - 1) * stride + (j - 1)] + s;
    int up_val   = H[(i - 1) * stride +  j     ] + gap;
    int left_val = H[ i      * stride + (j - 1)] + gap;

    int val = max(0, max(diag_val, max(up_val, left_val)));
    H[i * stride + j] = val;
}

/*
 * Parallel reduction to find maximum value and position in DP matrix.
 * Uses work-group reductions with local memory for efficiency.
 */
__kernel void find_max_parallel(
    __global const int* H,
    int rows,
    int cols,
    __global int* out_score,
    __global int* out_row,
    __global int* out_col,
    __local int* local_vals,
    __local int* local_rows,
    __local int* local_cols)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int wg_size = get_local_size(0);
    
    int total_elems = rows * cols;
    
    // Each thread finds max in its partition
    int max_val = 0, max_r = 0, max_c = 0;
    for (int idx = gid; idx < total_elems; idx += get_global_size(0)) {
        int i = idx / cols;
        int j = idx % cols;
        if (i >= 1 && j >= 1) {
            int v = H[idx];
            if (v > max_val) {
                max_val = v;
                max_r = i;
                max_c = j;
            }
        }
    }
    
    // Store in local memory
    local_vals[lid] = max_val;
    local_rows[lid] = max_r;
    local_cols[lid] = max_c;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Reduce within work-group
    for (int offset = wg_size / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            if (local_vals[lid + offset] > local_vals[lid]) {
                local_vals[lid] = local_vals[lid + offset];
                local_rows[lid] = local_rows[lid + offset];
                local_cols[lid] = local_cols[lid + offset];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // First thread in each work-group writes to global (needs final reduction if multiple groups)
    if (lid == 0) {
        // Use atomic compare-and-swap pattern for global max
        // Since OpenCL doesn't have built-in atomic max with position tracking,
        // we'll use a simpler approach: have work-group 0 do final reduction
        if (get_group_id(0) == 0) {
            *out_score = local_vals[0];
            *out_row = local_rows[0];
            *out_col = local_cols[0];
        } else {
            // For multi-group, would need a second pass or atomic operations
            // For now, using single work-group launch recommended
        }
    }
}

/*
 * Reduce across the complete DP matrix to find the maximum value and its
 * position.  Launched with global_size = 1 (single work-item) after the
 * wavefront is fully complete. (Original serial version)
 */
__kernel void find_max(
    __global const int* H,
    int rows,
    int cols,
    __global int* out_score,
    __global int* out_row,
    __global int* out_col)
{
    int max_val = 0, max_r = 0, max_c = 0;
    for (int i = 1; i < rows; ++i) {
        for (int j = 1; j < cols; ++j) {
            int v = H[i * cols + j];
            if (v > max_val) {
                max_val = v;
                max_r   = i;
                max_c   = j;
            }
        }
    }
    *out_score = max_val;
    *out_row   = max_r;
    *out_col   = max_c;
}
