/*
 * Smith-Waterman anti-diagonal wavefront kernel.
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
 * Reduce across the complete DP matrix to find the maximum value and its
 * position.  Launched with global_size = 1 (single work-item) after the
 * wavefront is fully complete.
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
