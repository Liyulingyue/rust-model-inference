//! `linear_sum_assignment` — the reference's own Hungarian solver.
//!
//! Mirrors `gliner2.training.matching.linear_sum_assignment`
//! (`target/gliner2-oracle/gliner2/training/matching.py:22`).
//!
//! # Why port the reference's solver and not scipy
//!
//! The reference has two paths. If `scipy` imports it calls
//! `scipy.optimize.linear_sum_assignment` with a **sub-ULP lexicographic offset**
//! added to break exact ties reproducibly; otherwise it runs the internal
//! O(n²m) Jonker-Volgenant-style shortest augmenting path below. The module
//! docstring says the internal one exists "rather than depending on SciPy, so
//! results are reproducible and dependency-free".
//!
//! This port implements the internal solver, which is the one that is actually
//! specified — it is deterministic on its own, with no floating-point
//! perturbation that would make the result depend on a dependency being
//! installed. The oracle venv has no scipy, so the reference takes the internal
//! path there and the fixtures pin this one.
//!
//! # The tie-break is load-bearing
//!
//! The decoder's scalar assignment is a genuine optimisation problem, and its
//! optimum is frequently **not unique** (two spans with the same probability).
//! Which optimum the solver returns decides which span each record field binds,
//! so it is part of the output, not an implementation detail. Two places fix it:
//!
//! * `minv[j] < delta` is a *strict* comparison, so the lowest column index wins
//!   a tie for the next augmenting column.
//! * The final pairs are built by scanning columns left to right and then sorted
//!   by row, so the output order is a function of the assignment alone.
//!
//! A solver that returns a different optimum — a different library, a different
//! epsilon — would produce different spans while every cost in the table stays
//! identical. That is why the oracle checks the pairs, not the total.

/// Minimum-cost assignment. `cost` is `[rows][cols]`.
///
/// Returns `(row_ind, col_ind)` with `row_ind` ascending and
/// `min(rows, cols)` pairs, matching the reference.
pub fn linear_sum_assignment(cost: &[Vec<f64>]) -> Result<(Vec<usize>, Vec<usize>), String> {
    let rows = cost.len();
    if rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let cols = cost[0].len();
    if cost.iter().any(|row| row.len() != cols) {
        return Err("cost_matrix rows must all have the same length".into());
    }
    if cols == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if cost.iter().flatten().any(|value| value.is_nan()) {
        return Err("cost_matrix contains NaN".into());
    }

    // The reference promotes to float64 and maps +/-inf to a large finite
    // sentinel, so a fully-infinite row still admits a least-bad column instead
    // of failing the search.
    let needs_sentinel = cost.iter().flatten().any(|value| value.is_infinite());
    let mut work: Vec<Vec<f64>> = if needs_sentinel {
        let scale = cost
            .iter()
            .flatten()
            .filter(|value| value.is_finite())
            .fold(1.0f64, |acc, value| acc.max(value.abs()));
        let big = 1.0e6 * (scale + 1.0);
        cost.iter()
            .map(|row| {
                row.iter()
                    .map(|value| match value {
                        v if v.is_infinite() && *v > 0.0 => big,
                        v if v.is_infinite() => -big,
                        v => *v,
                    })
                    .collect()
            })
            .collect()
    } else {
        cost.to_vec()
    };

    // `transposed = c < r`: the solver assumes `n <= m`, so a tall matrix is
    // solved transposed and the result swapped back.
    let transposed = cols < rows;
    if transposed {
        work = transpose(&work);
    }
    let n = work.len();
    let m = work[0].len();

    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    // `p[j]` is the row assigned to column j, 1-indexed; 0 means unassigned.
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=m {
                if used[j] {
                    continue;
                }
                let cur = work[i0 - 1][j - 1] - u[i0] - v[j];
                if cur < minv[j] {
                    minv[j] = cur;
                    way[j] = j0;
                }
                // Strict `<`, so the lowest column index wins a tie for the
                // next augmenting column. See the module note on tie-breaks.
                if minv[j] < delta {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // Walk the augmenting path back, flipping the assignment along it.
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(n.min(m));
    for j in 1..=m {
        if p[j] != 0 {
            pairs.push((p[j] - 1, j - 1));
        }
    }
    pairs.sort_unstable();
    if transposed {
        pairs = pairs.into_iter().map(|(row, col)| (col, row)).collect();
        pairs.sort_unstable();
    }
    let (row_ind, col_ind): (Vec<usize>, Vec<usize>) = pairs.into_iter().unzip();
    Ok((row_ind, col_ind))
}

fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if matrix.is_empty() {
        return Vec::new();
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    let mut out = vec![vec![0.0f64; rows]; cols];
    for (i, row) in matrix.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            out[j][i] = *value;
        }
    }
    out
}

/// The total cost of an assignment, for tests and for the reference-comparison
/// in the oracle.
pub fn assignment_cost(cost: &[Vec<f64>], rows: &[usize], cols: &[usize]) -> f64 {
    rows.iter()
        .zip(cols)
        .map(|(row, col)| cost[*row][*col])
        .sum()
}

/// True when no two pairs share a row or a column. Cheap sanity check for a
/// solver that should never produce anything else.
pub fn is_a_permutation(rows: &[usize], cols: &[usize], n_rows: usize, n_cols: usize) -> bool {
    if rows.len() != cols.len() || rows.len() != n_rows.min(n_cols) {
        return false;
    }
    let mut seen_rows = vec![false; n_rows];
    let mut seen_cols = vec![false; n_cols];
    for (&row, &col) in rows.iter().zip(cols) {
        if row >= n_rows || col >= n_cols || seen_rows[row] || seen_cols[col] {
            return false;
        }
        seen_rows[row] = true;
        seen_cols[col] = true;
    }
    true
}
