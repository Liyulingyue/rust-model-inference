//! Parity for the record decoder's assignment solver — `linear_sum_assignment`.
//!
//! Mirrors `tools/oracle/gliner_boundary/dump_assignment.py`, which calls the
//! reference's own `gliner2.training.matching.linear_sum_assignment` on the same
//! 21 cost matrices. Pure function, so no GGUF and it always runs.
//!
//! The test asserts the exact `(row, col)` pairs, not the total cost. The scalar
//! field assignment's optimum is frequently **not unique** — two candidates with
//! equal probability give two optima of identical cost — and which one comes back
//! decides which span each record field binds. Every solver passes a
//! total-cost-only check, including ones that return visibly different spans.
//!
//! Which solver: the reference prefers SciPy when it imports, adding a sub-ULP
//! lexicographic offset to break ties. The oracle venv has no SciPy, so it takes
//! the internal Jonker-Volgenant path, and that is what the Rust port implements
//! and what these cases pin.

use rust_model_inference::models::gliner_boundary::matching::{
    assignment_cost, is_a_permutation, linear_sum_assignment,
};

const FIXTURE: &str = "tests/fixtures/gliner2.5-base-v1/assignment-golden.json";

fn fixture() -> serde_json::Value {
    let raw = std::fs::read_to_string(FIXTURE)
        .unwrap_or_else(|_| panic!("missing {FIXTURE}; regenerate with dump_assignment.py"));
    serde_json::from_str(&raw).expect("parse assignment-golden.json")
}

const INF: &str = "inf";
const NEG_INF: &str = "-inf";
const NAN: &str = "nan";

/// The fixture spells infinities and NaN as strings so the JSON stays valid and
/// so a case meant to be infinite cannot be silently rounded by the writer.
fn to_matrix(case: &serde_json::Value) -> Vec<Vec<f64>> {
    case["cost"]
        .as_array()
        .expect("cost is a list of rows")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("each cost row is a list")
                .iter()
                .map(|value| match value {
                    serde_json::Value::String(text) if text == INF => f64::INFINITY,
                    serde_json::Value::String(text) if text == NEG_INF => f64::NEG_INFINITY,
                    serde_json::Value::String(text) if text == NAN => f64::NAN,
                    number => number.as_f64().expect("a cost number"),
                })
                .collect()
        })
        .collect()
}

fn usize_list(value: &serde_json::Value) -> Vec<usize> {
    value
        .as_array()
        .expect("a list of indices")
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn assignment_matches_the_reference_exactly() {
    let data = fixture();
    let cases = data["cases"].as_array().expect("cases");
    assert!(cases.len() >= 20, "the fixture should cover the shapes");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let cost = to_matrix(case);
        let want_error = case["error"].as_str();

        let result = linear_sum_assignment(&cost);
        if want_error.is_some() {
            let error = result.expect_err(&format!("{name} should have been rejected"));
            assert!(
                error.contains("NaN"),
                "{name}: error should name the cause, got {error}"
            );
            assert_eq!(
                case["row_ind"].as_array().unwrap().len(),
                0,
                "{name}: a rejected matrix has no assignment"
            );
            continue;
        }

        let (rows, cols) = result.unwrap_or_else(|error| panic!("{name}: {error}"));
        let want_rows = usize_list(&case["row_ind"]);
        let want_cols = usize_list(&case["col_ind"]);
        assert_eq!(rows, want_rows, "{name}: row_ind");
        assert_eq!(cols, want_cols, "{name}: col_ind");

        let n_rows = case["rows"].as_u64().unwrap() as usize;
        let n_cols = case["cols"].as_u64().unwrap() as usize;
        assert!(
            is_a_permutation(&rows, &cols, n_rows, n_cols),
            "{name}: not a permutation: {rows:?} -> {cols:?}"
        );
        assert!(
            rows.windows(2).all(|w| w[0] < w[1]),
            "{name}: row_ind must be strictly ascending, got {rows:?}"
        );

        // The optimum itself, so a regression that keeps the pairs but breaks
        // the arithmetic shows up as a cost mismatch rather than nothing.
        let want_total = match &case["total"] {
            serde_json::Value::String(text) if text == INF => f64::INFINITY,
            serde_json::Value::String(text) if text == NEG_INF => f64::NEG_INFINITY,
            number => number.as_f64().expect("a finite total"),
        };
        let got_total = assignment_cost(&cost, &rows, &cols);
        if want_total.is_finite() {
            assert!(
                (got_total - want_total).abs() < 1e-9,
                "{name}: total {got_total} vs reference {want_total}"
            );
        } else {
            assert!(
                !got_total.is_finite(),
                "{name}: expected a non-finite total, got {got_total}"
            );
        }
    }
}

/// The fixture has to keep exercising the case that a greedy solver would get
/// wrong, or the parity test above degrades into "everything is easy".
#[test]
fn the_fixture_keeps_its_discriminating_cases() {
    let data = fixture();
    let cases = data["cases"].as_array().unwrap();
    let names: Vec<&str> = cases
        .iter()
        .map(|case| case["name"].as_str().unwrap())
        .collect();
    for required in [
        // A contested column where two optimal matchings exist and the choice
        // is the whole point.
        "ties_contested_column",
        // A cost plateau in the tail, same idea at 3x3.
        "ties_plateau_tail",
        // Identical rows: the cost is unique, the assignment is not.
        "ties_identical_rows",
        // All-equal matrices at three sizes.
        "ties_all_equal_2x2",
        "ties_all_equal_3x3",
        "ties_all_equal_4x4",
        // Both rectangular orientations, since the solver transposes a tall
        // matrix and has to swap the roles back.
        "wide_2x4",
        "tall_4x2",
        // A unique-optimum baseline, so the tie cases cannot pass by accident.
        "unique_3x3",
        // Edge cases.
        "empty_0x3",
        "empty_3x0",
        "single_cell",
        "infinite_row",
        "nan_cell_is_rejected",
    ] {
        assert!(
            names.contains(&required),
            "the fixture lost its {required:?} case; it is what keeps this test honest"
        );
    }

    // The contested-column case must really be ambiguous, or it no longer tests
    // anything. Rows 0 and 1 both want column 0 for free, so which one gets it is
    // free too — and it is exactly what a different tie-break would change.
    let contested = cases
        .iter()
        .find(|case| case["name"] == "ties_contested_column")
        .expect("ties_contested_column");
    let cost = to_matrix(contested);
    // `row -> col` for each of the two optimal matchings. Each is a valid
    // assignment (every column used once), and they cost the same.
    let total_of = |row_to_col: [usize; 3]| -> f64 {
        let mut used = [false; 3];
        for &col in &row_to_col {
            assert!(col < 3, "column out of range");
            assert!(!used[col], "column {col} used twice: not an assignment");
            used[col] = true;
        }
        row_to_col
            .iter()
            .enumerate()
            .map(|(row, col)| cost[row][*col])
            .sum()
    };
    let row_gets_contested = total_of([0, 2, 1]);
    let other_gets_contested = total_of([2, 0, 1]);
    assert!(
        (row_gets_contested - other_gets_contested).abs() < 1e-12,
        "ties_contested_column is no longer ambiguous ({row_gets_contested} vs \
         {other_gets_contested}), so it cannot discriminate a tie-break"
    );
    // And the reference's answer must be one of the two, not a third thing.
    let reference: Vec<usize> = usize_list(&contested["col_ind"]);
    assert!(
        reference == vec![0, 2, 1] || reference == vec![2, 0, 1],
        "the reference returned {reference:?}, which is neither optimal matching"
    );
}

#[test]
fn rectangular_shapes_return_min_rows_cols_pairs() {
    let data = fixture();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if case["error"].as_str().is_some() {
            continue;
        }
        let n_rows = case["rows"].as_u64().unwrap() as usize;
        let n_cols = case["cols"].as_u64().unwrap() as usize;
        let expected = n_rows.min(n_cols);
        assert_eq!(
            usize_list(&case["row_ind"]).len(),
            expected,
            "{name}: min(rows, cols) pairs"
        );
    }
}

#[test]
fn ragged_rows_are_rejected() {
    // The reference's `cost_matrix.ndim != 2` check does not catch this, because
    // a ragged list-of-lists is still 2-D to torch's view only if the rows match.
    // Ours refuses it rather than indexing past a short row.
    let ragged = vec![vec![1.0, 2.0], vec![3.0]];
    let error = linear_sum_assignment(&ragged).expect_err("ragged rows must be rejected");
    assert!(
        error.contains("same length"),
        "error should say what is wrong, got {error}"
    );
}
