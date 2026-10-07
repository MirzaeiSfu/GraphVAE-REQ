from motif_counting.factorbase_count_audit import compare_cp_count_rows


def test_smoothed_added_rows_preserve_observed_counts():
    result = compare_cp_count_rows(
        cp_rows=[("A", 4), ("B", 2)],
        smoothed_rows=[("A", 4), ("B", 2), ("C", 0)],
    )

    assert result["matches"] is True
    assert result["cp_rows"] == 2
    assert result["smoothed_rows"] == 3
    assert result["added_zero_rows"] == 1


def test_smoothed_nonzero_count_mismatch_is_reported():
    result = compare_cp_count_rows(
        cp_rows=[("A", 4)],
        smoothed_rows=[("A", 5)],
    )

    assert result["matches"] is False
    assert result["mismatch_count"] == 1
    assert "CP=4.0 CP_smoothed=5.0" in result["mismatches"][0]
