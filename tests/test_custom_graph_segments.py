from tpa_analyzer.plotting.custom_graphs import eligible_annotation_keys, semantic_segment_keys


def test_segment_registry_lists_all_supported_semantic_segments() -> None:
    assert semantic_segment_keys() == [
        "b1_start_to_peak1",
        "peak1_to_b1_end",
        "b1_end_to_b2_start",
        "b2_start_to_peak2",
        "peak2_to_b2_end",
        "modulus_window",
    ]


def test_annotations_are_filtered_by_segment_meaning() -> None:
    assert eligible_annotation_keys("b1_start_to_peak1") == ["hardness_peak1"]
    assert eligible_annotation_keys("b1_end_to_b2_start") == ["adhesiveness"]
    assert eligible_annotation_keys("modulus_window") == ["modulus_window"]
