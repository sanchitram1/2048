from training.eval_report import summarize_rollouts


def test_summarize_rollouts_counts_and_variance() -> None:
    scores = [100.0, 200.0, 300.0, 400.0]
    max_tiles = [256, 512, 1024, 2048]
    m = summarize_rollouts(scores, max_tiles)
    assert m["mean_score"] == 250.0
    assert m["median_score"] == 250.0
    assert m["times_reached_256"] == 4
    assert m["times_reached_512"] == 3
    assert m["times_reached_1024"] == 2
    assert m["times_reached_2048"] == 1
    assert m["score_variance"] > 0
