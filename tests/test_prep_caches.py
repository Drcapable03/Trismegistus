from scripts.prep_caches import cache_snapshot, print_prep_report


def test_cache_snapshot_keys():
    snap = cache_snapshot()
    assert "understat" in snap
    assert "statsbomb" in snap
    assert "chaos" in snap


def test_print_prep_report(capsys):
    print_prep_report("Test")
    out = capsys.readouterr().out
    assert "understat xG" in out
    assert "chaos cache" in out