from scripts.calibrate_intel import format_intel_report, run_static_checks


def test_run_static_checks_pass_without_reddit_creds(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    checks = run_static_checks()
    names = {c.name for c in checks}
    assert "intel_train_policy" in names
    assert "youtube_channels" in names
    assert "calibration_probes" in names
    policy = next(c for c in checks if c.name == "intel_train_policy")
    channels = next(c for c in checks if c.name == "youtube_channels")
    assert policy.passed
    assert channels.passed


def test_format_intel_report_includes_probes():
    from scripts.calibrate_intel import IntelCheck

    checks = [IntelCheck("news_enabled", True, "ok")]
    probes = [{
        "team": "Arsenal", "opponent": "Chelsea", "div": "E0",
        "news_attention": 0.2, "news_sentiment": 0.6,
        "reddit_sentiment": 0.5, "youtube_sentiment": 0.55,
    }]
    text = format_intel_report(checks, probes)
    assert "Arsenal vs Chelsea" in text
    assert "youtube=0.55" in text