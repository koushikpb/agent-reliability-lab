from typer.testing import CliRunner

from arlab.cli import app


def test_help_lists_commands():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "retrieval-eval", "run", "gate"):
        assert cmd in result.output
