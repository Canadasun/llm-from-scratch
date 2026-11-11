from deploy import measure


def test_memory_units_parse():
    assert measure._bytes("512MiB ") == 512 * 1024 ** 2
    assert measure._bytes("1.5GiB") == int(1.5 * 1024 ** 3)
    assert measure._bytes("900kB") == 900_000
    assert measure._bytes("--") == 0


def test_cli_refuses_mutable_tags(tmp_path, capsys):
    try:
        measure.main(["--image", "ghcr.io/x/y:latest", "--expect-checkpoint", "0" * 64,
                      "--json", str(tmp_path / "o.json")])
    except SystemExit as exc:
        assert exc.code == 1
    assert "immutable" in capsys.readouterr().err
