"""Isolated TCP preflight regressions; no models, API startup or media."""
import errno
import socket
from unittest.mock import MagicMock, Mock

import pytest

from tools import check_backend as diagnostic


@pytest.fixture
def probe(monkeypatch):
    factory = MagicMock()
    probe = factory.return_value.__enter__.return_value
    monkeypatch.setattr(diagnostic.socket, "socket", factory)
    return factory, probe


def test_free_windows_port_avoids_nonblocking_connect(monkeypatch, probe):
    factory, sock = probe
    monkeypatch.setattr(diagnostic.sys, "platform", "win32")
    monkeypatch.setattr(socket, "SO_EXCLUSIVEADDRUSE", -5, raising=False)
    sock.connect_ex.return_value = 10035
    assert diagnostic.port_available(8000) is True
    factory.assert_called_once_with(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt.assert_called_once_with(socket.SOL_SOCKET, -5, 1)
    sock.bind.assert_called_once_with(("0.0.0.0", 8000))
    sock.connect_ex.assert_not_called()
    sock.settimeout.assert_not_called()
    factory.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("code", [errno.EADDRINUSE, 10048])
def test_occupied_port(probe, code):
    factory, sock = probe
    sock.bind.side_effect = OSError(code, "address in use")
    assert diagnostic.port_available(8000) is False
    factory.return_value.__exit__.assert_called_once()


@pytest.mark.parametrize("error", [
    BlockingIOError(10035, "would block"),
    TimeoutError("timed out"),
    OSError(10060, "connection timed out"),
    PermissionError(errno.EACCES, "access denied"),
])
def test_unknown_bind_errors_stop_startup(probe, monkeypatch, capsys, error):
    factory, sock = probe
    sock.bind.side_effect = error
    settings = Mock(side_effect=AssertionError("must stop before config"))
    monkeypatch.setattr(diagnostic, "load_settings", settings)
    assert diagnostic.main(["--skip-api", "--require-port-free"]) == 1
    output = capsys.readouterr().out
    assert "resultado indeterminado" in output
    assert "ocupada" not in output
    settings.assert_not_called()
    factory.return_value.__exit__.assert_called_once()


def test_free_port_continues_diagnostic(probe, monkeypatch, capsys):
    # Reaching config demonstrates that the CLI gate admitted the free port.
    settings = Mock(side_effect=RuntimeError("reached configuration"))
    monkeypatch.setattr(diagnostic, "load_settings", settings)
    diagnostic.main(["--skip-api", "--require-port-free"])
    settings.assert_called_once_with()
    assert "Porta local 8000: livre" in capsys.readouterr().out


@pytest.mark.parametrize("host", ["127.0.0.1", "0.0.0.0"])
@pytest.mark.parametrize("listening", [False, True])
def test_real_bind_is_released_and_detects_bound_socket(host, listening):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((host, 0))
        if listening:
            server.listen(1)
        port = server.getsockname()[1]
        assert diagnostic.port_available(port) is False
    assert diagnostic.port_available(port) is True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", port))


@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 unavailable")
def test_ipv6_only_listener_does_not_block_ipv4():
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as server:
        try:
            server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            server.bind(("::1", 0))
            server.listen(1)
        except OSError as error:
            pytest.skip(f"IPv6 loopback unavailable: {error}")
        assert diagnostic.port_available(server.getsockname()[1]) is True
