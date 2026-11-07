"""URL policy and a loopback SOCKS5 gateway for the controlled browser."""

from __future__ import annotations

import ipaddress
import re
import select
import socket
from contextlib import contextmanager
from socketserver import BaseRequestHandler, ThreadingTCPServer
from threading import Event, Lock, Thread
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit


BROWSER_TOOL_NAMES = frozenset({"open_browser", "open_url", "get_current_page", "search_web", "get_page_content"})


class InvalidBrowserURL(ValueError):
    pass


class BlockedBrowserDestination(ValueError):
    pass


_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", re.ASCII)
_TRANSITION_NETWORKS = tuple(map(ipaddress.ip_network, (
    "64:ff9b::/96", "64:ff9b:1::/48", "2001::/32", "2002::/16",
)))


def _check_address(address: str) -> None:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    if (
        not ip.is_global or ip.is_multicast or ip.is_reserved
        or (ip.version == 6 and any(ip in network for network in _TRANSITION_NETWORKS))
    ):
        raise BlockedBrowserDestination("Browser destinations must use public unicast addresses")


def _validate_host(host: str) -> str:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        try:
            host = host.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise InvalidBrowserURL("Invalid hostname") from exc
        labels = host.split(".")
        if len(host) > 253 or any(not _LABEL.fullmatch(label) for label in labels):
            raise InvalidBrowserURL("Invalid hostname")
        # Chromium recognizes legacy numeric IPv4 spellings that ipaddress rejects.
        if re.fullmatch(r"[0-9]+|0x[0-9a-f]+", labels[-1]):
            raise InvalidBrowserURL("Invalid numeric hostname")
        if len(labels) == 1 or host.endswith((".localhost", ".local", ".internal", ".home.arpa")):
            raise BlockedBrowserDestination("Local browser destinations are blocked")
    else:
        _check_address(host)
    return host


def validate_browser_url(url: str) -> None:
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url) or "\\" in url:
        raise InvalidBrowserURL("Invalid URL characters")
    try:
        parsed = urlsplit(url)
        host, port = parsed.hostname, parsed.port
    except ValueError as exc:
        raise InvalidBrowserURL("Invalid URL authority or port") from exc
    if (
        parsed.scheme not in {"http", "https"} or not host
        or any(c in parsed.netloc for c in "@%")
        or parsed.netloc.endswith(":") or port == 0
        or not re.fullmatch(r"(?:\[[^\[\]]+\]|[^:\[\]]+)(?::[0-9]+)?", parsed.netloc)
    ):
        raise InvalidBrowserURL("Expected an absolute HTTP or HTTPS URL without credentials")
    if parsed.netloc.startswith("[") and ":" not in host:
        raise InvalidBrowserURL("Bracketed hosts must be IPv6 literals")
    _validate_host(host)


def safe_browser_url(url: str) -> str:
    """Expose only scheme, authority, and path. Query names do not imply safety."""
    try:
        parsed = urlsplit(url)
        authority = parsed.netloc.rsplit("@", 1)[-1]
        if parsed.scheme not in {"http", "https"}:
            return "about:blank" if url == "about:blank" else "[REDACTED]"
        validate_browser_url(urlunsplit((parsed.scheme, authority, "/", "", "")))
        return urlunsplit((parsed.scheme, authority, parsed.path, "", ""))
    except (ValueError, UnicodeError):
        return "[REDACTED]"


def safe_browser_title(url: str, title: str | None) -> str | None:
    """Omit titles that reflect values removed from the corresponding URL."""
    if not title:
        return title
    try:
        parsed = urlsplit(url)
        values = [value for _, value in parse_qsl(parsed.query.replace(";", "&"))]
        values.extend(unquote(value) for value in (parsed.username, parsed.password, parsed.fragment) if value)
        for part in re.split(r"[&;]", parsed.query):
            key, separator, value = part.partition("=")
            values.append(value if separator else unquote(key))
    except (ValueError, UnicodeError):
        return None
    folded = title.casefold()
    return None if any(value and value.casefold() in folded for value in values) else title


def connect_public(host: str, port: int) -> socket.socket:
    """Validate every DNS answer, then connect to a checked numeric sockaddr."""
    host = _validate_host(host)
    if not 1 <= port <= 65535:
        raise InvalidBrowserURL("Invalid destination port")
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    for _, _, _, _, address in addresses:
        _check_address(address[0])
    for family, kind, protocol, _, address in addresses:
        upstream = socket.socket(family, kind, protocol)
        try:
            upstream.settimeout(5)
            upstream.connect(address)
            return upstream
        except OSError:
            upstream.close()
    raise OSError("Public destination was unreachable")


def _read(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError
        data.extend(chunk)
    return bytes(data)


class _SocksHandler(BaseRequestHandler):
    def handle(self) -> None:
        try:
            with self.server.track(self.request):
                self._serve_client(self.request)
        except OSError:
            return

    def _serve_client(self, client: socket.socket) -> None:
        client.settimeout(5)
        try:
            version, count = _read(client, 2)
            if version != 5 or 0 not in _read(client, count):
                return
            client.sendall(b"\x05\x00")
            version, command, reserved, address_type = _read(client, 4)
            if (version, command, reserved) != (5, 1, 0):
                raise ValueError("Only SOCKS5 TCP CONNECT is supported")
            if address_type == 3:
                host = _read(client, _read(client, 1)[0]).decode("ascii")
            elif address_type in (1, 4):
                host = str(ipaddress.ip_address(_read(client, 4 if address_type == 1 else 16)))
            else:
                raise ValueError("Invalid address type")
            port = int.from_bytes(_read(client, 2), "big")
            upstream = connect_public(host, port)
        except (OSError, EOFError, ValueError):
            try:
                client.sendall(b"\x05\x02\x00\x01" + bytes(6))
            except OSError:
                pass
            return
        with upstream, self.server.track(upstream):
            try:
                client.sendall(b"\x05\x00\x00\x01" + bytes(6))
                peers = {client: upstream, upstream: client}
                while not self.server.stopping.is_set():
                    ready, _, _ = select.select(list(peers), [], [], 0.25)
                    for source in ready:
                        chunk = source.recv(65536)
                        if not chunk:
                            return
                        peers[source].sendall(chunk)
            except OSError:
                return


class _GatewayServer(ThreadingTCPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _SocksHandler)
        self.stopping = Event()
        self._sockets: set[socket.socket] = set()
        self._lock = Lock()

    @contextmanager
    def track(self, connection: socket.socket):
        with self._lock:
            if self.stopping.is_set():
                raise OSError("Browser gateway is closed")
            self._sockets.add(connection)
        try:
            yield
        finally:
            with self._lock:
                self._sockets.discard(connection)

    def stop_connections(self) -> None:
        with self._lock:
            self.stopping.set()
            for connection in self._sockets:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def handle_error(self, request, client_address) -> None:
        # Request data and transport exception text are private.
        pass


class BrowserGateway:
    """Per-browser TCP proxy. Chromium resolves destinations through SOCKS5."""

    def __init__(self) -> None:
        self._server = _GatewayServer()
        self._thread = Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        try:
            self._thread.start()
        except BaseException:
            self._server.server_close()
            raise

    @property
    def url(self) -> str:
        return f"socks5://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.stop_connections()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
