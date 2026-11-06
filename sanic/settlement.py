from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping


if TYPE_CHECKING:
    from sanic.handlers.directory import DirectoryHandler
    from sanic.models.futures import FutureStatic
    from sanic.request import Request
    from sanic.response import BaseHTTPResponse


class DispatchSettlementError(Exception):
    """Raised when a dispatch settlement is declared or advanced
    without the evidence required to settle it."""


class SettlePlane:
    """Frozen plane tags shared by every carrier that participates in a
    dispatch settlement.

    The dynamic-route plane (``Router.add`` / ``Router.get``) and the
    static plane (``app.static`` / file, stream, range, index and
    directory exits) are deliberately distinct planes.
    """

    DYNAMIC = "dynamic"
    STATIC = "static"


class SettleLabel:
    """Frozen plane labels describing *which exit* a request settled at.

    Dynamic plane:

    * ``VERSIONED`` - resolved through a route registered with a version
      (its uri was prefixed with ``version_prefix + version``).
    * ``UNVERSIONED`` - resolved through a route registered without a
      version.
    * ``NOT_FOUND`` - ``Router.resolve`` had no path match.
    * ``METHOD_NOT_ALLOWED`` - the path matched, but the method was not in
      the allowed set.

    Static plane:

    * ``FILE`` - one-shot ``file`` body.
    * ``FILE_STREAM`` - ``file_stream`` transport.
    * ``NOT_MODIFIED`` - conditional ``If-Modified-Since`` terminal 304.
    * ``CONTENT_RANGE`` - ``ContentRangeHandler`` cut a byte window (206).
    * ``INDEX`` - ``DirectoryHandler`` served a configured index file.
    * ``DIRECTORY_VIEW`` - ``DirectoryHandler`` rendered the listing.
    """

    VERSIONED = "versioned"
    UNVERSIONED = "unversioned"
    NOT_FOUND = "not_found"
    METHOD_NOT_ALLOWED = "method_not_allowed"

    FILE = "file"
    FILE_STREAM = "file_stream"
    NOT_MODIFIED = "not_modified"
    CONTENT_RANGE = "content_range"
    INDEX = "index"
    DIRECTORY_VIEW = "directory_view"

    STATIC_EXITS = frozenset(
        {
            FILE,
            FILE_STREAM,
            NOT_MODIFIED,
            CONTENT_RANGE,
            INDEX,
            DIRECTORY_VIEW,
        }
    )
    DYNAMIC_EXITS = frozenset(
        {VERSIONED, UNVERSIONED, NOT_FOUND, METHOD_NOT_ALLOWED}
    )


# Per-request recorder. Bound on ``http.routing.before`` (inside the
# request task, so it is transport agnostic) and released on
# ``http.lifecycle.response``. Defaults to ``None`` so that every carrier
# is a no-op when no settlement session is active.
_current_recorder: ContextVar[Any] = ContextVar(
    "sanic_dispatch_settlement_recorder", default=None
)


def current_probe() -> _ProbeRecorder | None:
    return _current_recorder.get()


def settle_note(plane: str, label: str, /, **details: Any) -> None:
    """Record a plane label for the in-flight probe.

    No-op outside of an active settlement request, which is what keeps the
    pinned behaviour untouched when no settlement has been declared.
    """
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.note(plane, label, **details)


def settlement_of(obj: Any) -> DispatchSettlement | None:
    return getattr(obj, "_dispatch_settlement", None)


@dataclass(frozen=True)
class RouteDeclaration:
    """Evidence of a dynamic route registration observed at ``Router.add``."""

    uri: str
    composed_uri: str
    version: str | None
    version_prefix: str
    methods: tuple[str, ...] | None
    static: bool


@dataclass(frozen=True)
class StaticDeclaration:
    """Evidence of an ``app.static`` registration observed at
    ``StaticMixin.static`` / ``_apply_static``."""

    uri: str
    target: str
    resource_type: str
    use_modified_since: bool
    use_content_range: bool
    stream_large_files: bool | int
    index: tuple[str, ...]
    directory_view: bool
    directory_handler: DirectoryHandler


@dataclass(frozen=True)
class FrozenVerdict:
    """An immutable settlement verdict.

    ``status`` and ``body`` are snapshotted from the concrete response, so
    advancing the settlement (and later re-calling ``Router.get``,
    mutating static options or registering another version) can never
    write through to a frozen verdict.
    """

    sequence: int
    method: str
    path: str
    plane: str
    label: str
    status: int
    body: bytes
    details: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def is_dynamic(self) -> bool:
        return self.plane == SettlePlane.DYNAMIC

    @property
    def is_static(self) -> bool:
        return self.plane == SettlePlane.STATIC


@dataclass(frozen=True)
class SettlementGeneration:
    """A frozen ledger of verdicts captured before the settlement advanced."""

    generation: int
    verdicts: tuple[FrozenVerdict, ...]

    @property
    def labels(self) -> frozenset[tuple[str, str]]:
        return frozenset((v.plane, v.label) for v in self.verdicts)


class _ProbeRecorder:
    """Mutable, per-request accumulation. Never shared with frozen
    verdicts: freezing copies the data out."""

    __slots__ = (
        "session",
        "sequence",
        "method",
        "path",
        "plane",
        "label",
        "details",
        "trace",
        "_token",
    )

    def __init__(
        self,
        session: DispatchSettlementSession,
        sequence: int,
        method: str,
        path: str,
    ) -> None:
        self.session = session
        self.sequence = sequence
        self.method = method
        self.path = path
        self.plane: str | None = None
        self.label: str | None = None
        self.details: dict[str, Any] = {}
        self.trace: list[tuple[str, str]] = []
        self._token: Any = None

    def note(self, plane: str, label: str, **details: Any) -> None:
        self.trace.append((plane, label))
        self.plane = plane
        self.label = label
        if details:
            self.details.update(details)

    def freeze(self, status: int, body: bytes) -> FrozenVerdict:
        if self.plane is None or self.label is None:
            raise DispatchSettlementError(
                f"Probe {self.method} {self.path} did not settle on any "
                f"plane (trace={self.trace!r})."
            )
        return FrozenVerdict(
            sequence=self.sequence,
            method=self.method,
            path=self.path,
            plane=self.plane,
            label=self.label,
            status=status,
            body=bytes(body),
            details=MappingProxyType(dict(self.details)),
        )


class DispatchSettlementSession:
    """One settlement session over one application.

    Probes are sequential. ``advance`` freezes the current generation's
    verdicts; later probes form the next generation and settle against the
    registrations that are live *at that time*.
    """

    def __init__(self, settlement: DispatchSettlement) -> None:
        self._settlement = settlement
        self.generation = 0
        self.verdicts: list[FrozenVerdict] = []
        self.generations: tuple[SettlementGeneration, ...] = ()
        self._sequence = 0
        self._closed = False

    @property
    def app(self):
        return self._settlement.app

    @property
    def labels(self) -> frozenset[tuple[str, str]]:
        return frozenset((v.plane, v.label) for v in self.verdicts)

    def issue_recorder(self, request: Request) -> _ProbeRecorder:
        self._sequence += 1
        return _ProbeRecorder(
            self,
            self._sequence,
            request.method,
            request.path,
        )

    def probe(
        self,
        path: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        *,
        client: Any = None,
    ) -> FrozenVerdict:
        """Drive one request through the application and freeze its verdict
        against the concrete status and body."""
        if self._closed:
            raise DispatchSettlementError(
                "Cannot probe a closed settlement session."
            )
        if self._settlement.active_session is not self:
            raise DispatchSettlementError(
                "This settlement session is not the active session."
            )

        test_client = client or self.app.test_client
        self._settlement.begin_probe(self)
        try:
            _, response = getattr(test_client, method.lower())(
                path, headers=headers
            )
        finally:
            recorder = self._settlement.end_probe(self)
        if response is None:
            raise DispatchSettlementError(
                f"No response returned for probe {method} {path}."
            )

        body = getattr(response, "body", None)
        if body is None:
            body = getattr(response, "content", b"")
        verdict = recorder.freeze(response.status, body)
        self.verdicts.append(verdict)
        return verdict

    def _coverage(self, verdicts: list[FrozenVerdict]) -> list[str]:
        labels = {(v.plane, v.label) for v in verdicts}
        dynamic = {
            label
            for plane, label in labels
            if plane == SettlePlane.DYNAMIC
        }
        static = {
            label
            for plane, label in labels
            if plane == SettlePlane.STATIC
        }

        missing: list[str] = []
        if SettleLabel.VERSIONED not in dynamic:
            missing.append("a versioned dynamic route hit")
        if not (
            SettleLabel.UNVERSIONED in dynamic
            or SettleLabel.NOT_FOUND in dynamic
        ):
            missing.append("an unversioned dynamic route hit or a NotFound")
        if SettleLabel.METHOD_NOT_ALLOWED not in dynamic:
            missing.append("a MethodNotAllowed fork")
        if len(static & SettleLabel.STATIC_EXITS) < 2:
            missing.append(
                "at least two distinct static exits among "
                "file, file_stream, not_modified, content_range, index, "
                "directory_view"
            )
        return missing

    def advance(self) -> SettlementGeneration:
        """Freeze the current generation.

        Raises ``DispatchSettlementError`` unless the session observed, on
        the same application, a versioned hit, an unversioned hit or a
        NotFound, a MethodNotAllowed, and at least two distinct static
        exits.
        """
        missing = self._coverage(self.verdicts)
        if missing:
            raise DispatchSettlementError(
                "Cannot advance dispatch settlement; the session is still "
                "missing: " + "; ".join(missing)
            )
        generation = SettlementGeneration(
            self.generation, tuple(self.verdicts)
        )
        self.generations += (generation,)
        self.verdicts = []
        self.generation += 1
        return generation

    def close(self) -> SettlementGeneration | None:
        """Freeze any remaining verdicts (coverage enforced) and deactivate
        the session."""
        final: SettlementGeneration | None = None
        if self.verdicts:
            final = self.advance()
        self._settlement.deactivate(self)
        self._closed = True
        return final

    def abandon(self) -> None:
        """Deactivate without advancing; live verdicts are discarded."""
        self._settlement.deactivate(self)
        self._closed = True

    def __enter__(self) -> DispatchSettlementSession:
        return self

    def __exit__(self, *exc: Any) -> None:
        if not self._closed:
            self.close()


class DispatchSettlement:
    """The declaration attached to an application via
    ``app.declare_dispatch_settlement()``.

    It is the only shared object between the disconnected planes:
    registration evidence arrives from ``Router.add`` and
    ``StaticMixin.static``; per-request labels arrive through the recorder
    bound by lifecycle signals; sessions pair them with concrete response
    status and body and freeze them.
    """

    def __init__(self, app: Any) -> None:
        self.app = app
        self.route_declarations: list[RouteDeclaration] = []
        self.static_declarations: list[StaticDeclaration] = []
        self.active_session: DispatchSettlementSession | None = None
        self._live: _ProbeRecorder | None = None
        self._install()

    def _install(self) -> None:
        self.app.add_signal(self._before_routing, "http.routing.before")
        self.app.add_signal(self._on_response, "http.lifecycle.response")

    # --------------------------------------------------------------- #
    # Registration evidence
    # --------------------------------------------------------------- #
    def declare_route(
        self,
        *,
        uri: str,
        composed_uri: str,
        version: str | None,
        version_prefix: str,
        methods: Any,
        static: bool,
    ) -> None:
        if static:
            return
        method_tuple = tuple(sorted(methods)) if methods else None
        self.route_declarations.append(
            RouteDeclaration(
                uri=uri,
                composed_uri=composed_uri,
                version=version,
                version_prefix=version_prefix,
                methods=method_tuple,
                static=static,
            )
        )

    def declare_static(
        self, static: FutureStatic, resource_type: str
    ) -> None:
        handler = static.directory_handler
        self.static_declarations.append(
            StaticDeclaration(
                uri=static.uri,
                target=str(static.file_or_directory),
                resource_type=resource_type,
                use_modified_since=static.use_modified_since,
                use_content_range=static.use_content_range,
                stream_large_files=static.stream_large_files,
                index=tuple(handler.index),
                directory_view=handler.directory_view,
                directory_handler=handler,
            )
        )

    # --------------------------------------------------------------- #
    # Session lifecycle
    # --------------------------------------------------------------- #
    def open_session(self) -> DispatchSettlementSession:
        self._require_planes_registered()
        session = DispatchSettlementSession(self)
        self.active_session = session
        return session

    def deactivate(self, session: DispatchSettlementSession) -> None:
        if self.active_session is session:
            self.active_session = None
            self._live = None

    def _require_planes_registered(self) -> None:
        missing: list[str] = []
        if not any(
            declaration.version is not None
            for declaration in self.route_declarations
        ):
            missing.append("a dynamic route registered with a version")
        if not any(
            declaration.version is None and not declaration.static
            for declaration in self.route_declarations
        ):
            missing.append("a dynamic route registered without a version")
        if not self.static_declarations:
            missing.append("at least one app.static mount")
        if missing:
            raise DispatchSettlementError(
                "A dispatch settlement must be declared on one application "
                "that simultaneously carries: " + "; ".join(missing)
            )

    # --------------------------------------------------------------- #
    # Per-request binding (signal handlers)
    # --------------------------------------------------------------- #
    def begin_probe(self, session: DispatchSettlementSession) -> None:
        # The recorder itself is created in ``_before_routing`` so that it
        # lives inside the request task.
        self._live = None

    def end_probe(
        self, session: DispatchSettlementSession
    ) -> _ProbeRecorder:
        recorder = self._live
        self._live = None
        if recorder is None:
            raise DispatchSettlementError(
                "No in-flight probe was recorded; settlement signals did "
                "not fire."
            )
        return recorder

    async def _before_routing(
        self, request: Request, **_ignored: Any
    ) -> None:
        session = self.active_session
        if session is None:
            return
        recorder = session.issue_recorder(request)
        self._live = recorder
        recorder._token = _current_recorder.set(recorder)

    async def _on_response(
        self, request: Request, response: BaseHTTPResponse, **_ignored: Any
    ) -> None:
        recorder = _current_recorder.get()
        if recorder is not None and recorder._token is not None:
            _current_recorder.reset(recorder._token)
            recorder._token = None
