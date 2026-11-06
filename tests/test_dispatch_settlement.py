"""Acceptance for the declarable dispatch settlement.

The settlement is exercised on **one** application that simultaneously
carries a versioned dynamic route, an unversioned dynamic route and several
static mounts. One session probes, in order, the versioned hit, the
unversioned hit, the NotFound fork, the MethodNotAllowed fork and all six
static exits (file, file_stream, not-modified, content range, index,
directory view), then advances.

After advancing, the frozen verdicts (plane label, status and body) must
not be written through by re-calling ``Router.get``, mutating static
options or registering another version; only new probes settle against the
then-current registration.
"""

from __future__ import annotations

import pytest

from sanic import text
from sanic.exceptions import MethodNotAllowed, NotFound
from sanic.settlement import (
    DispatchSettlementError,
    SettleLabel,
    SettlePlane,
)


pytestmark = pytest.mark.xdist_group(name="process_spawning")

SMALL_BYTES = b"settle-small\n"
BIG_BYTES = b"Z" * 96
INDEX_BYTES = b"<p>index settle</p>"
FUTURE_DATE = "Wed, 21 Oct 2099 07:28:00 GMT"


@pytest.fixture
def settled_app(app, tmp_path):
    # Declare first; registrations that follow are the settlement evidence.
    settlement = app.declare_dispatch_settlement()

    async def plain_handler(request):
        return text("plain settle")

    async def versioned_handler(request):
        return text("version settle")

    app.add_route(plain_handler, "/settle", methods=["GET"])
    app.add_route(
        versioned_handler,
        "/settle",
        methods=["GET"],
        version=1,
        version_prefix="/v",
    )

    small = tmp_path / "small.txt"
    small.write_bytes(SMALL_BYTES)

    big = tmp_path / "big.txt"
    big.write_bytes(BIG_BYTES)

    index_dir = tmp_path / "indexdir"
    index_dir.mkdir()
    (index_dir / "index.html").write_bytes(INDEX_BYTES)

    view_dir = tmp_path / "viewdir"
    view_dir.mkdir()
    (view_dir / "listed.txt").write_bytes(b"listed\n")

    mut_dir = tmp_path / "mutdir"
    mut_dir.mkdir()
    (mut_dir / "inside.txt").write_bytes(b"inside\n")

    app.static("/plain-file", str(small), name="settle_plain_file")
    app.static(
        "/big-file",
        str(big),
        stream_large_files=64,
        name="settle_big_file",
    )
    app.static(
        "/ranged-file",
        str(small),
        use_content_range=True,
        name="settle_ranged_file",
    )
    app.static("/cached-file", str(small), name="settle_cached_file")
    app.static(
        "/index-dir",
        str(index_dir),
        index="index.html",
        name="settle_index_dir",
    )
    app.static(
        "/view-dir",
        str(view_dir),
        directory_view=True,
        name="settle_view_dir",
    )
    # No index and no directory view; only becomes a usable listing after
    # the settlement advanced and the options were changed.
    app.static("/mut-dir", str(mut_dir), name="settle_mut_dir")

    return app, settlement, {
        "small": small,
        "big": big,
        "index_dir": index_dir,
        "view_dir": view_dir,
        "mut_dir": mut_dir,
    }


def _snapshot(generation):
    return tuple(
        (
            verdict.method,
            verdict.path,
            verdict.plane,
            verdict.label,
            verdict.status,
            bytes(verdict.body),
            tuple(sorted(verdict.details.items())),
        )
        for verdict in generation.verdicts
    )


def test_session_probes_all_planes_on_one_app_and_freezes(settled_app):
    app, settlement, _ = settled_app

    # Registration evidence: the versioned and unversioned paths are
    # distinct registrations on the same application, and static mounts
    # were observed on the static plane.
    composed = {
        declaration.uri: declaration
        for declaration in settlement.route_declarations
        if declaration.version is None
    }
    versioned = {
        declaration.composed_uri: declaration
        for declaration in settlement.route_declarations
        if declaration.version is not None
    }
    assert "/settle" in composed
    assert "/v1/settle" in versioned
    assert versioned["/v1/settle"].version == "1"
    assert versioned["/v1/settle"].version_prefix == "/v"
    assert composed["/settle"].composed_uri == "/settle"

    static_kinds = {
        declaration.uri: declaration.resource_type
        for declaration in settlement.static_declarations
    }
    assert static_kinds["/plain-file"] == "file"
    assert static_kinds["/index-dir"] == "dir"
    assert static_kinds["/view-dir"] == "dir"

    session = settlement.open_session()

    # ---- dynamic plane: version / unversion / not-found / 405 ----
    versioned_hit = session.probe("/v1/settle")
    unversioned_hit = session.probe("/settle")
    not_found = session.probe("/no-such-route")
    method_not_allowed = session.probe("/settle", method="POST")

    assert versioned_hit.plane == SettlePlane.DYNAMIC
    assert versioned_hit.label == SettleLabel.VERSIONED
    assert versioned_hit.status == 200
    assert versioned_hit.body == b"version settle"

    assert unversioned_hit.plane == SettlePlane.DYNAMIC
    assert unversioned_hit.label == SettleLabel.UNVERSIONED
    assert unversioned_hit.status == 200
    assert unversioned_hit.body == b"plain settle"

    assert not_found.plane == SettlePlane.DYNAMIC
    assert not_found.label == SettleLabel.NOT_FOUND
    assert not_found.status == 404

    assert method_not_allowed.plane == SettlePlane.DYNAMIC
    assert method_not_allowed.label == SettleLabel.METHOD_NOT_ALLOWED
    assert method_not_allowed.status == 405
    assert "GET" in (method_not_allowed.details.get("allowed_methods") or ())

    # ---- static plane: every terminal exit ----
    whole_file = session.probe("/plain-file")
    streamed_file = session.probe("/big-file")
    not_modified = session.probe(
        "/cached-file", headers={"If-Modified-Since": FUTURE_DATE}
    )
    ranged = session.probe("/ranged-file", headers={"Range": "bytes=2-6"})
    indexed = session.probe("/index-dir/")
    directory_view = session.probe("/view-dir/")

    assert whole_file.plane == SettlePlane.STATIC
    assert whole_file.label == SettleLabel.FILE
    assert whole_file.status == 200
    assert whole_file.body == SMALL_BYTES

    assert streamed_file.plane == SettlePlane.STATIC
    assert streamed_file.label == SettleLabel.FILE_STREAM
    assert streamed_file.status == 200
    assert streamed_file.body == BIG_BYTES

    assert not_modified.plane == SettlePlane.STATIC
    assert not_modified.label == SettleLabel.NOT_MODIFIED
    assert not_modified.status == 304
    assert not_modified.body == b""

    assert ranged.plane == SettlePlane.STATIC
    assert ranged.label == SettleLabel.CONTENT_RANGE
    assert ranged.status == 206
    assert ranged.body == SMALL_BYTES[2:7]
    assert ranged.details["window"] == (2, 6, len(SMALL_BYTES))
    assert ranged.details["transport"] == SettleLabel.FILE

    assert indexed.plane == SettlePlane.STATIC
    assert indexed.label == SettleLabel.INDEX
    assert indexed.status == 200
    assert indexed.body == INDEX_BYTES

    assert directory_view.plane == SettlePlane.STATIC
    assert directory_view.label == SettleLabel.DIRECTORY_VIEW
    assert directory_view.status == 200
    assert b"listed.txt" in directory_view.body

    # ---- advancing freezes the coupled session ----
    generation = session.advance()
    assert generation.generation == 0
    assert len(generation.verdicts) == 10
    assert (SettlePlane.DYNAMIC, SettleLabel.VERSIONED) in generation.labels
    assert (SettlePlane.DYNAMIC, SettleLabel.UNVERSIONED) in generation.labels
    assert (SettlePlane.DYNAMIC, SettleLabel.NOT_FOUND) in generation.labels
    assert (
        SettlePlane.DYNAMIC,
        SettleLabel.METHOD_NOT_ALLOWED,
    ) in generation.labels
    for static_label in SettleLabel.STATIC_EXITS:
        assert (SettlePlane.STATIC, static_label) in generation.labels

    # The live generation starts empty; the frozen verdicts themselves are
    # immutable values.
    assert session.verdicts == []
    with pytest.raises(AttributeError):
        versioned_hit.status = 500
    with pytest.raises(TypeError):
        versioned_hit.details["x"] = 1


def test_advance_requires_the_coupled_coverage(settled_app):
    _, settlement, _ = settled_app
    session = settlement.open_session()

    session.probe("/v1/settle")
    session.probe("/settle")
    session.probe("/settle", method="POST")
    # Only one static exit and no NotFound/unversion coverage is short of
    # the required settlement.
    session.probe("/plain-file")

    with pytest.raises(DispatchSettlementError) as excinfo:
        session.advance()
    message = str(excinfo.value)
    assert "static exits" in message
    session.abandon()


def test_open_session_requires_all_three_planes(app, tmp_path):
    settlement = app.declare_dispatch_settlement()

    async def handler(request):
        return text("only unversioned")

    app.add_route(handler, "/lone", methods=["GET"])

    with pytest.raises(DispatchSettlementError) as excinfo:
        settlement.open_session()
    message = str(excinfo.value)
    assert "version" in message
    assert "app.static" in message

    # Adding the missing planes lets the session open.
    async def versioned_handler(request):
        return text("v")

    app.add_route(
        versioned_handler, "/lone", methods=["GET"], version=1
    )
    static_file = tmp_path / "f.txt"
    static_file.write_bytes(b"f")
    app.static("/f", str(static_file))

    session = settlement.open_session()
    try:
        assert session.generation == 0
    finally:
        session.close()


def test_advanced_settlement_is_not_written_through(settled_app):
    app, settlement, paths = settled_app
    session = settlement.open_session()

    versioned_hit = session.probe("/v1/settle")
    unversioned_hit = session.probe("/settle")
    not_found = session.probe("/no-such-route")
    method_not_allowed = session.probe("/settle", method="POST")
    whole_file = session.probe("/plain-file")
    directory_view = session.probe("/view-dir/")

    generation_zero = session.advance()
    snapshot = _snapshot(generation_zero)

    # (1) Re-calling Router.get directly must not touch frozen verdicts.
    # The router is reset after each test-client server cycle, so finalize
    # it again before resolving outside of a request.
    app.router.finalize()
    for _ in range(3):
        app.router.get("/settle", "GET", None)
    with pytest.raises(NotFound) as not_found_exc:
        app.router.get("/no-such-route", "GET", None)
    with pytest.raises(MethodNotAllowed) as method_exc:
        app.router.get("/settle", "POST", None)
    assert not_found_exc.value.settle_label == SettleLabel.NOT_FOUND
    assert method_exc.value.settle_label == SettleLabel.METHOD_NOT_ALLOWED

    # (2) Changing static options after the freeze must not touch frozen
    # verdicts; the change only applies to new requests.
    mut_declaration = next(
        declaration
        for declaration in settlement.static_declarations
        if declaration.uri == "/mut-dir"
    )
    assert mut_declaration.directory_view is False
    mut_declaration.directory_handler.directory_view = True

    # (3) Registering another version after the freeze.
    async def version_two_handler(request):
        return text("version two settle")

    app.add_route(
        version_two_handler, "/settle", methods=["GET"], version=2
    )

    # Frozen objects retain exactly the original evidence.
    assert versioned_hit.body == b"version settle"
    assert unversioned_hit.label == SettleLabel.UNVERSIONED
    assert not_found.status == 404
    assert method_not_allowed.status == 405
    assert whole_file.body == SMALL_BYTES
    assert directory_view.body != b""

    # The new generation settles against the registrations live *now*.
    version_two = session.probe("/v2/settle")
    assert version_two.plane == SettlePlane.DYNAMIC
    assert version_two.label == SettleLabel.VERSIONED
    assert version_two.status == 200
    assert version_two.body == b"version two settle"

    mutated_view = session.probe("/mut-dir/")
    assert mutated_view.plane == SettlePlane.STATIC
    assert mutated_view.label == SettleLabel.DIRECTORY_VIEW
    assert mutated_view.status == 200
    assert b"inside.txt" in mutated_view.body

    # Re-probing an old path produces a NEW verdict with the same content.
    unversioned_again = session.probe("/settle")
    assert unversioned_again is not unversioned_hit
    assert unversioned_again.status == unversioned_hit.status
    assert unversioned_again.body == unversioned_hit.body

    # Cover the new generation with the remaining required forks and
    # advance it; generation zero must remain byte-for-byte identical.
    session.probe("/no-such-route")
    session.probe("/settle", method="POST")
    session.probe("/plain-file")
    generation_one = session.advance()

    assert generation_one.generation == 1
    assert _snapshot(generation_zero) == snapshot
    assert len(session.generations) == 2
    assert session.generations[0].verdicts is generation_zero.verdicts

    new_paths = {verdict.path for verdict in generation_one.verdicts}
    assert "/v2/settle" in new_paths
    assert "/mut-dir/" in new_paths
    old_paths = {verdict.path for verdict in generation_zero.verdicts}
    assert "/v2/settle" not in old_paths
    assert "/mut-dir/" not in old_paths

    session.close()


def test_pin_behaviour_without_settlement(app, tmp_path):
    assert app.dispatch_settlement is None

    async def handler(request):
        return text("pin ok")

    app.add_route(handler, "/pin", methods=["GET"], version=1)
    static_file = tmp_path / "pin.txt"
    static_file.write_bytes(b"pin body")
    app.static("/pin-file", str(static_file))

    _, response = app.test_client.get("/v1/pin")
    assert response.status == 200
    assert response.body == b"pin ok"

    _, response = app.test_client.get("/pin-file")
    assert response.status == 200
    assert response.body == b"pin body"

    # Router-level forks remain the same exceptions; the fork labels are
    # observable without any declared settlement. The router is finalized
    # again because the test client reset it on shutdown.
    app.router.finalize()
    with pytest.raises(NotFound) as not_found_exc:
        app.router.get("/missing", "GET", None)
    assert not_found_exc.value.settle_plane == SettlePlane.DYNAMIC
    assert not_found_exc.value.settle_label == SettleLabel.NOT_FOUND

    with pytest.raises(MethodNotAllowed) as method_exc:
        app.router.get("/v1/pin", "POST", None)
    assert method_exc.value.settle_label == SettleLabel.METHOD_NOT_ALLOWED
    assert "GET" in (method_exc.value.allowed_methods or ())

    assert app.dispatch_settlement is None
