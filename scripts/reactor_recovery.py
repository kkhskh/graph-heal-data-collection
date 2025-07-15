#!/usr/bin/env python3
"""Reactor-specific recovery adapter.

This module maps the *logical* recovery actions produced by Graph-Heal onto
actual commands sent to a PLC / DCS network (e.g. via OPC-UA).  All methods are
thin wrappers that log the command and – in a real deployment – would perform
a network write.

The implementation here uses *opcua* **if it is available**; otherwise each
method just logs the intended action so you can run the stack end-to-end on a
developer laptop without an OPC server.
"""
from __future__ import annotations

import logging
from typing import Any

try:
    from opcua import Client  # type: ignore
except ImportError:  # pragma: no cover – optional dependency
    Client = None  # type: ignore

_LOG = logging.getLogger(__name__)


class ReactorRecoveryAdapter:  # noqa: D101 – simple I/O wrapper
    def __init__(self, endpoint_url: str | None = None):
        self._client: Any | None = None
        if Client and endpoint_url:
            self._client = Client(endpoint_url)
            self._client.connect()
        elif not Client:
            _LOG.warning("python-opcua not installed – running in *dry-run* mode")
        else:
            _LOG.warning("No OPC-UA endpoint provided – dry-run mode")

    # ------------------------------------------------------------------
    # High-level recovery actions – add/rename to match your YAML policies
    # ------------------------------------------------------------------

    def isolate_valve(self, valve_id: str):
        self._write(f"Valves/{valve_id}/CloseCommand", True)
        _LOG.info("Isolated valve %s", valve_id)

    def trip_pump(self, pump_id: str):
        self._write(f"Pumps/{pump_id}/StopCommand", True)
        _LOG.info("Tripped pump %s", pump_id)

    def start_backup_pump(self, pump_id: str):
        self._write(f"Pumps/{pump_id}/StartBackup", True)
        _LOG.info("Started backup pump %s", pump_id)

    def reroute_flow(self, loop_id: str):
        self._write(f"Loops/{loop_id}/SwitchCommand", 1)
        _LOG.info("Rerouted flow via loop %s", loop_id)

    def shutdown_reactor(self):
        self._write("Reactor/ControlRodDrive/TripAll", True)
        _LOG.critical("Emergency reactor shutdown triggered")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write(self, node_path: str, value: Any):
        if not self._client:
            _LOG.debug("Dry-run write to %s = %s", node_path, value)
            return
        try:
            node = self._client.get_node(f"ns=2;s={node_path}")
            node.set_value(value)
        except Exception as exc:  # noqa: BLE001 – catch-all for field demo
            _LOG.exception("OPC-UA write to %s failed: %s", node_path, exc)

    def close(self):
        if self._client:
            self._client.disconnect() 