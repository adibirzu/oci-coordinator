from __future__ import annotations

from types import SimpleNamespace

from src.mcp.server import auth


def test_get_oci_config_uses_file_for_api_key(monkeypatch):
    monkeypatch.delenv("OCI_CLI_AUTH", raising=False)
    monkeypatch.setattr(
        auth,
        "_load_oci_config_from_file",
        lambda profile=None: {"tenancy": "ocid1.tenancy.oc1..test"},
    )

    config = auth.get_oci_config()
    assert config["tenancy"] == "ocid1.tenancy.oc1..test"


def test_get_oci_config_signer_mode_falls_back_to_file(monkeypatch):
    monkeypatch.setenv("OCI_CLI_AUTH", "resource_principal")
    monkeypatch.setattr(auth, "_create_signer", lambda mode: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(
        auth,
        "_load_oci_config_from_file",
        lambda profile=None: {"tenancy": "ocid1.tenancy.oc1..fallback"},
    )

    config = auth.get_oci_config()
    assert config["tenancy"] == "ocid1.tenancy.oc1..fallback"


def test_resolve_oci_auth_with_signer(monkeypatch):
    signer = SimpleNamespace(region="us-ashburn-1", tenancy_id="ocid1.tenancy.oc1..signer")
    monkeypatch.setenv("OCI_CLI_AUTH", "instance_principal")
    monkeypatch.setattr(auth, "_create_signer", lambda mode: signer)

    config, resolved_signer = auth._resolve_oci_auth()
    assert resolved_signer is signer
    assert config["region"] == "us-ashburn-1"
    assert config["tenancy"] == "ocid1.tenancy.oc1..signer"
