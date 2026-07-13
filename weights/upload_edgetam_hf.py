"""Validate and optionally upload the LibreEdgeTAM Hugging Face payload.

The default mode is a fully local dry run. Remote mutation requires both the
explicit ``--upload`` switch and an ``HF_TOKEN`` environment variable. The
target repository is intentionally fixed so a typo cannot upload the model to
an unrelated namespace.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from convert_edgetam_weights import (
        OFFICIAL_CHECKPOINT_SHA256,
        OFFICIAL_REVISION,
        OFFICIAL_SOURCE_COMMIT,
        PAYLOAD_FILES,
        REFERENCE_REVISION,
        TARGET_REPO,
        compare_state_dicts,
        load_safetensors_state,
        validate_payload,
        validate_reference_dir,
    )
except ModuleNotFoundError:  # Imported as weights.upload_edgetam_hf.
    from weights.convert_edgetam_weights import (
        OFFICIAL_CHECKPOINT_SHA256,
        OFFICIAL_REVISION,
        OFFICIAL_SOURCE_COMMIT,
        PAYLOAD_FILES,
        REFERENCE_REVISION,
        TARGET_REPO,
        compare_state_dicts,
        load_safetensors_state,
        validate_payload,
        validate_reference_dir,
    )


def validate_model_card(payload: str | Path, *, model_sha256: str) -> None:
    """Require the model card and notice to expose the pinned provenance."""
    directory = Path(payload).resolve()
    readme = (directory / "README.md").read_text(encoding="utf-8")
    notice = (directory / "NOTICE").read_text(encoding="utf-8")
    required_readme_fragments = {
        "license metadata": "license: apache-2.0",
        "library metadata": "library_name: transformers",
        "base model metadata": "base_model: facebook/EdgeTAM",
        "official checkpoint revision": OFFICIAL_REVISION,
        "official checkpoint digest": OFFICIAL_CHECKPOINT_SHA256,
        "official source commit": OFFICIAL_SOURCE_COMMIT,
        "reference revision": REFERENCE_REVISION,
        "reference metadata disclosure": (
            "byte-for-byte from the hash-pinned reference revision"
        ),
        "reference license": "reference repository declares Apache-2.0",
        "actual generated model digest": (
            f"Generated `model.safetensors` SHA-256: `{model_sha256}`"
        ),
    }
    required_notice_fragments = {
        "official checkpoint revision": OFFICIAL_REVISION,
        "official checkpoint digest": OFFICIAL_CHECKPOINT_SHA256,
        "official source commit": OFFICIAL_SOURCE_COMMIT,
        "reference revision": REFERENCE_REVISION,
        "license": "Apache License 2.0",
        "raw checkpoint exclusion": "checkpoint was omitted from this mirror",
        "reference metadata disclosure": (
            "Files copied byte-for-byte after SHA-256 verification"
        ),
        "reference license": (
            "License declared by reference model card: Apache License 2.0"
        ),
        "actual generated model digest": (
            f"Generated model.safetensors SHA-256: {model_sha256}"
        ),
    }
    for label, fragment in required_readme_fragments.items():
        if fragment not in readme:
            raise ValueError(f"README.md is missing {label}: {fragment!r}")
    for label, fragment in required_notice_fragments.items():
        if fragment not in notice:
            raise ValueError(f"NOTICE is missing {label}: {fragment!r}")


def validate_local_upload(
    payload: str | Path, *, reference_dir: str | Path | None = None
) -> dict[str, Any]:
    """Validate an upload payload without contacting Hugging Face."""
    result = validate_payload(payload)
    validate_model_card(payload, model_sha256=result["model_sha256"])
    result["target_repo"] = TARGET_REPO
    result["mode"] = "dry-run"

    if reference_dir is not None:
        reference = validate_reference_dir(reference_dir)
        mirror_state = load_safetensors_state(Path(payload) / "model.safetensors")
        reference_state = load_safetensors_state(reference / "model.safetensors")
        result["exact_reference_tensors"] = compare_state_dicts(
            mirror_state,
            reference_state,
            actual_label="mirror payload",
            expected_label="pinned reference",
        )
    return result


def _remote_repo_exists(api: Any) -> bool:
    """Check repository visibility after the caller has authenticated."""
    from huggingface_hub.errors import RepositoryNotFoundError

    try:
        api.repo_info(repo_id=TARGET_REPO, repo_type="model")
        return True
    except RepositoryNotFoundError:
        # The Hub intentionally uses RepositoryNotFoundError (often HTTP 401)
        # for absent repositories. Authentication is checked independently
        # below before this lookup, so this result is safe to treat as absent.
        return False


def _validate_existing_remote_tree(api: Any) -> None:
    """Refuse updates that would preserve files outside the payload contract."""
    remote_files = set(api.list_repo_files(repo_id=TARGET_REPO, repo_type="model"))
    unexpected = sorted(remote_files - PAYLOAD_FILES)
    if unexpected:
        raise RuntimeError(
            f"{TARGET_REPO} contains files outside the validated payload contract: "
            f"{unexpected}. Refusing an update that would leave stale artifacts."
        )


def upload_payload(
    payload: str | Path,
    *,
    allow_existing: bool,
    token: str,
) -> dict[str, Any]:
    """Create or explicitly update the fixed target repository."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication failed; refusing remote mutation"
        ) from exc
    exists = _remote_repo_exists(api)
    if exists and not allow_existing:
        raise FileExistsError(
            f"{TARGET_REPO} already exists. Refusing to update it without --allow-existing."
        )
    if exists:
        _validate_existing_remote_tree(api)
    if not exists:
        api.create_repo(
            repo_id=TARGET_REPO,
            repo_type="model",
            private=False,
            exist_ok=False,
        )

    commit = api.upload_folder(
        repo_id=TARGET_REPO,
        repo_type="model",
        folder_path=str(Path(payload).resolve()),
        commit_message="Add reproducible LibreEdgeTAM snapshot",
    )
    return {
        "mode": "uploaded",
        "target_repo": TARGET_REPO,
        "updated_existing_repo": exists,
        "commit_url": getattr(commit, "commit_url", None),
        "commit_oid": getattr(commit, "oid", None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--payload",
        type=Path,
        default=Path("_hf_build_LibreEdgeTAM"),
        help="Local mirror payload directory (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help=(
            "Local pinned reference snapshot for exact tensor comparison; "
            "optional for dry-run and required with --upload."
        ),
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help=f"Mutate the fixed remote target {TARGET_REPO}; omitted means local dry-run.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Explicitly permit adding a commit when the fixed target already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.upload and args.reference_dir is None:
        raise SystemExit(
            "--upload requires --reference-dir so every tensor is compared "
            "with the pinned local conversion reference before mutation"
        )
    result = validate_local_upload(args.payload, reference_dir=args.reference_dir)

    if not args.upload:
        if args.allow_existing:
            raise SystemExit("--allow-existing has no effect without --upload")
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("--upload requires the HF_TOKEN environment variable")
    upload_result = upload_payload(
        args.payload,
        allow_existing=args.allow_existing,
        token=token,
    )
    result.update(upload_result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
