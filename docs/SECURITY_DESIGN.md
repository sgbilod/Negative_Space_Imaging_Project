# Security & Integrity Design (Track E)

## Objectives
Provide verifiable integrity and provenance across the imaging pipeline:
1. Cryptographic hashing for every persisted artifact (raw acquisition, processed image, reconstruction outputs).
2. Digital signatures (Ed25519) for authenticity and tamper detection.
3. Provenance chain linking stage outputs (acquisition → processing → reconstruction → analysis).
4. Lightweight verification workflow callable from CLI / demo script.

## Scope (Phase 1)
- Hash algorithms: SHA-256 (default), optional SHA-512.
- Signing: Ed25519 key pair (fast + modern). Fallback to RSA (2048) possible later.
- Key Storage (dev prototype): In-memory ephemeral keys or local `keys/` directory.
- Metadata Embedding: Add `security` block to acquisition & reconstruction results.
- Tamper Detection: Recompute hash + verify signature; report mismatch.

## Out of Scope (Phase 1)
- Hardware Security Modules
- Encrypted artifact storage
- Key rotation policy automation
- Multi-signature threshold schemes (already separate workflow script)

## Data Model Extensions
### Acquisition Metadata (append section `security`):
```
security: {
  hash_algorithm: "sha256",
  artifact_hash: <hex>,
  signature: <base64>,
  key_id: <identifier>,
  provenance_chain: [ { stage: "acquisition", artifact: <path>, hash: <hex>, timestamp: <iso> } ]
}
```

### Reconstruction Provenance Extension:
Add `security` field inside `provenance`:
```
security: {
  artifacts: [
    { name: "mask_raw", path: <path>, hash: <hex>, signature: <base64> },
    { name: "mask_morphed", path: <path>, hash: <hex>, signature: <base64> },
    { name: "labeled_regions", path: <path>, hash: <hex>, signature: <base64> }
  ],
  key_id: <identifier>,
  hash_algorithm: "sha256"
}
```

## Provenance Chain Strategy
Each stage appends an element referencing previous stage hash:
```
{
  stage: "reconstruction",
  parent_hash: <acquisition_hash>,
  artifact_hash: <current_primary_artifact_hash>,
  timestamp: <iso>,
  linkage_verified: true|false
}
```

## Core Functions (security_module.py)
- `generate_ed25519_keypair() -> (private_key, public_key, key_id)`
- `load_or_create_keys(key_dir="keys")`
- `compute_hash(file_path, algorithm='sha256') -> str`
- `sign_bytes(private_key, data: bytes) -> signature_bytes`
- `verify_signature(public_key, data: bytes, signature: bytes) -> bool`
- `sign_file(private_key, file_path, algorithm='sha256') -> (hash_hex, signature_b64)`
- `verify_file(public_key, file_path, expected_hash, signature_b64) -> bool`
- `build_provenance_entry(stage, artifact_path, parent_hash=None)`

## Integration Plan
1. **Acquisition Service**:
   - After writing raw file: compute hash, sign file.
   - Inject `security` section in metadata.
2. **Advanced Reconstructor**:
   - After saving each artifact: hash & sign.
   - Extend provenance with security bundle.
3. **End-to-End Demo**:
   - Optional flag `--verify-security` to re-verify all artifact signatures.

## Tamper Detection Flow
1. User runs verification command.
2. For each artifact: recompute hash; compare; verify signature.
3. Aggregate results; fail if any mismatch.

## Error Handling
- Hash/Sign failures raise `SecurityError`.
- Verification returns structured report with per-artifact status.

## Testing Strategy
- Unit: hash determinism, signature round-trip, tamper modification fails.
- Integration: acquisition metadata contains valid signatures; reconstruction artifacts verify.
- Negative: modify bytes of an artifact; verification should fail.

## Performance Considerations
- SHA-256 + Ed25519 negligible overhead for artifact sizes (< few MB).
- Batch signing is linear; future optimization: streaming hash + parallel sign.

## Future Enhancements (Phase 2)
- Key rotation schedule & revocation list.
- Multi-signature (threshold) integration unify with secure workflow.
- Merkle tree of artifacts per run.
- Transparency log (append-only ledger).

## Security Caveats (Prototype)
- Ephemeral dev keys; not secure for production.
- Private key stored locally unencrypted.
- No audit log persistence.

## Acceptance Criteria
- All new artifacts (raw + reconstruction) have hash + signature.
- Provenance chain persists across run with parent hash linkage.
- Verification CLI reports success on untampered run.
- Tests pass for tamper detection.
