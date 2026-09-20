# Development

## Checks

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
cargo doc --workspace --no-deps --locked
```

The default build uses CPU float32 with Burn Flex. Tests cover request validation, stable distributions, tokenizer boundaries, ragged masks, candidate permutation, batch padding isolation, state dependence, checkpoint integrity, typed-key isolation, gradient flow, and the supervised objective. Run the full `train-demo` command to measure actual learning and saved-model inference.

Do not infer model quality from a passing build. Dataset design, leakage checks beyond exact normalized states, calibration, unseen-domain behavior, and performance all need separate evaluation.

## Useful commands

```sh
cargo run --locked -p assort-cli -- --help
cargo run --locked -p assort-cli -- train --help
cargo run --locked -p assort-cli -- config --preset tiny
cargo run --locked -p assort-cli -- config --preset small --vocab-size 32000
```

Use `--release` for meaningful training/inference timings. Debug builds prioritize diagnostics and can be much slower. Larger vocabularies, padded candidate batches, and long state sequences increase cost substantially.

The `wgpu` feature is an extension hook on `assort-model`, not a GPU command-line switch. To run another backend, instantiate `DecisionModel<YourBackend>` in your Rust application and enable the corresponding Burn feature. Add numerical and device-specific tests before relying on it.

## Scope and ownership

The model, tokenizer adapter, data collator, trainer, and typed API are independent crates so you can replace any layer without rebuilding the whole application interface. Packages use workspace path dependencies and are not published to crates.io.

The baseline deliberately has no HTTP server, distributed trainer, pretrained checkpoint download, teacher API integration, production monitoring, model registry, optimizer-resume support, or automatic publication. Add these when your experiment needs them.

Keep real datasets, model weights, metrics, reports, screenshots, and diagnostic artifacts in ignored local directories or outside the repository.
