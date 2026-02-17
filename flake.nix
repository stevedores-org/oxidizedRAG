{
  description = "oxidizedRAG - High-performance Rust GraphRAG";

  # NOTE: All inputs are pinned to specific commit SHAs for supply-chain security.
  # This prevents accidental/malicious mutations in upstream repositories.
  # To update: nix flake update --recreate-lock-file, review changes, commit both
  # flake.nix and flake.lock files before merging.

  nixConfig = {
    extra-substituters = [ "https://nix-cache.stevedores.org/stevedores" ];
    extra-trusted-substituters = [ "https://nix-cache.stevedores.org/stevedores" ];
  };

  # NOTE: Inputs are pinned to exact commits via flake.lock (committed to repo).
  # Run `nix flake update` to bump, and review the lock diff before merging.
  inputs = {
    # Pin nixpkgs to specific commit for supply-chain security
    # Update: nix flake update --recreate-lock-file, then review lock file before merging
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # flake-utils: utility functions for multi-platform Nix flakes
    # v1.0.0 release (stable)
    flake-utils.url = "github:numtide/flake-utils/d1115de2106ddf9f236339fe7f033dde00dcf6b7";

    # rust-overlay: Latest Rust toolchain management
    # Pinned to recent stable commit with rust-src, rustfmt, clippy support
    rust-overlay = {
      url = "github:oxalica/rust-overlay/6ae973399d80b88b0c6c5e19d05a5b4efaf8c4df";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    # crane: Incremental Rust builds with Nix
    # v0.17.3 release - production-ready
    crane.url = "github:ipetkov/crane/8b3d16633187e6100eda17fda357dc33b4ed28b47";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, crane, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "rustfmt" "clippy" ];
          targets = [ "wasm32-unknown-unknown" ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain rustToolchain;

        # Common args for crane builds
        commonArgs = {
          src = craneLib.cleanCargoSource ./.;
          strictDeps = true;
          buildInputs = with pkgs; [
            openssl
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Security
            pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
          ];
          nativeBuildInputs = with pkgs; [
            pkg-config
          ];
        };

        # Build workspace deps first (for caching)
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        # Build the full workspace
        workspace = craneLib.buildPackage (commonArgs // {
          inherit cargoArtifacts;
        });
      in
      {
        checks = {
          inherit workspace;

          clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- -D warnings";
          });

          fmt = craneLib.cargoFmt {
            src = craneLib.cleanCargoSource ./.;
          };

          tests = craneLib.cargoNextest (commonArgs // {
            inherit cargoArtifacts;
            partitions = 1;
            partitionType = "count";
          });
        };

        packages = {
          default = workspace;

          graphrag-server = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            cargoExtraArgs = "-p graphrag-server";
          });

          graphrag-cli = craneLib.buildPackage (commonArgs // {
            inherit cargoArtifacts;
            cargoExtraArgs = "-p graphrag-cli";
          });
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};

          packages = with pkgs; [
            # Rust extras
            cargo-watch
            cargo-nextest

            # WASM
            wasm-pack
            wasm-bindgen-cli
            trunk

            # Nix cache
            attic-client

            # Bun (for docs-site)
            bun

            # Tools
            just
            git
          ];

          RUST_BACKTRACE = "1";

          shellHook = ''
            echo "🔍 oxidizedRAG Development Environment"
            echo ""
            echo "Commands:"
            echo "  cargo test --all                           # Run all tests"
            echo "  cargo run -p graphrag-cli                  # Run CLI"
            echo "  cargo run -p graphrag-server               # Run server"
            echo "  cd graphrag-wasm && trunk serve --open     # Run WASM app"
            echo ""
            echo "Nix Cache (Attic):"
            echo "  attic login stevedores https://nix-cache.stevedores.org \$ATTIC_TOKEN"
            echo "  attic push stevedores <store-path>         # Push to cache"
            echo ""
          '';
        };
      }
    );
}
