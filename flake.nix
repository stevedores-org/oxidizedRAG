{
  description = "oxidizedRAG - High-performance Rust GraphRAG";

  nixConfig = {
    extra-substituters = [ "https://nix-cache.stevedores.org/stevedores" ];
    extra-trusted-substituters = [ "https://nix-cache.stevedores.org/stevedores" ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    crane.url = "github:ipetkov/crane";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, crane }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        rustToolchain = pkgs.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" ];
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
          ];

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
