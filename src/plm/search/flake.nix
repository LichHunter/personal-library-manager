{
  # This flake is imported by the main flake - inputs are passed in
  outputs = inputs@{ self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
        python = pkgs.python312;

        pname = "plm-search-service";
        version = "0.1.0";

        # Load workspace from pyproject.toml at repo root
        # Path: src/plm/search/ -> ../../../ = repo root (3 levels up)
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ../../../.;
        };

        # Create overlay from pyproject.toml (for runtime dependencies)
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Override build-system packages to use wheels (fixes PEP 639 license format issues)
        # calver and trove-classifiers use new PEP 639 license format that old setuptools can't parse
        # Use pyproject-build-systems.overlays.wheel instead of default (which uses sdist)
        buildSystemWheelOverlay = final: prev: {
          pythonPkgsBuildHost = prev.pythonPkgsBuildHost.overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel  # Use wheel preference for build systems
            ]
          );
        };

        # Build Python package set
        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel  # Use wheel preference
            overlay
            buildSystemWheelOverlay
          ]
        );

        # Virtual environment with all dependencies
        venv = pythonSet.mkVirtualEnv "plm-search-env" workspace.deps.default;

      in {
        packages = rec {
          # The Python package/venv
          search-service = venv;

          # Create passwd/group files for user identification (needed by torch)
          passwdFile = pkgs.writeTextDir "etc/passwd" ''
            root:x:0:0:root:/root:/bin/bash
            nobody:x:65534:65534:nobody:/nonexistent:/bin/false
          '';
          groupFile = pkgs.writeTextDir "etc/group" ''
            root:x:0:
            nogroup:x:65534:
          '';

          # Docker image
          search-service-docker = pkgs.dockerTools.buildLayeredImage {
            name = pname;
            tag = version;

            contents = [
              venv
              pkgs.dockerTools.caCertificates
              pkgs.coreutils
              pkgs.bash
              passwdFile
              groupFile
            ];

            config = {
              Cmd = [
                "${venv}/bin/uvicorn"
                "plm.search.service.app:app"
                "--host" "0.0.0.0"
                "--port" "8000"
              ];
              WorkingDir = "/app";
              ExposedPorts = {
                "8000/tcp" = {};
              };

              Env = [
                "PYTHONUNBUFFERED=1"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "HOME=/root"
                "USER=root"
                # Default paths - override via docker compose environment
                "INDEX_PATH=/data/index"
                "WATCH_DIR=/data/watch"
                "PROCESS_EXISTING=false"
              ];

              Volumes = {
                "/data/index" = {};  # SQLite DB + BM25S index
                "/data/watch" = {};  # Watched dir for auto-ingestion
              };
            };
          };

          default = search-service;
        };

        devShell = pkgs.mkShell {
          buildInputs = [ venv ];
          shellHook = ''
            echo "PLM Search Service development shell"
          '';
        };
      }
    );
}
