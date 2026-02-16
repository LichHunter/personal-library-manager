{
  # This flake is imported by the main flake - inputs are passed in
  outputs = inputs@{ self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        pname = "plm-slow-extraction";
        version = "0.1.0";

        # Load workspace from pyproject.toml at repo root
        # Path: src/plm/extraction/slow/ -> ../../../../ = repo root
        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ../../../../.;
        };

        # Create overlay from pyproject.toml
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        # Build Python package set
        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (
          pkgs.lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
          ]
        );

        # Virtual environment with all dependencies
        venv = pythonSet.mkVirtualEnv "plm-env" workspace.deps.default;

      in {
        packages = rec {
          # The Python package/venv
          slow-extraction = venv;

          # Docker image
          slow-extraction-docker = pkgs.dockerTools.buildLayeredImage {
            name = pname;
            tag = version;

            contents = [
              venv
              pkgs.dockerTools.caCertificates
              pkgs.coreutils
              pkgs.bash
            ];

            config = {
              Cmd = [ "${venv}/bin/python" "-m" "plm.extraction.cli" ];
              WorkingDir = "/app";

              Env = [
                "PYTHONUNBUFFERED=1"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];

              Volumes = {
                "/data/input" = {};
                "/data/output" = {};
                "/data/logs" = {};
                "/data/vocabularies" = {};
                "/auth" = {};
              };
            };
          };

          default = slow-extraction;
        };

        devShell = pkgs.mkShell {
          buildInputs = [ venv ];
          shellHook = ''
            echo "PLM Slow Extraction development shell"
          '';
        };
      }
    );
}
