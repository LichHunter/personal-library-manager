{
  outputs = inputs@{ self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;
        python = pkgs.python312;

        pname = "plm-fast-extraction";
        version = "0.1.0";

        workspace = uv2nix.lib.workspace.loadWorkspace {
          workspaceRoot = ../../../../.;
        };

        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };

        buildSystemWheelOverlay = final: prev: {
          pythonPkgsBuildHost = prev.pythonPkgsBuildHost.overrideScope (
            lib.composeManyExtensions [
              pyproject-build-systems.overlays.wheel
            ]
          );
        };

        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (
          lib.composeManyExtensions [
            pyproject-build-systems.overlays.wheel
            overlay
            buildSystemWheelOverlay
          ]
        );

        venv = pythonSet.mkVirtualEnv "plm-env" workspace.deps.default;

      in {
        packages = rec {
          fast-extraction = venv;

          passwdFile = pkgs.writeTextDir "etc/passwd" ''
            root:x:0:0:root:/root:/bin/bash
            nobody:x:65534:65534:nobody:/nonexistent:/bin/false
          '';
          groupFile = pkgs.writeTextDir "etc/group" ''
            root:x:0:
            nogroup:x:65534:
          '';

          fast-extraction-docker = pkgs.dockerTools.buildLayeredImage {
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
              Entrypoint = [ "${venv}/bin/python" "-m" "plm.extraction.fast.cli" ];
              WorkingDir = "/app";

              Env = [
                "PYTHONUNBUFFERED=1"
                "HOME=/root"
                "USER=root"
                "HF_HUB_OFFLINE=1"
                "TRANSFORMERS_OFFLINE=1"
              ];

              Volumes = {
                "/data/input" = {};
                "/data/output" = {};
                "/data/logs" = {};
                "/data/models" = {};
              };
            };
          };

          default = fast-extraction;
        };

        devShell = pkgs.mkShell {
          buildInputs = [ venv ];
          shellHook = ''
            echo "PLM Fast Extraction development shell"
          '';
        };
      }
    );
}
