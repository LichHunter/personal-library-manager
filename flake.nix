{
  description = "Personal Library Manager development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    # uv2nix for Python packaging
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    inputs@{ self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
      linuxSystem = "x86_64-linux";

      # Import slow extraction build flake
      slow-extraction-flake = import ./src/plm/extraction/slow/flake.nix;
      slow-extraction-outputs = slow-extraction-flake.outputs {
        inherit self nixpkgs flake-utils pyproject-nix uv2nix pyproject-build-systems;
      };
      linux-slow-extraction-pkgs = slow-extraction-outputs.packages.${linuxSystem};

      # Import search service build flake
      search-service-flake = import ./src/plm/search/flake.nix;
      search-service-outputs = search-service-flake.outputs {
        inherit self nixpkgs flake-utils pyproject-nix uv2nix pyproject-build-systems;
      };
      linux-search-service-pkgs = search-service-outputs.packages.${linuxSystem};

      # Keep multi-system support for devShells
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      # Packages only for Linux (Docker builds require Linux)
      packages = {
        "${linuxSystem}" = {
          # Slow extraction packages
          slow-extraction = linux-slow-extraction-pkgs.slow-extraction;
          slow-extraction-docker = linux-slow-extraction-pkgs.slow-extraction-docker;

          # Search service packages
          search-service = linux-search-service-pkgs.search-service;
          search-service-docker = linux-search-service-pkgs.search-service-docker;

          # Default
          default = linux-slow-extraction-pkgs.default;
        };
      };

      # DevShells for all systems (preserved from original)
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              opencode
              bun

              python312
              python312Packages.pip
              uv

              ollama
              tmux
            ];

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              "/run/opengl-driver"
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];

            shellHook = ''
              echo "Personal Library Manager dev environment"
              echo "Python $(python --version 2>&1)"

              if [ ! -d ".venv" ]; then
                echo "Creating Python virtual environment..."
                uv venv
              fi

              source .venv/bin/activate

              # Ensure LD_LIBRARY_PATH is set for Python packages with C extensions
              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.zlib
                  pkgs.stdenv.cc.cc.lib
                ]
              }/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            '';
          };
        }
      );
    };
}
