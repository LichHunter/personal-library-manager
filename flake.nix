{
  description = "Personal Library Manager development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
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
            ];

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              "/run/opengl-driver"
            ];

            shellHook = ''
              echo "Personal Library Manager dev environment"
              echo "Python $(python --version 2>&1)"

              if [ ! -d ".venv" ]; then
                echo "Creating Python virtual environment..."
                uv venv
              fi

              source .venv/bin/activate
              export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            '';
          };
        }
      );
    };
}
