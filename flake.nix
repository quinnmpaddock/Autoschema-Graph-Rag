{
  description = "Python project for AI graphrag with authoschemaKG";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    base-python-env = {
      url = "path:/home/quinn/CODE_PROJECTS/dev-flakes/python";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, base-python-env, ... }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs {
          inherit system;
          config.permittedInsecurePackages = [
            "python-2.7.18.8"
          ];
        };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }:
        let
          # Define our Python version for consistency.
          python = pkgs.python313;

          # Create a single, self-contained Python environment with our packages.
          pythonWithPackages = python.withPackages (ps: with ps; [
            # Add packages here if using Nix-managed Python deps
            # e.g., ps.torch, ps.sentence-transformers (if available)
          ]);
        in
        {
          default = pkgs.mkShell {
            venvDir = ".venv";
            # Inherit everything from base flake (e.g., venvShellHook)
            inputsFrom = [ base-python-env.devShells.${pkgs.system}.default ];

            packages = with pkgs; [
              # System tools
              neo4j
              cacert
              jdk21
              pythonWithPackages
              gcc.cc.lib   # ✅ Provides libstdc++.so.6
              glibc        # ✅ Required for dynamic linking
              git-filter-repo
            ];

            # ✅ Explicitly expose C++ runtime libraries
            LD_LIBRARY_PATH = "${pkgs.gcc.cc.lib}/lib64:${pkgs.gcc.cc.lib}/lib:${pkgs.glibc}/lib";

            # This hook runs after the venv is activated
            postShellHook = ''
              export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              export NIX_SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            '';
          };
        });
    };
}
