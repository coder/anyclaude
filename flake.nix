{
  description = "AnyClaude development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs = { self, nixpkgs, flake-utils, treefmt-nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        treefmt = treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs = {
            nixpkgs-fmt.enable = true;
            shfmt.enable = true;
            shellcheck.enable = true;
          };
        };
      in
      {
        # Format the source tree
        formatter = treefmt.config.build.wrapper;

        # Check formatting
        checks.formatting = treefmt.config.build.check self;

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            # Primary runtime and package manager
            bun

            # Node.js for compatibility (required by some tools)
            nodejs_22

            # Code quality tools (already included by treefmt)
            treefmt.config.build.wrapper

            # Version control
            git

            # Utilities
            jq
            ripgrep
            bat
          ];

          # Environment variables
          NODE_ENV = "development";
        };
      });
}
