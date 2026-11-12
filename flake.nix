{
  description = "semantic-ai: local text-to-speech and speech-to-text in pure Rust";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { nixpkgs, ... }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAllSystems (pkgs: {
        default = pkgs.mkShell {
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.clippy
            pkgs.rustfmt
            pkgs.rust-analyzer
            # For the CLI's `audio-device` feature: cpal links ALSA on Linux, found through
            # pkg-config. On macOS it uses CoreAudio, which needs nothing extra.
            pkgs.pkg-config
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ pkgs.alsa-lib ];

          # rust-analyzer finds the standard library sources here.
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";
        };
      });
    };
}
