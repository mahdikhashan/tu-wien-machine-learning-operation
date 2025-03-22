{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.python3Packages.virtualenv
  ];

  shellHook = ''
    if [ ! -d "venv" ]; then
       virtualenv venv
    fi
    source venv/bin/activate
    pip install -r tests/requirements-test.txt
  '';
}
