{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python3Packages.virtualenv
    pkgs.python312Packages.notebook
    pkgs.python312Packages.pip
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.pandas
    pkgs.python312Packages.seaborn
    pkgs.python312Packages.scikit-learn
  ];

  shellHook = ''
    python --version

    VENV_DIR="venv" # Define variable for clarity

    if [ ! -d "$VENV_DIR" ]; then
        python -m venv "$VENV_DIR" # Use python -m venv
    else
        echo "Virtual environment $VENV_DIR already exists."
    fi

    source "$VENV_DIR/bin/activate"
    
    pip install -r tests/requirements-test.txt
    pip install -r pipelines/requirements.txt
    pip install -r models/requirements.txt
  '';
}
