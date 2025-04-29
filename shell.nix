{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.python311
    pkgs.python3Packages.virtualenv
    pkgs.python311Packages.notebook
    pkgs.python311Packages.jupyterlab
    pkgs.python311Packages.pip
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.pandas
    pkgs.python311Packages.seaborn
    pkgs.python311Packages.scikit-learn
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
  '';
}
