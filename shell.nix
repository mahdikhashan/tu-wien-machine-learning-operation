{ pkgs ? import <nixpkgs> {} }:

with pkgs;

mkShell {
  buildInputs = [
    pkgs.python311
    # pkgs.python3Packages.virtualenv # python311 includes venv module, this might not be strictly needed
    pkgs.python311Packages.notebook
    pkgs.python311Packages.jupyterlab
    pkgs.python311Packages.pip
    pkgs.python311Packages.matplotlib
    pkgs.python311Packages.pandas
    pkgs.python311Packages.seaborn
    pkgs.python311Packages.scikit-learn
  ];

  shellHook = ''
    echo "Setting up Python virtual environment..."
    python --version

    VENV_DIR="venv" # Define variable for clarity

    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR..."
        python -m venv "$VENV_DIR" # Use python -m venv
    else
        echo "Virtual environment $VENV_DIR already exists."
    fi

    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"

    echo "Installing base requirements..."
    # Ensure pip and ipykernel are up-to-date and installed within the venv
    pip install --upgrade pip ipykernel

    # Install your project requirements
    if [ -f "tests/requirements-test.txt" ]; then
        echo "Installing requirements from tests/requirements-test.txt..."
        pip install -r tests/requirements-test.txt
    else
        echo "Warning: tests/requirements-test.txt not found."
    fi

    # Install a Jupyter kernel spec specifically for this venv
    # Check if kernel already exists to avoid duplicates on subsequent shell entries
    KERNEL_NAME="nix-shell-venv"
    if ! jupyter kernelspec list | grep -q "$KERNEL_NAME"; then
      echo "Installing Jupyter kernel spec '$KERNEL_NAME'..."
      python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="Python 3 (nix-shell venv)"
    else
      echo "Jupyter kernel spec '$KERNEL_NAME' already installed."
    fi

    echo "Setup complete. Virtual environment '$VENV_DIR' is active."
    echo "Run 'jupyter lab' or 'jupyter notebook'."
    echo "Ensure you select the kernel: 'Python 3 (nix-shell venv)' in Jupyter."
    # Optionally: print python executable path to confirm it's the venv one
    echo "Current Python executable: $(which python)"
  '';
}
