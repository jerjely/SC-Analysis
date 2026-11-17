#!/bin/bash
# SC Analysis installer for macOS (double-click in Finder)
cd "$(dirname "$0")"
ENV=sc_analysis
echo "=== SC Analysis installer ==="
CONDA=""
for c in "$CONDA_EXE" "$(command -v conda)" "$HOME/miniconda3/bin/conda" "$HOME/opt/miniconda3/bin/conda" \
         "$HOME/anaconda3/bin/conda" "$HOME/opt/anaconda3/bin/conda" "/opt/homebrew/Caskroom/miniconda/base/bin/conda" \
         "/usr/local/Caskroom/miniconda/base/bin/conda" "/opt/miniconda3/bin/conda"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CONDA="$c"; break; fi
done
if [ -z "$CONDA" ]; then
  echo "Could not find conda. Install Miniconda (https://docs.conda.io/en/latest/miniconda.html) and run again."
  read -p "Press Enter to close"; exit 1
fi
echo "Using conda: $CONDA"
if ! "$CONDA" env list | grep -q "^$ENV "; then
  "$CONDA" create -y -n $ENV --override-channels -c conda-forge python=3.11 || { read -p "Failed. Press Enter"; exit 1; }
fi
PREFIX=$("$CONDA" run -n $ENV python -c "import sys; print(sys.prefix)")
"$PREFIX/bin/python" -m pip install --upgrade pip
"$PREFIX/bin/python" -m pip install -r requirements.txt || { read -p "Failed. Press Enter"; exit 1; }
"$PREFIX/bin/python" -m pip install --no-deps --upgrade . || { read -p "Failed. Press Enter"; exit 1; }
L="$HOME/Desktop/SC Analysis.command"
printf '#!/bin/bash\n"%s/bin/python" -m sc_analysis "$@"\n' "$PREFIX" > "$L"
chmod +x "$L"
echo "Done. Double-click 'SC Analysis.command' on your Desktop."
read -p "Press Enter to close"
