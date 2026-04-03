#!/bin/bash
# Run this when building the Pi image. Creates a desktop icon for the operator.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="/opt/b210_fft"
DESKTOP_FILE="/usr/share/applications/b210-fft.desktop"

echo "Installing B210 FFT app..."

# Copy app to /opt
sudo mkdir -p "$APP_DIR"
sudo cp "$SCRIPT_DIR/b210_fft.py" "$APP_DIR/"
sudo chmod +x "$APP_DIR/b210_fft.py"

# Create system-wide .desktop entry
sudo tee "$DESKTOP_FILE" > /dev/null << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=B210 FFT
Comment=USRP B210 Spectrum Analyzer
Exec=/opt/b210_fft/b210_fft.py
Icon=utilities-system-monitor
Terminal=false
Categories=Development;Electronics;
EOF

sudo chmod +x "$DESKTOP_FILE"

# Create desktop shortcut for default Pi user
for USER_HOME in /home/*; do
    if [ -d "$USER_HOME/Desktop" ]; then
        SHORTCUT="$USER_HOME/Desktop/B210_FFT.desktop"
        sudo cp "$DESKTOP_FILE" "$SHORTCUT"
        sudo chmod +x "$SHORTCUT"
        # Mark as trusted (no prompt on click)
        sudo -u "$(basename "$USER_HOME")" gio set "$SHORTCUT" metadata::trusted true 2>/dev/null || true
        echo "Created desktop shortcut: $SHORTCUT"
    fi
done

echo "Done! Operator can now click 'B210 FFT' on desktop."
