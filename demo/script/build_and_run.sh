#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
VERIFY=0
BUILD_ONLY=0
MODE=""
APP_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --verify) VERIFY=1; shift ;;
    --build-only) BUILD_ONLY=1; shift ;;
    --logs|--telemetry|--debug) MODE="$1"; shift ;;
    --args) shift; APP_ARGS=("$@"); break ;;
    *) APP_ARGS+=("$1"); shift ;;
  esac
done
pkill -x LLVNRDemo 2>/dev/null || true
swift build -c release --product LLVNRDemo -j 2
BIN_DIR="$(swift build -c release --show-bin-path)"
APP="$PROJECT_DIR/dist/LLVNR Demo.app"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/LLVNRDemo" "$APP/Contents/MacOS/LLVNRDemo"
/usr/bin/strip -x "$APP/Contents/MacOS/LLVNRDemo"
RESOURCE_BUNDLE="LLVNRDemo_LLVNRDemo.bundle"
rm -rf "$APP/Contents/Resources/$RESOURCE_BUNDLE"
ditto "$BIN_DIR/$RESOURCE_BUNDLE" "$APP/Contents/Resources/$RESOURCE_BUNDLE"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleExecutable</key><string>LLVNRDemo</string>
<key>CFBundleIdentifier</key><string>com.llvnr.mac.demo</string>
<key>CFBundleName</key><string>LLVNR Demo</string>
<key>CFBundleVersion</key><string>6</string>
<key>CFBundleShortVersionString</key><string>0.4.0</string>
<key>LSMinimumSystemVersion</key><string>14.0</string>
<key>NSPrincipalClass</key><string>NSApplication</string>
<key>NSCameraUsageDescription</key><string>在本机逐帧运行 LLVNR，测量 1080p30 降噪、延迟和源帧缺口。图像仅在本机处理。</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
codesign --force --sign - "$APP"
if [ "$BUILD_ONLY" -eq 1 ]; then exit 0; fi
if [ "${#APP_ARGS[@]}" -gt 0 ]; then
  open -n "$APP" --args "${APP_ARGS[@]}"
else
  open -n "$APP"
fi
if [ "$MODE" = "--debug" ]; then lldb -n LLVNRDemo; fi
if [ "$VERIFY" -eq 1 ]; then
  for attempt in 1 2 3 4 5; do
    if pgrep -x LLVNRDemo >/dev/null; then
      pgrep -x LLVNRDemo
      exit 0
    fi
    sleep 1
  done
  exit 1
fi
if [ "$MODE" = "--logs" ] || [ "$MODE" = "--telemetry" ]; then
  /usr/bin/log stream --style compact --level info --predicate 'process == "LLVNRDemo"'
fi
