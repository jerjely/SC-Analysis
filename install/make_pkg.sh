#!/bin/bash
# Wrap "dist/SC Analysis.app" in an installer package that puts it in /Applications.
#   bash install/make_pkg.sh <version> <AppleSilicon|Intel> <hostArchitectures>
set -euo pipefail
VERSION="$1"; LABEL="$2"; HOSTARCH="${3:-arm64}"
APP="dist/SC Analysis.app"
ID="hu.pte.volgyilab.scanalysis"
OUT="dist/SC-Analysis-${VERSION}-macOS-${LABEL}.pkg"
WORK="$(mktemp -d)"
[ -d "$APP" ] || { echo "missing $APP"; exit 1; }

mkdir -p "$WORK/root"
ditto "$APP" "$WORK/root/SC Analysis.app"
xattr -cr "$WORK/root/SC Analysis.app" || true
# ad-hoc signature for the whole bundle (Apple silicon refuses unsigned code)
codesign --force --deep --sign - "$WORK/root/SC Analysis.app"

# always install into /Applications (no relocation to an older copy elsewhere)
pkgbuild --analyze --root "$WORK/root" "$WORK/component.plist"
/usr/libexec/PlistBuddy -c "Set :0:BundleIsRelocatable false" "$WORK/component.plist"
pkgbuild --root "$WORK/root" --component-plist "$WORK/component.plist" \
         --identifier "$ID" --version "$VERSION" --install-location /Applications \
         "$WORK/SCAnalysis-component.pkg"

cat > "$WORK/distribution.xml" <<XML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>SC Analysis ${VERSION}</title>
  <options customize="never" require-scripts="false" hostArchitectures="${HOSTARCH}"/>
  <domains enable_localSystem="true"/>
  <volume-check><allowed-os-versions><os-version min="11.0"/></allowed-os-versions></volume-check>
  <choices-outline><line choice="default"><line choice="${ID}"/></line></choices-outline>
  <choice id="default"/>
  <choice id="${ID}" visible="false"><pkg-ref id="${ID}"/></choice>
  <pkg-ref id="${ID}" version="${VERSION}" onConclusion="none">SCAnalysis-component.pkg</pkg-ref>
</installer-gui-script>
XML
productbuild --distribution "$WORK/distribution.xml" --package-path "$WORK" "$OUT"
rm -rf "$WORK"
ls -la "$OUT"
