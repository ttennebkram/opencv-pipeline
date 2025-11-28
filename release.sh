#!/bin/bash
# Build all platforms and create a GitHub release
# Usage: ./release.sh <version>
# Example: ./release.sh 1.5.0

set -e

VERSION=${1:-}

if [ -z "$VERSION" ]; then
    echo "Usage: ./release.sh <version>"
    echo "Example: ./release.sh 1.5.0"
    exit 1
fi

echo "Building opencv-pipeline v$VERSION for all platforms..."
echo ""

# Build all platforms
./build-all-platforms.sh

echo ""
echo "Creating GitHub release v$VERSION..."

# Create the release and upload all JARs
gh release create "v$VERSION" \
  target/opencv-pipeline-macos-aarch64.jar \
  target/opencv-pipeline-macos-x86_64.jar \
  target/opencv-pipeline-linux-x86_64.jar \
  target/opencv-pipeline-linux-aarch64.jar \
  target/opencv-pipeline-windows-x86_64.jar \
  --title "v$VERSION" \
  --notes "## What's New

- **Nested pipelines** - Create reusable pipeline containers for modular designs
- **Negative backpressure** - Automatically compensates for slow nodes to maintain smooth pipeline flow
- **JTransforms FFT filters** - High-pass and low-pass frequency domain filtering with Butterworth smoothing
- **Cross-platform support** - Pre-built JARs for macOS, Linux, and Windows

## Downloads

| Platform | JAR |
|----------|-----|
| macOS Apple Silicon | opencv-pipeline-macos-aarch64.jar |
| macOS Intel | opencv-pipeline-macos-x86_64.jar |
| Linux x86_64 | opencv-pipeline-linux-x86_64.jar |
| Linux ARM64 (Raspberry Pi) | opencv-pipeline-linux-aarch64.jar |
| Windows x64 | opencv-pipeline-windows-x86_64.jar |

## Running

\`\`\`bash
# macOS (required flag)
java -XstartOnFirstThread -jar opencv-pipeline-macos-aarch64.jar

# Linux
java -jar opencv-pipeline-linux-x86_64.jar

# Windows
java -jar opencv-pipeline-windows-x86_64.jar
\`\`\`

Requires Java 17+."

echo ""
echo "Release v$VERSION created successfully!"
echo "View at: https://github.com/ttennebkram/opencv-pipeline/releases/tag/v$VERSION"
