#!/bin/bash
# Build platform-specific uber-jars for all supported platforms

set -e

echo "Building all platform jars..."

# Clean first
mvn clean -q

# macOS Apple Silicon (default profile)
echo "Building mac-aarch64..."
mvn package -q
cp target/opencv-pipeline.jar target/opencv-pipeline-mac-aarch64.jar

# macOS Intel
echo "Building mac..."
mvn package -Pmac -P'!mac-aarch64' -q
cp target/opencv-pipeline.jar target/opencv-pipeline-mac.jar

# Linux x86_64
echo "Building linux..."
mvn package -Plinux -P'!mac-aarch64' -q
cp target/opencv-pipeline.jar target/opencv-pipeline-linux.jar

# Linux ARM64 (Raspberry Pi)
echo "Building linux-aarch64..."
mvn package -Plinux-aarch64 -P'!mac-aarch64' -q
cp target/opencv-pipeline.jar target/opencv-pipeline-linux-aarch64.jar

# Windows x86_64
echo "Building win..."
mvn package -Pwin -P'!mac-aarch64' -q
cp target/opencv-pipeline.jar target/opencv-pipeline-win.jar

echo ""
echo "Done! Built jars:"
ls -lh target/opencv-pipeline-*.jar
