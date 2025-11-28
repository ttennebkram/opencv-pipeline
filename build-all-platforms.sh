#!/bin/bash
# Build opencv-pipeline for all supported platforms
# Each JAR will be created in target/ with platform-specific naming

set -e

PLATFORMS="macos-aarch64 macos-x86_64 linux-x86_64 linux-aarch64 windows-x86_64"

echo "Building opencv-pipeline for all platforms..."
echo ""

# Clean first
mvn clean -q

for platform in $PLATFORMS; do
    echo "Building for $platform..."
    mvn package -P $platform -DskipTests -q
    echo "  -> target/opencv-pipeline-$platform.jar"
done

echo ""
echo "Build complete! JARs created:"
ls -lh target/opencv-pipeline-*.jar 2>/dev/null || echo "No JARs found"
