package com.ttennebkram.pipeline;

/**
 * Launcher class for the JavaFX application.
 * This is needed because JavaFX Application classes cannot be launched
 * directly from a shaded/uber JAR - we need a non-Application main class.
 *
 * Note: When running from an uber-jar, JavaFX will print a warning:
 * "Unsupported JavaFX configuration: classes were loaded from 'unnamed module'"
 * This is harmless - the app works fine, it just means JavaFX isn't running
 * in its preferred module configuration.
 */
public class PipelineEditorLauncher {
    public static void main(String[] args) {
        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();

        // Launch the JavaFX application
        PipelineEditorApp.main(args);
    }
}
