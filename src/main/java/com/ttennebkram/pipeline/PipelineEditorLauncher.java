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

    private static final String APP_NAME = "OpenCV Pipeline";

    public static void main(String[] args) {
        // Set macOS application name (must be done before any AWT/JavaFX initialization)
        // Multiple properties are needed for different Java/macOS versions
        System.setProperty("apple.awt.application.name", APP_NAME);
        System.setProperty("com.apple.mrj.application.apple.menu.about.name", APP_NAME);
        // For Java 9+ and JavaFX
        System.setProperty("jdk.gtk.version", "2"); // Helps with some display issues

        // Load OpenCV native library
        nu.pattern.OpenCV.loadLocally();

        // Launch the JavaFX application
        PipelineEditorApp.main(args);
    }
}
