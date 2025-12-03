package com.ttennebkram.pipeline;

import java.util.logging.Filter;
import java.util.logging.Logger;

/**
 * Launcher class for the JavaFX application.
 * This is needed because JavaFX Application classes cannot be launched
 * directly from a shaded/uber JAR - we need a non-Application main class.
 */
public class PipelineEditorLauncher {

    private static final String APP_NAME = "OpenCV Pipeline";

    public static void main(String[] args) {
        // Suppress harmless JavaFX "Unsupported configuration" warning when running from classpath
        // This warning occurs because JavaFX isn't running in module mode, but works fine anyway
        suppressJavaFXModuleWarning();

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

    /**
     * Suppress the harmless "Unsupported JavaFX configuration" warning that occurs
     * when JavaFX is loaded from the classpath rather than as a proper module.
     */
    private static void suppressJavaFXModuleWarning() {
        // The warning comes from javafx.graphics module via java.util.logging
        // We filter it out since it's harmless and just noise
        Logger javafxLogger = Logger.getLogger("javafx");
        Filter existingFilter = javafxLogger.getFilter();
        javafxLogger.setFilter(record -> {
            String msg = record.getMessage();
            if (msg != null && msg.contains("Unsupported JavaFX configuration")) {
                return false;  // Suppress this message
            }
            // Apply existing filter if any
            return existingFilter == null || existingFilter.isLoggable(record);
        });
    }
}
