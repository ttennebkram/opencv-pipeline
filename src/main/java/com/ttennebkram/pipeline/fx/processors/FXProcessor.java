package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import com.ttennebkram.pipeline.processing.ImageProcessor;
import org.opencv.core.Mat;

/**
 * Interface for self-contained FX processors.
 * Each processor encapsulates:
 * - Processing logic (OpenCV operations)
 * - Properties dialog UI (JavaFX controls)
 * - Serialization/deserialization (JSON)
 *
 * This replaces the monolithic ProcessorFactory with individual processor classes.
 */
public interface FXProcessor {

    /**
     * Get the node type name (e.g., "GaussianBlur", "CannyEdge").
     * Must match the type name used in FXNodeRegistry.
     */
    String getNodeType();

    /**
     * Get the category for toolbar grouping (e.g., "Blur", "Edges").
     */
    String getCategory();

    /**
     * Get a description of this processor for tooltips.
     * Should include the OpenCV function signature.
     */
    String getDescription();

    /**
     * Process an input image and return the result.
     *
     * @param input The input Mat (do not modify or release)
     * @return The processed output Mat (caller will release)
     */
    Mat process(Mat input);

    /**
     * Create an ImageProcessor lambda for use with ThreadedProcessor.
     * Default implementation wraps the process() method.
     */
    default ImageProcessor createImageProcessor() {
        return this::process;
    }

    /**
     * Build the properties dialog UI.
     * Add controls to the dialog using its helper methods.
     * Set the onOk callback to save values back to this processor.
     *
     * @param dialog The properties dialog to populate
     */
    void buildPropertiesDialog(FXPropertiesDialog dialog);

    /**
     * Check if this processor has configurable properties.
     * If false, no properties dialog will be shown.
     */
    default boolean hasProperties() {
        return true;
    }

    /**
     * Serialize processor-specific properties to JSON.
     * Called when saving the pipeline.
     *
     * @param json The JSON object to add properties to
     */
    void serializeProperties(JsonObject json);

    /**
     * Deserialize processor-specific properties from JSON.
     * Called when loading a pipeline.
     *
     * @param json The JSON object to read properties from
     */
    void deserializeProperties(JsonObject json);

    /**
     * Sync properties from an FXNode's properties map.
     * Used during transition from old Map-based properties.
     *
     * @param node The FXNode to read properties from
     */
    default void syncFromFXNode(FXNode node) {
        // Default: no-op. Subclasses override to read from node.properties
    }

    /**
     * Sync properties back to an FXNode's properties map.
     * Used during transition from old Map-based properties.
     *
     * @param node The FXNode to write properties to
     */
    default void syncToFXNode(FXNode node) {
        // Default: no-op. Subclasses override to write to node.properties
    }
}
