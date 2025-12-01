package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.processing.ImageProcessor;
import org.opencv.core.Mat;

import java.util.Map;

/**
 * Abstract base class for FX processors.
 * Provides common functionality and helper methods.
 */
public abstract class FXProcessorBase implements FXProcessor {

    /** Reference to the FXNode for live property updates */
    protected FXNode fxNode;

    @Override
    public void setFXNode(FXNode node) {
        this.fxNode = node;
    }

    @Override
    public FXNode getFXNode() {
        return fxNode;
    }

    /**
     * Refresh internal properties from the FXNode's properties map.
     * Call this at the start of process() to pick up live parameter changes.
     * This calls syncFromFXNode if fxNode is set.
     */
    protected void refreshProperties() {
        if (fxNode != null) {
            syncFromFXNode(fxNode);
        }
    }

    /**
     * Override createImageProcessor to automatically refresh properties before each process() call.
     * This enables real-time parameter updates without modifying individual processors.
     */
    @Override
    public ImageProcessor createImageProcessor() {
        return input -> {
            refreshProperties();  // Re-read from node.properties before processing
            return process(input);
        };
    }

    /**
     * Standard null/empty check for input validation.
     * Call at the start of process() method.
     */
    protected boolean isInvalidInput(Mat input) {
        return input == null || input.empty();
    }

    /**
     * Helper to safely get an int from FXNode properties map.
     */
    protected int getInt(Map<String, Object> props, String key, int defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Number) {
                return ((Number) val).intValue();
            }
        }
        return defaultValue;
    }

    /**
     * Helper to safely get a double from FXNode properties map.
     */
    protected double getDouble(Map<String, Object> props, String key, double defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Number) {
                return ((Number) val).doubleValue();
            }
        }
        return defaultValue;
    }

    /**
     * Helper to safely get a boolean from FXNode properties map.
     */
    protected boolean getBoolean(Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) {
                return (Boolean) val;
            }
        }
        return defaultValue;
    }

    /**
     * Helper to safely get a String from FXNode properties map.
     */
    protected String getString(Map<String, Object> props, String key, String defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof String) {
                return (String) val;
            }
        }
        return defaultValue;
    }

    /**
     * Helper to safely get an int from JSON.
     */
    protected int getJsonInt(JsonObject json, String key, int defaultValue) {
        if (json.has(key)) {
            return json.get(key).getAsInt();
        }
        return defaultValue;
    }

    /**
     * Helper to safely get a double from JSON.
     */
    protected double getJsonDouble(JsonObject json, String key, double defaultValue) {
        if (json.has(key)) {
            return json.get(key).getAsDouble();
        }
        return defaultValue;
    }

    /**
     * Helper to safely get a boolean from JSON.
     */
    protected boolean getJsonBoolean(JsonObject json, String key, boolean defaultValue) {
        if (json.has(key)) {
            return json.get(key).getAsBoolean();
        }
        return defaultValue;
    }

    /**
     * Helper to safely get a String from JSON.
     */
    protected String getJsonString(JsonObject json, String key, String defaultValue) {
        if (json.has(key)) {
            return json.get(key).getAsString();
        }
        return defaultValue;
    }
}
