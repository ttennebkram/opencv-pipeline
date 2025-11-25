package com.ttennebkram.pipeline.serialization;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;

/**
 * Interface for nodes that can serialize/deserialize their properties.
 * Each node class implements this to handle its own property persistence.
 */
public interface NodeSerializable {

    /**
     * Serialize node-specific properties to JSON.
     * Common properties (x, y, threadPriority, etc.) are handled by base class.
     *
     * @param json the JSON object to write properties to
     */
    void serializeProperties(JsonObject json);

    /**
     * Deserialize node-specific properties from JSON.
     * Common properties (x, y, threadPriority, etc.) are handled by base class.
     *
     * @param json the JSON object to read properties from
     */
    void deserializeProperties(JsonObject json);

    /**
     * Get the canonical type name for serialization.
     * Default implementation reads from @NodeInfo annotation.
     *
     * @return the type name to store in JSON
     */
    default String getSerializationType() {
        NodeInfo info = getClass().getAnnotation(NodeInfo.class);
        if (info != null) {
            return info.name();
        }
        // Fallback: derive from class name
        String className = getClass().getSimpleName();
        if (className.endsWith("Node")) {
            return className.substring(0, className.length() - 4);
        }
        return className;
    }
}
