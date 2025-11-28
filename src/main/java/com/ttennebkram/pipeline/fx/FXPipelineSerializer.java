package com.ttennebkram.pipeline.fx;

import com.google.gson.*;
import javafx.scene.paint.Color;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Handles serialization and deserialization of JavaFX pipeline documents.
 * Saves and loads FXNode and FXConnection objects to/from JSON format.
 */
public class FXPipelineSerializer {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    /**
     * Result of loading a pipeline document.
     */
    public static class PipelineDocument {
        public final List<FXNode> nodes;
        public final List<FXConnection> connections;

        public PipelineDocument(List<FXNode> nodes, List<FXConnection> connections) {
            this.nodes = nodes;
            this.connections = connections;
        }
    }

    /**
     * Save a pipeline to a JSON file.
     */
    public static void save(String path, List<FXNode> nodes, List<FXConnection> connections) throws IOException {
        JsonObject root = new JsonObject();

        // Serialize nodes
        JsonArray nodesArray = new JsonArray();
        for (int i = 0; i < nodes.size(); i++) {
            FXNode node = nodes.get(i);
            JsonObject nodeJson = new JsonObject();

            // Node identity
            nodeJson.addProperty("id", node.id);
            nodeJson.addProperty("index", i);  // For connection reference
            nodeJson.addProperty("type", node.nodeType);
            nodeJson.addProperty("label", node.label);

            // Position and size
            nodeJson.addProperty("x", node.x);
            nodeJson.addProperty("y", node.y);
            nodeJson.addProperty("width", node.width);
            nodeJson.addProperty("height", node.height);

            // State
            nodeJson.addProperty("enabled", node.enabled);

            // Node configuration
            nodeJson.addProperty("hasInput", node.hasInput);
            nodeJson.addProperty("hasDualInput", node.hasDualInput);
            nodeJson.addProperty("outputCount", node.outputCount);

            // Background color
            if (node.backgroundColor != null) {
                nodeJson.addProperty("bgColorR", node.backgroundColor.getRed());
                nodeJson.addProperty("bgColorG", node.backgroundColor.getGreen());
                nodeJson.addProperty("bgColorB", node.backgroundColor.getBlue());
            }

            // Node-type specific properties
            if ("WebcamSource".equals(node.nodeType)) {
                nodeJson.addProperty("cameraIndex", node.cameraIndex);
            }

            nodesArray.add(nodeJson);
        }
        root.add("nodes", nodesArray);

        // Create node index map for connection serialization
        Map<Integer, Integer> nodeIdToIndex = new HashMap<>();
        for (int i = 0; i < nodes.size(); i++) {
            nodeIdToIndex.put(nodes.get(i).id, i);
        }

        // Serialize connections
        JsonArray connectionsArray = new JsonArray();
        for (FXConnection conn : connections) {
            JsonObject connJson = new JsonObject();

            // Source
            int sourceIndex = conn.source != null ? nodeIdToIndex.getOrDefault(conn.source.id, -1) : -1;
            connJson.addProperty("sourceIndex", sourceIndex);
            connJson.addProperty("sourceOutputIndex", conn.sourceOutputIndex);

            // Target
            int targetIndex = conn.target != null ? nodeIdToIndex.getOrDefault(conn.target.id, -1) : -1;
            connJson.addProperty("targetIndex", targetIndex);
            connJson.addProperty("targetInputIndex", conn.targetInputIndex);

            // Free endpoints (when not connected)
            if (conn.source == null) {
                connJson.addProperty("freeSourceX", conn.freeSourceX);
                connJson.addProperty("freeSourceY", conn.freeSourceY);
            }
            if (conn.target == null) {
                connJson.addProperty("freeTargetX", conn.freeTargetX);
                connJson.addProperty("freeTargetY", conn.freeTargetY);
            }

            connectionsArray.add(connJson);
        }
        root.add("connections", connectionsArray);

        // Write to file
        try (FileWriter writer = new FileWriter(path)) {
            GSON.toJson(root, writer);
        }
    }

    /**
     * Load a pipeline from a JSON file.
     */
    public static PipelineDocument load(String path) throws IOException {
        JsonObject root;
        try (FileReader reader = new FileReader(path)) {
            root = JsonParser.parseReader(reader).getAsJsonObject();
        }

        List<FXNode> nodes = new ArrayList<>();
        List<FXConnection> connections = new ArrayList<>();

        // Deserialize nodes
        if (root.has("nodes")) {
            for (JsonElement elem : root.getAsJsonArray("nodes")) {
                JsonObject nodeJson = elem.getAsJsonObject();

                String type = nodeJson.get("type").getAsString();
                String label = nodeJson.has("label") ? nodeJson.get("label").getAsString() : type;
                double x = nodeJson.has("x") ? nodeJson.get("x").getAsDouble() : 0;
                double y = nodeJson.has("y") ? nodeJson.get("y").getAsDouble() : 0;

                // Create node using factory
                FXNode node = FXNodeFactory.createFXNode(type, (int) x, (int) y);

                // Override label if different from default
                node.label = label;

                // Restore position and size
                node.x = x;
                node.y = y;
                if (nodeJson.has("width")) {
                    node.width = nodeJson.get("width").getAsDouble();
                }
                if (nodeJson.has("height")) {
                    node.height = nodeJson.get("height").getAsDouble();
                }

                // Restore state
                if (nodeJson.has("enabled")) {
                    node.enabled = nodeJson.get("enabled").getAsBoolean();
                }

                // Restore node configuration
                if (nodeJson.has("hasInput")) {
                    node.hasInput = nodeJson.get("hasInput").getAsBoolean();
                }
                if (nodeJson.has("hasDualInput")) {
                    node.hasDualInput = nodeJson.get("hasDualInput").getAsBoolean();
                }
                if (nodeJson.has("outputCount")) {
                    node.outputCount = nodeJson.get("outputCount").getAsInt();
                }

                // Restore background color
                if (nodeJson.has("bgColorR") && nodeJson.has("bgColorG") && nodeJson.has("bgColorB")) {
                    node.backgroundColor = Color.color(
                        nodeJson.get("bgColorR").getAsDouble(),
                        nodeJson.get("bgColorG").getAsDouble(),
                        nodeJson.get("bgColorB").getAsDouble()
                    );
                }

                // Restore node-type specific properties
                if ("WebcamSource".equals(type) && nodeJson.has("cameraIndex")) {
                    node.cameraIndex = nodeJson.get("cameraIndex").getAsInt();
                }

                nodes.add(node);
            }
        }

        // Deserialize connections
        if (root.has("connections")) {
            for (JsonElement elem : root.getAsJsonArray("connections")) {
                JsonObject connJson = elem.getAsJsonObject();

                int sourceIndex = connJson.get("sourceIndex").getAsInt();
                int targetIndex = connJson.get("targetIndex").getAsInt();
                int sourceOutputIndex = connJson.has("sourceOutputIndex") ? connJson.get("sourceOutputIndex").getAsInt() : 0;
                int targetInputIndex = connJson.has("targetInputIndex") ? connJson.get("targetInputIndex").getAsInt() : 0;

                FXNode sourceNode = (sourceIndex >= 0 && sourceIndex < nodes.size()) ? nodes.get(sourceIndex) : null;
                FXNode targetNode = (targetIndex >= 0 && targetIndex < nodes.size()) ? nodes.get(targetIndex) : null;

                FXConnection conn;
                if (sourceNode != null && targetNode != null) {
                    // Complete connection
                    conn = new FXConnection(sourceNode, sourceOutputIndex, targetNode, targetInputIndex);
                } else if (sourceNode != null) {
                    // Source-dangling
                    double freeX = connJson.has("freeTargetX") ? connJson.get("freeTargetX").getAsDouble() : 0;
                    double freeY = connJson.has("freeTargetY") ? connJson.get("freeTargetY").getAsDouble() : 0;
                    conn = FXConnection.createFromSource(sourceNode, sourceOutputIndex, freeX, freeY);
                } else if (targetNode != null) {
                    // Target-dangling
                    double freeX = connJson.has("freeSourceX") ? connJson.get("freeSourceX").getAsDouble() : 0;
                    double freeY = connJson.has("freeSourceY") ? connJson.get("freeSourceY").getAsDouble() : 0;
                    conn = FXConnection.createToTarget(targetNode, targetInputIndex, freeX, freeY);
                } else {
                    // Free connection - skip for now as this is rarely useful
                    continue;
                }

                connections.add(conn);
            }
        }

        return new PipelineDocument(nodes, connections);
    }
}
