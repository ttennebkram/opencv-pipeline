package com.ttennebkram.pipeline.serialization;

import com.google.gson.*;
import com.ttennebkram.pipeline.model.*;
import com.ttennebkram.pipeline.nodes.*;
import com.ttennebkram.pipeline.registry.NodeRegistry;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.widgets.Canvas;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles serialization and deserialization of pipeline documents.
 * Each node handles its own property serialization via NodeSerializable interface.
 *
 * Connection model: All connections (complete, dangling, free) are stored in a single
 * unified list. Each connection can have:
 * - sourceId: node index or -1 if source is disconnected
 * - targetId: node index or -1 if target is disconnected
 * - freeSourceX/Y: position of source end when disconnected
 * - freeTargetX/Y: position of target end when disconnected
 */
public class PipelineSerializer {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    /**
     * Result of loading a pipeline document.
     */
    public static class PipelineDocument {
        public final List<PipelineNode> nodes;
        public final List<Connection> connections;

        public PipelineDocument(List<PipelineNode> nodes, List<Connection> connections) {
            this.nodes = nodes;
            this.connections = connections;
        }

        // Legacy constructor for backwards compatibility during transition
        @Deprecated
        public PipelineDocument(List<PipelineNode> nodes,
                                List<Connection> connections,
                                List<?> danglingConnections,
                                List<?> reverseDanglingConnections,
                                List<?> freeConnections) {
            this.nodes = nodes;
            this.connections = connections;
        }
    }

    /**
     * Save a pipeline to a JSON file.
     */
    public static void save(String path,
                           List<PipelineNode> nodes,
                           List<Connection> connections) throws IOException {

        JsonObject root = new JsonObject();

        // Serialize nodes
        JsonArray nodesArray = new JsonArray();
        for (int i = 0; i < nodes.size(); i++) {
            PipelineNode node = nodes.get(i);
            JsonObject nodeJson = new JsonObject();

            // Add id for reference
            nodeJson.addProperty("id", i);

            // Common properties (includes type)
            node.serializeCommon(nodeJson);

            // For ProcessingNodes, add the name field for compatibility
            if (node instanceof ProcessingNode) {
                nodeJson.addProperty("name", ((ProcessingNode) node).getName());
            }

            // Node-specific properties (handled by each node)
            node.serializeProperties(nodeJson);

            nodesArray.add(nodeJson);
        }
        root.add("nodes", nodesArray);

        // Serialize connections (unified format)
        JsonArray connectionsArray = new JsonArray();
        for (Connection conn : connections) {
            JsonObject connJson = new JsonObject();

            // Source: node index or -1 if disconnected
            int sourceId = conn.source != null ? nodes.indexOf(conn.source) : -1;
            connJson.addProperty("sourceId", sourceId);
            connJson.addProperty("outputIndex", conn.outputIndex);

            // Target: node index or -1 if disconnected
            int targetId = conn.target != null ? nodes.indexOf(conn.target) : -1;
            connJson.addProperty("targetId", targetId);
            connJson.addProperty("inputIndex", conn.inputIndex);

            // Free endpoint positions (only when disconnected)
            if (conn.source == null && conn.getFreeSourcePoint() != null) {
                connJson.addProperty("freeSourceX", conn.getFreeSourcePoint().x);
                connJson.addProperty("freeSourceY", conn.getFreeSourcePoint().y);
            }
            if (conn.target == null && conn.getFreeTargetPoint() != null) {
                connJson.addProperty("freeTargetX", conn.getFreeTargetPoint().x);
                connJson.addProperty("freeTargetY", conn.getFreeTargetPoint().y);
            }

            // Queue state
            connJson.addProperty("queueCapacity", conn.getConfiguredCapacity());
            connJson.addProperty("queueCount", conn.getQueueSize());
            connJson.addProperty("totalFramesSent", conn.getTotalFramesSent());

            connectionsArray.add(connJson);
        }
        root.add("connections", connectionsArray);

        // Write to file
        try (FileWriter writer = new FileWriter(path)) {
            GSON.toJson(root, writer);
        }
    }

    /**
     * Legacy save method for backwards compatibility during transition.
     * Converts old connection types to unified format.
     */
    @Deprecated
    public static void save(String path,
                           List<PipelineNode> nodes,
                           List<Connection> connections,
                           List<DanglingConnection> danglingConnections,
                           List<ReverseDanglingConnection> reverseDanglingConnections,
                           List<FreeConnection> freeConnections) throws IOException {

        // Convert all connection types to unified Connection list
        List<Connection> allConnections = new ArrayList<>(connections);

        // Convert dangling connections (source connected, target free)
        for (DanglingConnection dc : danglingConnections) {
            Connection conn = Connection.createSourceDangling(dc.source, dc.outputIndex, dc.freeEnd);
            conn.setConfiguredCapacity(dc.getConfiguredCapacity());
            conn.setLastQueueSize(dc.getQueueSize());
            allConnections.add(conn);
        }

        // Convert reverse dangling connections (target connected, source free)
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            Connection conn = Connection.createTargetDangling(rdc.target, 1, rdc.freeEnd);
            conn.setConfiguredCapacity(rdc.getConfiguredCapacity());
            conn.setLastQueueSize(rdc.getQueueSize());
            allConnections.add(conn);
        }

        // Convert free connections (both ends free)
        for (FreeConnection fc : freeConnections) {
            Connection conn = Connection.createFree(fc.startEnd, fc.arrowEnd);
            conn.setConfiguredCapacity(fc.getConfiguredCapacity());
            conn.setLastQueueSize(fc.getQueueSize());
            allConnections.add(conn);
        }

        // Save using unified format
        save(path, nodes, allConnections);
    }

    /**
     * Load a pipeline from a JSON file.
     */
    public static PipelineDocument load(String path, Display display, Shell shell, Canvas canvas)
            throws IOException {

        JsonObject root;
        try (FileReader reader = new FileReader(path)) {
            root = JsonParser.parseReader(reader).getAsJsonObject();
        }

        List<PipelineNode> nodes = new ArrayList<>();
        List<Connection> connections = new ArrayList<>();

        // Deserialize nodes
        if (root.has("nodes")) {
            for (JsonElement elem : root.getAsJsonArray("nodes")) {
                JsonObject nodeJson = elem.getAsJsonObject();

                String type = nodeJson.get("type").getAsString();
                int x = nodeJson.has("x") ? nodeJson.get("x").getAsInt() : 0;
                int y = nodeJson.has("y") ? nodeJson.get("y").getAsInt() : 0;

                PipelineNode node = createNode(type, nodeJson, display, shell, canvas, x, y);

                if (node != null) {
                    node.deserializeCommon(nodeJson);
                    node.deserializeProperties(nodeJson);
                    nodes.add(node);
                } else {
                    System.err.println("Failed to create node of type: " + type);
                }
            }
        }

        // Deserialize connections (new unified format)
        if (root.has("connections")) {
            for (JsonElement elem : root.getAsJsonArray("connections")) {
                JsonObject connJson = elem.getAsJsonObject();

                int sourceId = connJson.get("sourceId").getAsInt();
                int targetId = connJson.get("targetId").getAsInt();
                int inputIndex = connJson.has("inputIndex") ? connJson.get("inputIndex").getAsInt() : 1;
                int outputIndex = connJson.has("outputIndex") ? connJson.get("outputIndex").getAsInt() : 0;

                // Determine connection type based on source/target IDs
                PipelineNode sourceNode = (sourceId >= 0 && sourceId < nodes.size()) ? nodes.get(sourceId) : null;
                PipelineNode targetNode = (targetId >= 0 && targetId < nodes.size()) ? nodes.get(targetId) : null;

                Connection conn;
                if (sourceNode != null && targetNode != null) {
                    // Complete connection
                    conn = new Connection(sourceNode, targetNode, inputIndex, outputIndex);
                } else if (sourceNode != null) {
                    // Source-dangling (target free)
                    int freeX = connJson.has("freeTargetX") ? connJson.get("freeTargetX").getAsInt() : 0;
                    int freeY = connJson.has("freeTargetY") ? connJson.get("freeTargetY").getAsInt() : 0;
                    conn = Connection.createSourceDangling(sourceNode, outputIndex, new Point(freeX, freeY));
                } else if (targetNode != null) {
                    // Target-dangling (source free)
                    int freeX = connJson.has("freeSourceX") ? connJson.get("freeSourceX").getAsInt() : 0;
                    int freeY = connJson.has("freeSourceY") ? connJson.get("freeSourceY").getAsInt() : 0;
                    conn = Connection.createTargetDangling(targetNode, inputIndex, new Point(freeX, freeY));
                } else {
                    // Free connection (both ends free)
                    int srcX = connJson.has("freeSourceX") ? connJson.get("freeSourceX").getAsInt() : 0;
                    int srcY = connJson.has("freeSourceY") ? connJson.get("freeSourceY").getAsInt() : 0;
                    int tgtX = connJson.has("freeTargetX") ? connJson.get("freeTargetX").getAsInt() : 0;
                    int tgtY = connJson.has("freeTargetY") ? connJson.get("freeTargetY").getAsInt() : 0;
                    conn = Connection.createFree(new Point(srcX, srcY), new Point(tgtX, tgtY));
                }

                // Restore queue state
                if (connJson.has("queueCapacity")) {
                    conn.setConfiguredCapacity(connJson.get("queueCapacity").getAsInt());
                }
                if (connJson.has("queueCount")) {
                    conn.setLastQueueSize(connJson.get("queueCount").getAsInt());
                }
                if (connJson.has("totalFramesSent")) {
                    conn.setPendingTotalFrames(connJson.get("totalFramesSent").getAsLong());
                }

                connections.add(conn);
            }
        }

        // Load legacy connection formats for backwards compatibility
        loadLegacyConnections(root, nodes, connections);

        return new PipelineDocument(nodes, connections);
    }

    /**
     * Load legacy connection formats (danglingConnections, reverseDanglingConnections, freeConnections)
     * and convert them to unified Connection objects.
     */
    private static void loadLegacyConnections(JsonObject root, List<PipelineNode> nodes, List<Connection> connections) {
        // Load legacy dangling connections
        if (root.has("danglingConnections")) {
            for (JsonElement elem : root.getAsJsonArray("danglingConnections")) {
                JsonObject dcJson = elem.getAsJsonObject();
                int sourceId = dcJson.get("sourceId").getAsInt();
                int outputIndex = dcJson.has("outputIndex") ? dcJson.get("outputIndex").getAsInt() : 0;
                int endX = dcJson.has("freeEndX") ? dcJson.get("freeEndX").getAsInt() : dcJson.get("endX").getAsInt();
                int endY = dcJson.has("freeEndY") ? dcJson.get("freeEndY").getAsInt() : dcJson.get("endY").getAsInt();

                if (sourceId >= 0 && sourceId < nodes.size()) {
                    Connection conn = Connection.createSourceDangling(nodes.get(sourceId), outputIndex, new Point(endX, endY));
                    if (dcJson.has("queueCapacity")) {
                        conn.setConfiguredCapacity(dcJson.get("queueCapacity").getAsInt());
                    }
                    if (dcJson.has("queueCount")) {
                        conn.setLastQueueSize(dcJson.get("queueCount").getAsInt());
                    }
                    connections.add(conn);
                }
            }
        }

        // Load legacy reverse dangling connections
        if (root.has("reverseDanglingConnections")) {
            for (JsonElement elem : root.getAsJsonArray("reverseDanglingConnections")) {
                JsonObject rdcJson = elem.getAsJsonObject();
                int targetId = rdcJson.get("targetId").getAsInt();
                int inputIndex = rdcJson.has("inputIndex") ? rdcJson.get("inputIndex").getAsInt() : 1;
                int startX = rdcJson.has("freeEndX") ? rdcJson.get("freeEndX").getAsInt() :
                             rdcJson.has("freeStartX") ? rdcJson.get("freeStartX").getAsInt() :
                             rdcJson.get("startX").getAsInt();
                int startY = rdcJson.has("freeEndY") ? rdcJson.get("freeEndY").getAsInt() :
                             rdcJson.has("freeStartY") ? rdcJson.get("freeStartY").getAsInt() :
                             rdcJson.get("startY").getAsInt();

                if (targetId >= 0 && targetId < nodes.size()) {
                    Connection conn = Connection.createTargetDangling(nodes.get(targetId), inputIndex, new Point(startX, startY));
                    if (rdcJson.has("queueCapacity")) {
                        conn.setConfiguredCapacity(rdcJson.get("queueCapacity").getAsInt());
                    }
                    if (rdcJson.has("queueCount")) {
                        conn.setLastQueueSize(rdcJson.get("queueCount").getAsInt());
                    }
                    connections.add(conn);
                }
            }
        }

        // Load legacy free connections
        if (root.has("freeConnections")) {
            for (JsonElement elem : root.getAsJsonArray("freeConnections")) {
                JsonObject fcJson = elem.getAsJsonObject();
                int startX = fcJson.has("startEndX") ? fcJson.get("startEndX").getAsInt() : fcJson.get("startX").getAsInt();
                int startY = fcJson.has("startEndY") ? fcJson.get("startEndY").getAsInt() : fcJson.get("startY").getAsInt();
                int endX = fcJson.has("arrowEndX") ? fcJson.get("arrowEndX").getAsInt() : fcJson.get("endX").getAsInt();
                int endY = fcJson.has("arrowEndY") ? fcJson.get("arrowEndY").getAsInt() : fcJson.get("endY").getAsInt();

                Connection conn = Connection.createFree(new Point(startX, startY), new Point(endX, endY));
                if (fcJson.has("queueCapacity")) {
                    conn.setConfiguredCapacity(fcJson.get("queueCapacity").getAsInt());
                }
                if (fcJson.has("queueCount")) {
                    conn.setLastQueueSize(fcJson.get("queueCount").getAsInt());
                }
                connections.add(conn);
            }
        }
    }

    /**
     * Create a node by type, handling both source and processing nodes.
     * Also handles legacy type names for backward compatibility.
     */
    private static PipelineNode createNode(String type, JsonObject nodeJson,
                                            Display display, Shell shell, Canvas canvas, int x, int y) {
        // Handle legacy source node types
        if ("FileSource".equals(type) || "FileSourceNode".equals(type)) {
            return new FileSourceNode(shell, display, canvas, x, y);
        } else if ("WebcamSource".equals(type) || "WebcamSourceNode".equals(type)) {
            return new WebcamSourceNode(shell, display, canvas, x, y);
        } else if ("BlankSource".equals(type) || "BlankSourceNode".equals(type)) {
            return new BlankSourceNode(shell, display, x, y);
        } else if ("ContainerInput".equals(type) || "ContainerInputNode".equals(type)) {
            return new ContainerInputNode(shell, display, x, y);
        } else if ("ContainerOutput".equals(type) || "ContainerOutputNode".equals(type)) {
            return new ContainerOutputNode(display, shell, x, y);
        }

        // Handle legacy "Processing" type - look at "name" field
        if ("Processing".equals(type) && nodeJson.has("name")) {
            type = nodeJson.get("name").getAsString();
        }

        // Try via unified registry
        PipelineNode node = NodeRegistry.createNode(type, display, shell, canvas, x, y);
        if (node != null) {
            return node;
        }

        // Fallback: try as processing node with legacy name lookup
        ProcessingNode processingNode = NodeRegistry.createProcessingNode(type, display, shell, x, y);
        if (processingNode != null) {
            return processingNode;
        }

        System.err.println("Unknown node type: " + type);
        return null;
    }

    /**
     * Load a container's internal pipeline from a JSON file.
     * Unlike regular load, this updates the existing boundary nodes in the container
     * with positions from the file, rather than creating new ones.
     *
     * @param path The path to the JSON file
     * @param container The container whose internal pipeline is being loaded
     * @param display The SWT display
     * @param shell The SWT shell
     * @return true if loading succeeded, false otherwise
     */
    public static boolean loadContainerPipeline(String path, ContainerNode container,
                                                 Display display, Shell shell) throws IOException {
        JsonObject root;
        try (FileReader reader = new FileReader(path)) {
            root = JsonParser.parseReader(reader).getAsJsonObject();
        }

        // Clear existing child nodes and connections (but keep boundary nodes)
        container.getChildNodes().clear();
        container.getChildConnections().clear();

        // Build a list of all nodes for connection resolution
        // Index 0 will be boundary input, last will be boundary output
        List<PipelineNode> allNodes = new ArrayList<>();

        // Deserialize nodes
        if (root.has("nodes")) {
            for (JsonElement elem : root.getAsJsonArray("nodes")) {
                JsonObject nodeJson = elem.getAsJsonObject();

                String type = nodeJson.get("type").getAsString();
                int x = nodeJson.has("x") ? nodeJson.get("x").getAsInt() : 0;
                int y = nodeJson.has("y") ? nodeJson.get("y").getAsInt() : 0;

                // Handle boundary nodes specially - update existing ones
                if ("ContainerInput".equals(type) || "ContainerInputNode".equals(type)) {
                    ContainerInputNode boundaryInput = container.getBoundaryInput();
                    boundaryInput.setX(x);
                    boundaryInput.setY(y);
                    boundaryInput.deserializeCommon(nodeJson);
                    boundaryInput.deserializeProperties(nodeJson);
                    allNodes.add(boundaryInput);
                } else if ("ContainerOutput".equals(type) || "ContainerOutputNode".equals(type)) {
                    ContainerOutputNode boundaryOutput = container.getBoundaryOutput();
                    boundaryOutput.setX(x);
                    boundaryOutput.setY(y);
                    boundaryOutput.deserializeCommon(nodeJson);
                    boundaryOutput.deserializeProperties(nodeJson);
                    allNodes.add(boundaryOutput);
                } else {
                    // Regular node - create it
                    PipelineNode node = createNode(type, nodeJson, display, shell, null, x, y);
                    if (node != null) {
                        node.deserializeCommon(nodeJson);
                        node.deserializeProperties(nodeJson);
                        container.addChildNode(node);
                        allNodes.add(node);
                    } else {
                        System.err.println("Failed to create node of type: " + type);
                        allNodes.add(null); // Placeholder to keep indices aligned
                    }
                }
            }
        }

        // Deserialize connections
        if (root.has("connections")) {
            for (JsonElement elem : root.getAsJsonArray("connections")) {
                JsonObject connJson = elem.getAsJsonObject();
                int sourceId = connJson.get("sourceId").getAsInt();
                int targetId = connJson.get("targetId").getAsInt();
                int inputIndex = connJson.has("inputIndex") ? connJson.get("inputIndex").getAsInt() : 1;
                int outputIndex = connJson.has("outputIndex") ? connJson.get("outputIndex").getAsInt() : 0;

                if (sourceId >= 0 && sourceId < allNodes.size() &&
                    targetId >= 0 && targetId < allNodes.size() &&
                    allNodes.get(sourceId) != null && allNodes.get(targetId) != null) {

                    Connection conn = new Connection(allNodes.get(sourceId), allNodes.get(targetId), inputIndex, outputIndex);

                    // Restore queue capacity if specified
                    if (connJson.has("queueCapacity")) {
                        conn.setConfiguredCapacity(connJson.get("queueCapacity").getAsInt());
                    }
                    if (connJson.has("queueCount")) {
                        conn.setLastQueueSize(connJson.get("queueCount").getAsInt());
                    }
                    if (connJson.has("totalFramesSent")) {
                        conn.setPendingTotalFrames(connJson.get("totalFramesSent").getAsLong());
                    }

                    container.addChildConnection(conn);
                }
            }
        }

        return true;
    }
}
