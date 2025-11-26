package com.ttennebkram.pipeline.serialization;

import com.google.gson.*;
import com.ttennebkram.pipeline.model.*;
import com.ttennebkram.pipeline.nodes.*;
import com.ttennebkram.pipeline.registry.NodeRegistry;
import org.eclipse.swt.widgets.Canvas;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Handles serialization and deserialization of pipeline documents.
 * Each node handles its own property serialization via NodeSerializable interface.
 */
public class PipelineSerializer {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    /**
     * Result of loading a pipeline document.
     */
    public static class PipelineDocument {
        public final List<PipelineNode> nodes;
        public final List<Connection> connections;
        public final List<DanglingConnection> danglingConnections;
        public final List<ReverseDanglingConnection> reverseDanglingConnections;
        public final List<FreeConnection> freeConnections;

        public PipelineDocument(List<PipelineNode> nodes,
                                List<Connection> connections,
                                List<DanglingConnection> danglingConnections,
                                List<ReverseDanglingConnection> reverseDanglingConnections,
                                List<FreeConnection> freeConnections) {
            this.nodes = nodes;
            this.connections = connections;
            this.danglingConnections = danglingConnections;
            this.reverseDanglingConnections = reverseDanglingConnections;
            this.freeConnections = freeConnections;
        }
    }

    /**
     * Save a pipeline to a JSON file.
     */
    public static void save(String path,
                           List<PipelineNode> nodes,
                           List<Connection> connections,
                           List<DanglingConnection> danglingConnections,
                           List<ReverseDanglingConnection> reverseDanglingConnections,
                           List<FreeConnection> freeConnections) throws IOException {

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

        // Serialize connections
        JsonArray connectionsArray = new JsonArray();
        for (Connection conn : connections) {
            JsonObject connJson = new JsonObject();
            connJson.addProperty("sourceId", nodes.indexOf(conn.source));
            connJson.addProperty("targetId", nodes.indexOf(conn.target));
            connJson.addProperty("inputIndex", conn.inputIndex);
            connJson.addProperty("outputIndex", conn.outputIndex);
            connJson.addProperty("queueCapacity", conn.getConfiguredCapacity());
            connJson.addProperty("queueCount", conn.getQueueSize());
            connJson.addProperty("totalFramesSent", conn.getTotalFramesSent());
            connectionsArray.add(connJson);
        }
        root.add("connections", connectionsArray);

        // Serialize dangling connections
        JsonArray danglingArray = new JsonArray();
        for (DanglingConnection dc : danglingConnections) {
            JsonObject dcJson = new JsonObject();
            dcJson.addProperty("sourceId", nodes.indexOf(dc.source));
            dcJson.addProperty("outputIndex", dc.outputIndex);
            dcJson.addProperty("freeEndX", dc.freeEnd.x);
            dcJson.addProperty("freeEndY", dc.freeEnd.y);
            dcJson.addProperty("queueCapacity", dc.getConfiguredCapacity());
            dcJson.addProperty("queueCount", dc.getQueueSize());
            danglingArray.add(dcJson);
        }
        root.add("danglingConnections", danglingArray);

        // Serialize reverse dangling connections
        JsonArray reverseDanglingArray = new JsonArray();
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            JsonObject rdcJson = new JsonObject();
            rdcJson.addProperty("targetId", nodes.indexOf(rdc.target));
            rdcJson.addProperty("freeEndX", rdc.freeEnd.x);
            rdcJson.addProperty("freeEndY", rdc.freeEnd.y);
            rdcJson.addProperty("queueCapacity", rdc.getConfiguredCapacity());
            rdcJson.addProperty("queueCount", rdc.getQueueSize());
            reverseDanglingArray.add(rdcJson);
        }
        root.add("reverseDanglingConnections", reverseDanglingArray);

        // Serialize free connections
        JsonArray freeArray = new JsonArray();
        for (FreeConnection fc : freeConnections) {
            JsonObject fcJson = new JsonObject();
            fcJson.addProperty("startEndX", fc.startEnd.x);
            fcJson.addProperty("startEndY", fc.startEnd.y);
            fcJson.addProperty("arrowEndX", fc.arrowEnd.x);
            fcJson.addProperty("arrowEndY", fc.arrowEnd.y);
            fcJson.addProperty("queueCapacity", fc.getConfiguredCapacity());
            fcJson.addProperty("queueCount", fc.getQueueSize());
            freeArray.add(fcJson);
        }
        root.add("freeConnections", freeArray);

        // Write to file
        try (FileWriter writer = new FileWriter(path)) {
            GSON.toJson(root, writer);
        }
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
        List<DanglingConnection> danglingConnections = new ArrayList<>();
        List<ReverseDanglingConnection> reverseDanglingConnections = new ArrayList<>();
        List<FreeConnection> freeConnections = new ArrayList<>();

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

        // Deserialize connections
        if (root.has("connections")) {
            for (JsonElement elem : root.getAsJsonArray("connections")) {
                JsonObject connJson = elem.getAsJsonObject();
                int sourceId = connJson.get("sourceId").getAsInt();
                int targetId = connJson.get("targetId").getAsInt();
                int inputIndex = connJson.has("inputIndex") ? connJson.get("inputIndex").getAsInt() : 1;
                int outputIndex = connJson.has("outputIndex") ? connJson.get("outputIndex").getAsInt() : 0;

                if (sourceId >= 0 && sourceId < nodes.size() &&
                    targetId >= 0 && targetId < nodes.size()) {

                    Connection conn = new Connection(nodes.get(sourceId), nodes.get(targetId), inputIndex, outputIndex);

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

                    connections.add(conn);
                }
            }
        }

        // Deserialize dangling connections
        if (root.has("danglingConnections")) {
            for (JsonElement elem : root.getAsJsonArray("danglingConnections")) {
                JsonObject dcJson = elem.getAsJsonObject();
                int sourceId = dcJson.get("sourceId").getAsInt();
                int endX = dcJson.has("freeEndX") ? dcJson.get("freeEndX").getAsInt() : dcJson.get("endX").getAsInt();
                int endY = dcJson.has("freeEndY") ? dcJson.get("freeEndY").getAsInt() : dcJson.get("endY").getAsInt();

                if (sourceId >= 0 && sourceId < nodes.size()) {
                    DanglingConnection dc = new DanglingConnection(nodes.get(sourceId), endX, endY);
                    if (dcJson.has("queueCapacity")) {
                        dc.setConfiguredCapacity(dcJson.get("queueCapacity").getAsInt());
                    }
                    if (dcJson.has("queueCount")) {
                        dc.setLastQueueSize(dcJson.get("queueCount").getAsInt());
                    }
                    danglingConnections.add(dc);
                }
            }
        }

        // Deserialize reverse dangling connections
        if (root.has("reverseDanglingConnections")) {
            for (JsonElement elem : root.getAsJsonArray("reverseDanglingConnections")) {
                JsonObject rdcJson = elem.getAsJsonObject();
                int targetId = rdcJson.get("targetId").getAsInt();
                int inputIndex = rdcJson.has("inputIndex") ? rdcJson.get("inputIndex").getAsInt() : 0;
                // Support both old (freeEndX) and legacy (freeStartX) field names
                int startX = rdcJson.has("freeEndX") ? rdcJson.get("freeEndX").getAsInt() :
                             rdcJson.has("freeStartX") ? rdcJson.get("freeStartX").getAsInt() :
                             rdcJson.get("startX").getAsInt();
                int startY = rdcJson.has("freeEndY") ? rdcJson.get("freeEndY").getAsInt() :
                             rdcJson.has("freeStartY") ? rdcJson.get("freeStartY").getAsInt() :
                             rdcJson.get("startY").getAsInt();

                if (targetId >= 0 && targetId < nodes.size()) {
                    ReverseDanglingConnection rdc = new ReverseDanglingConnection(nodes.get(targetId), inputIndex, startX, startY);
                    if (rdcJson.has("queueCapacity")) {
                        rdc.setConfiguredCapacity(rdcJson.get("queueCapacity").getAsInt());
                    }
                    if (rdcJson.has("queueCount")) {
                        rdc.setLastQueueSize(rdcJson.get("queueCount").getAsInt());
                    }
                    reverseDanglingConnections.add(rdc);
                }
            }
        }

        // Deserialize free connections
        if (root.has("freeConnections")) {
            for (JsonElement elem : root.getAsJsonArray("freeConnections")) {
                JsonObject fcJson = elem.getAsJsonObject();
                // Support both old (startEndX) and new (startX) field names
                int startX = fcJson.has("startEndX") ? fcJson.get("startEndX").getAsInt() : fcJson.get("startX").getAsInt();
                int startY = fcJson.has("startEndY") ? fcJson.get("startEndY").getAsInt() : fcJson.get("startY").getAsInt();
                int endX = fcJson.has("arrowEndX") ? fcJson.get("arrowEndX").getAsInt() : fcJson.get("endX").getAsInt();
                int endY = fcJson.has("arrowEndY") ? fcJson.get("arrowEndY").getAsInt() : fcJson.get("endY").getAsInt();

                FreeConnection fc = new FreeConnection(startX, startY, endX, endY);
                if (fcJson.has("queueCapacity")) {
                    fc.setConfiguredCapacity(fcJson.get("queueCapacity").getAsInt());
                }
                if (fcJson.has("queueCount")) {
                    fc.setLastQueueSize(fcJson.get("queueCount").getAsInt());
                }
                freeConnections.add(fc);
            }
        }

        return new PipelineDocument(nodes, connections, danglingConnections,
                                    reverseDanglingConnections, freeConnections);
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
