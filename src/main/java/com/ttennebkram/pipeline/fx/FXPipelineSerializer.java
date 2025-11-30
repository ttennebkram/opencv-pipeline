package com.ttennebkram.pipeline.fx;

import com.google.gson.*;
import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import javafx.scene.image.WritableImage;
import javafx.scene.paint.Color;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.Base64;
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
     * Thumbnails are saved to a cache directory instead of being embedded in the JSON.
     */
    public static void save(String path, List<FXNode> nodes, List<FXConnection> connections) throws IOException {
        JsonObject root = new JsonObject();

        // Get cache directory and pipeline name for thumbnail caching
        String cacheDir = getCacheDirectory(path);
        String pipelineName = getPipelineName(path);

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

            // Position only (width/height derived from node type)
            nodeJson.addProperty("x", node.x);
            nodeJson.addProperty("y", node.y);

            // State
            nodeJson.addProperty("enabled", node.enabled);

            // Runtime stats (persist work completed, priority, and I/O counts across sessions)
            nodeJson.addProperty("workUnitsCompleted", node.workUnitsCompleted);
            nodeJson.addProperty("threadPriority", node.threadPriority);
            nodeJson.addProperty("inputCount", node.inputCount);
            nodeJson.addProperty("inputCount2", node.inputCount2);
            nodeJson.addProperty("outputCount1", node.outputCount1);
            nodeJson.addProperty("outputCount2", node.outputCount2);
            nodeJson.addProperty("outputCount3", node.outputCount3);
            nodeJson.addProperty("outputCount4", node.outputCount4);

            // Node configuration
            nodeJson.addProperty("hasInput", node.hasInput);
            nodeJson.addProperty("hasDualInput", node.hasDualInput);
            nodeJson.addProperty("queuesInSync", node.queuesInSync);
            nodeJson.addProperty("outputCount", node.outputCount);
            nodeJson.addProperty("isBoundaryNode", node.isBoundaryNode);

            // Background color
            if (node.backgroundColor != null) {
                nodeJson.addProperty("bgColorR", node.backgroundColor.getRed());
                nodeJson.addProperty("bgColorG", node.backgroundColor.getGreen());
                nodeJson.addProperty("bgColorB", node.backgroundColor.getBlue());
            }

            // Node-type specific properties
            if ("WebcamSource".equals(node.nodeType)) {
                nodeJson.addProperty("cameraIndex", node.cameraIndex);
            } else if ("FileSource".equals(node.nodeType)) {
                nodeJson.addProperty("filePath", node.filePath);
                nodeJson.addProperty("fps", node.fps);
            }

            // Save generic properties map (for processing nodes like Gain, FFT, BitPlanes, etc.)
            saveNodeProperties(nodeJson, node);

            // Container-specific properties - internal nodes and connections
            nodeJson.addProperty("isContainer", node.isContainer);
            if (node.isContainer) {
                // Save pipeline file path if set (external sub-diagram file)
                if (node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty()) {
                    nodeJson.addProperty("pipelineFile", node.pipelineFilePath);
                }
                // Also save inline inner nodes if present
                if (!node.innerNodes.isEmpty()) {
                    JsonArray innerNodesArray = serializeNodes(node.innerNodes);
                    nodeJson.add("innerNodes", innerNodesArray);

                    JsonArray innerConnectionsArray = serializeConnections(node.innerConnections, node.innerNodes);
                    nodeJson.add("innerConnections", innerConnectionsArray);
                }
            }

            // Save thumbnail to cache file (not embedded in JSON)
            if (node.thumbnail != null) {
                saveThumbnailToCache(cacheDir, pipelineName, node, i);
            }

            nodesArray.add(nodeJson);
        }
        root.add("nodes", nodesArray);

        // Serialize connections
        JsonArray connectionsArray = new JsonArray();
        for (FXConnection conn : connections) {
            JsonObject connJson = new JsonObject();

            // Source - use indexOf to find actual position, like main branch does
            int sourceIndex = conn.source != null ? nodes.indexOf(conn.source) : -1;
            connJson.addProperty("sourceIndex", sourceIndex);
            connJson.addProperty("sourceOutputIndex", conn.sourceOutputIndex);

            // Target - use indexOf to find actual position, like main branch does
            int targetIndex = conn.target != null ? nodes.indexOf(conn.target) : -1;
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
            JsonElement parsed = JsonParser.parseReader(reader);
            if (parsed == null || !parsed.isJsonObject()) {
                throw new IOException("Invalid pipeline file: not a valid JSON object");
            }
            root = parsed.getAsJsonObject();
        } catch (JsonSyntaxException e) {
            throw new IOException("Invalid pipeline file: " + e.getMessage());
        }

        // Validate this looks like a pipeline file
        if (!root.has("nodes")) {
            throw new IOException("Invalid pipeline file: missing 'nodes' array. This doesn't appear to be a pipeline file.");
        }

        List<FXNode> nodes = new ArrayList<>();
        List<FXConnection> connections = new ArrayList<>();

        // First pass: find the maximum ID used in the file and advance the counter
        // This prevents ID collisions when loading inner nodes
        int maxIdInFile = findMaxIdInJson(root.getAsJsonArray("nodes"));
        while (FXNode.generateNewId() <= maxIdInFile) {
            // Keep generating until we're past the max
        }

        // Deserialize nodes
        if (root.has("nodes")) {
            for (JsonElement elem : root.getAsJsonArray("nodes")) {
                JsonObject nodeJson = elem.getAsJsonObject();

                if (!nodeJson.has("type")) {
                    throw new IOException("Invalid pipeline file: node missing 'type' field");
                }
                String type = nodeJson.get("type").getAsString();
                String savedLabel = nodeJson.has("label") ? nodeJson.get("label").getAsString() : null;
                double x = nodeJson.has("x") ? nodeJson.get("x").getAsDouble() : 0;
                double y = nodeJson.has("y") ? nodeJson.get("y").getAsDouble() : 0;

                // Create node using factory - this sets the correct default label from registry
                FXNode node = FXNodeFactory.createFXNode(type, (int) x, (int) y);

                // Restore node ID if present in the file (important for connection references)
                if (nodeJson.has("id")) {
                    node.id = nodeJson.get("id").getAsInt();
                }

                // Only override label if user had customized it
                // Keep factory-assigned label (from registry displayName) if:
                // - savedLabel equals the type name (e.g., "WebcamSource")
                // - savedLabel equals the current displayName (e.g., "Webcam Source")
                // - savedLabel is an old-style short name we want to update (e.g., "Webcam" -> "Webcam Source")
                if (savedLabel != null && !savedLabel.equals(type) && !savedLabel.equals(node.label)) {
                    // Check if this is an old-style label that should be updated
                    // Old files might have "Webcam" instead of "Webcam Source"
                    boolean isOldStyleDefault = isOldStyleDefaultLabel(savedLabel, type);
                    if (!isOldStyleDefault) {
                        node.label = savedLabel;
                    }
                }

                // Restore position (but not size - use current constants for consistent sizing)
                node.x = x;
                node.y = y;
                // Don't restore width/height - let nodes use current constants
                // This ensures nodes adapt to updated sizing constants

                // Restore state
                if (nodeJson.has("enabled")) {
                    node.enabled = nodeJson.get("enabled").getAsBoolean();
                }

                // Restore runtime stats (work completed, priority, and I/O counts persisted across sessions)
                if (nodeJson.has("workUnitsCompleted")) {
                    node.workUnitsCompleted = nodeJson.get("workUnitsCompleted").getAsLong();
                }
                if (nodeJson.has("threadPriority")) {
                    node.threadPriority = nodeJson.get("threadPriority").getAsInt();
                }
                if (nodeJson.has("inputCount")) {
                    node.inputCount = nodeJson.get("inputCount").getAsInt();
                }
                if (nodeJson.has("inputCount2")) {
                    node.inputCount2 = nodeJson.get("inputCount2").getAsInt();
                }
                if (nodeJson.has("outputCount1")) {
                    node.outputCount1 = nodeJson.get("outputCount1").getAsInt();
                }
                if (nodeJson.has("outputCount2")) {
                    node.outputCount2 = nodeJson.get("outputCount2").getAsInt();
                }
                if (nodeJson.has("outputCount3")) {
                    node.outputCount3 = nodeJson.get("outputCount3").getAsInt();
                }
                if (nodeJson.has("outputCount4")) {
                    node.outputCount4 = nodeJson.get("outputCount4").getAsInt();
                }

                // Restore node configuration
                if (nodeJson.has("hasInput")) {
                    node.hasInput = nodeJson.get("hasInput").getAsBoolean();
                }
                if (nodeJson.has("hasDualInput")) {
                    node.hasDualInput = nodeJson.get("hasDualInput").getAsBoolean();
                }
                if (nodeJson.has("queuesInSync")) {
                    node.queuesInSync = nodeJson.get("queuesInSync").getAsBoolean();
                }
                if (nodeJson.has("outputCount")) {
                    node.outputCount = nodeJson.get("outputCount").getAsInt();
                }
                if (nodeJson.has("isBoundaryNode")) {
                    node.isBoundaryNode = nodeJson.get("isBoundaryNode").getAsBoolean();
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
                } else if ("FileSource".equals(type)) {
                    if (nodeJson.has("filePath")) {
                        node.filePath = nodeJson.get("filePath").getAsString();
                    }
                    if (nodeJson.has("fps")) {
                        node.fps = nodeJson.get("fps").getAsDouble();
                    }
                }

                // Restore generic properties for processing nodes
                loadNodeProperties(nodeJson, node);

                // For FileSource: if filePath is empty but imagePath exists in properties, sync it
                if ("FileSource".equals(type)) {
                    if ((node.filePath == null || node.filePath.isEmpty()) && node.properties.containsKey("imagePath")) {
                        Object imagePath = node.properties.get("imagePath");
                        if (imagePath != null && !imagePath.toString().isEmpty()) {
                            node.filePath = imagePath.toString();
                        }
                    }
                }

                // Try to load thumbnail from cache file first, fall back to embedded Base64 for backwards compatibility
                String cacheDir = getCacheDirectory(path);
                String pipelineName = getPipelineName(path);
                int nodeIndex = nodes.size(); // Current index before adding this node
                if (!loadThumbnailFromCache(cacheDir, pipelineName, node, nodeIndex)) {
                    // Fall back to embedded Base64 for older files
                    if (nodeJson.has("thumbnail")) {
                        String thumbnailBase64 = nodeJson.get("thumbnail").getAsString();
                        node.thumbnail = base64ToImage(thumbnailBase64);
                        // Save to cache for next time (migration from embedded to cached)
                        if (node.thumbnail != null) {
                            saveThumbnailToCache(cacheDir, pipelineName, node, nodeIndex);
                        }
                    }
                }

                // Restore container-specific properties
                if (nodeJson.has("isContainer")) {
                    node.isContainer = nodeJson.get("isContainer").getAsBoolean();
                }
                // Also mark as container if nodeType is "Container" (for files without isContainer flag)
                if ("Container".equals(type)) {
                    node.isContainer = true;
                }
                // Ensure container nodes use the standard container color
                if (node.isContainer) {
                    node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
                    // Load pipeline file path if present
                    if (nodeJson.has("pipelineFile")) {
                        node.pipelineFilePath = nodeJson.get("pipelineFile").getAsString();
                    }
                }
                // Load inner nodes: prioritize inline data (which may have thumbnails) over external file
                if (node.isContainer) {
                    boolean loadedInline = false;

                    // First try inline inner nodes - these may have thumbnails from a previous save
                    if (nodeJson.has("innerNodes")) {
                        node.innerNodes = deserializeNodes(nodeJson.getAsJsonArray("innerNodes"), path);
                        if (nodeJson.has("innerConnections")) {
                            node.innerConnections = deserializeConnections(nodeJson.getAsJsonArray("innerConnections"), node.innerNodes);
                        }
                        // Reassign IDs to inner nodes to avoid collisions with outer nodes
                        reassignInnerNodeIds(node.innerNodes);
                        loadedInline = true;
                        System.out.println("Loaded inline inner nodes for container: " + node.label +
                                           " (" + node.innerNodes.size() + " nodes)");
                    }

                    // Fall back to external pipeline file if no inline data
                    if (!loadedInline && node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty()) {
                        // Resolve relative paths against the parent document's directory
                        String resolvedPath = node.pipelineFilePath;
                        File pipelineFile = new File(resolvedPath);
                        if (!pipelineFile.isAbsolute()) {
                            // Relative path - resolve against parent document's directory
                            File parentDir = new File(path).getParentFile();
                            if (parentDir != null) {
                                pipelineFile = new File(parentDir, node.pipelineFilePath);
                                resolvedPath = pipelineFile.getAbsolutePath();
                            }
                        }
                        if (pipelineFile.exists()) {
                            try {
                                PipelineDocument externalDoc = load(resolvedPath);
                                node.innerNodes = externalDoc.nodes;
                                node.innerConnections = externalDoc.connections;
                                // Reassign IDs to inner nodes to avoid collisions with outer nodes
                                reassignInnerNodeIds(node.innerNodes);
                                System.out.println("Loaded external pipeline for container: " + resolvedPath);
                            } catch (IOException e) {
                                System.err.println("Failed to load external pipeline " + resolvedPath + ": " + e.getMessage());
                            }
                        } else {
                            System.err.println("External pipeline file not found: " + resolvedPath +
                                " (original: " + node.pipelineFilePath + ")");
                        }
                    }
                }

                nodes.add(node);
            }
        }

        // Build a map from node ID to node for ID-based connection lookup
        Map<Integer, FXNode> nodeById = new HashMap<>();
        for (FXNode node : nodes) {
            nodeById.put(node.id, node);
        }

        // Deserialize connections
        if (root.has("connections")) {
            for (JsonElement elem : root.getAsJsonArray("connections")) {
                JsonObject connJson = elem.getAsJsonObject();

                FXNode sourceNode = null;
                FXNode targetNode = null;

                // Support both formats: sourceIndex/targetIndex (new format) and sourceId/targetId (legacy format)
                if (connJson.has("sourceIndex") && connJson.has("targetIndex")) {
                    // New format: reference by array index
                    int sourceIndex = connJson.get("sourceIndex").getAsInt();
                    int targetIndex = connJson.get("targetIndex").getAsInt();
                    sourceNode = (sourceIndex >= 0 && sourceIndex < nodes.size()) ? nodes.get(sourceIndex) : null;
                    targetNode = (targetIndex >= 0 && targetIndex < nodes.size()) ? nodes.get(targetIndex) : null;
                } else if (connJson.has("sourceId") && connJson.has("targetId")) {
                    // Legacy format: reference by node ID
                    int sourceId = connJson.get("sourceId").getAsInt();
                    int targetId = connJson.get("targetId").getAsInt();
                    sourceNode = nodeById.get(sourceId);
                    targetNode = nodeById.get(targetId);
                } else {
                    System.err.println("Warning: skipping connection with missing source/target references");
                    continue;
                }

                // Handle output/input index - support both naming conventions
                int sourceOutputIndex = 0;
                if (connJson.has("sourceOutputIndex")) {
                    sourceOutputIndex = connJson.get("sourceOutputIndex").getAsInt();
                } else if (connJson.has("outputIndex")) {
                    sourceOutputIndex = connJson.get("outputIndex").getAsInt();
                }

                int targetInputIndex = 0;
                if (connJson.has("targetInputIndex")) {
                    targetInputIndex = connJson.get("targetInputIndex").getAsInt();
                } else if (connJson.has("inputIndex")) {
                    // Old format used 1-based indexing (1=primary, 2=secondary)
                    // Convert to 0-based (0=primary, 1=secondary)
                    targetInputIndex = connJson.get("inputIndex").getAsInt() - 1;
                    if (targetInputIndex < 0) targetInputIndex = 0;
                }

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

        // Ensure the global ID counter is past all existing IDs to prevent future collisions
        ensureUniqueIds(nodes);

        return new PipelineDocument(nodes, connections);
    }

    /**
     * Convert a JavaFX Image to Base64-encoded PNG string.
     */
    private static String imageToBase64(Image image) {
        if (image == null) return null;
        try {
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(image, null);
            if (bufferedImage == null) return null;

            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(bufferedImage, "png", baos);
            return Base64.getEncoder().encodeToString(baos.toByteArray());
        } catch (Exception e) {
            System.err.println("Failed to encode thumbnail: " + e.getMessage());
            return null;
        }
    }

    /**
     * Save a thumbnail to a cache file.
     * Filename pattern: {pipelineName}_node_{nodeType}_{index}_thumb.png
     *
     * @param cacheDir Directory to save thumbnails
     * @param pipelineName Base name of the pipeline file (without .json extension)
     * @param node The node whose thumbnail to save
     * @param index Index of the node in the pipeline
     */
    public static void saveThumbnailToCache(String cacheDir, String pipelineName, FXNode node, int index) {
        if (node.thumbnail == null) return;
        try {
            File cacheFolder = new File(cacheDir);
            if (!cacheFolder.exists()) {
                cacheFolder.mkdirs();
            }
            String thumbPath = cacheDir + File.separator + pipelineName + "_node_" + node.nodeType + "_" + index + "_thumb.png";
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(node.thumbnail, null);
            if (bufferedImage != null) {
                ImageIO.write(bufferedImage, "png", new File(thumbPath));
            }
        } catch (Exception e) {
            System.err.println("Failed to save thumbnail to cache: " + e.getMessage());
        }
    }

    /**
     * Load a thumbnail from a cache file.
     * Filename pattern: {pipelineName}_node_{nodeType}_{index}_thumb.png
     *
     * @param cacheDir Directory to load thumbnails from
     * @param pipelineName Base name of the pipeline file (without .json extension)
     * @param node The node to load the thumbnail into
     * @param index Index of the node in the pipeline
     * @return true if thumbnail was loaded successfully
     */
    public static boolean loadThumbnailFromCache(String cacheDir, String pipelineName, FXNode node, int index) {
        String thumbPath = cacheDir + File.separator + pipelineName + "_node_" + node.nodeType + "_" + index + "_thumb.png";
        File thumbFile = new File(thumbPath);
        if (thumbFile.exists()) {
            try {
                BufferedImage bufferedImage = ImageIO.read(thumbFile);
                if (bufferedImage != null) {
                    node.thumbnail = SwingFXUtils.toFXImage(bufferedImage, null);
                    return true;
                }
            } catch (Exception e) {
                System.err.println("Failed to load thumbnail from cache: " + e.getMessage());
            }
        }
        return false;
    }

    /**
     * Save all node thumbnails to a cache directory.
     *
     * @param cacheDir Directory to save thumbnails
     * @param pipelineName Base name of the pipeline file (without .json extension)
     * @param nodes List of nodes whose thumbnails to save
     */
    public static void saveAllThumbnailsToCache(String cacheDir, String pipelineName, List<FXNode> nodes) {
        for (int i = 0; i < nodes.size(); i++) {
            saveThumbnailToCache(cacheDir, pipelineName, nodes.get(i), i);
        }
    }

    /**
     * Load all node thumbnails from a cache directory.
     *
     * @param cacheDir Directory to load thumbnails from
     * @param pipelineName Base name of the pipeline file (without .json extension)
     * @param nodes List of nodes to load thumbnails into
     * @return Number of thumbnails successfully loaded
     */
    public static int loadAllThumbnailsFromCache(String cacheDir, String pipelineName, List<FXNode> nodes) {
        int loaded = 0;
        for (int i = 0; i < nodes.size(); i++) {
            if (loadThumbnailFromCache(cacheDir, pipelineName, nodes.get(i), i)) {
                loaded++;
            }
        }
        return loaded;
    }

    /**
     * Get the cache directory for a pipeline file.
     * Creates a .cache subdirectory next to the pipeline file.
     *
     * @param pipelinePath Path to the pipeline JSON file
     * @return Path to the cache directory
     */
    public static String getCacheDirectory(String pipelinePath) {
        File pipelineFile = new File(pipelinePath);
        File parentDir = pipelineFile.getParentFile();
        String baseName = pipelineFile.getName();
        if (baseName.endsWith(".json")) {
            baseName = baseName.substring(0, baseName.length() - 5);
        }
        return new File(parentDir, "." + baseName + "_cache").getAbsolutePath();
    }

    /**
     * Extract the pipeline name (base filename without .json extension) from a path.
     *
     * @param pipelinePath Path to the pipeline JSON file
     * @return Base name without extension
     */
    public static String getPipelineName(String pipelinePath) {
        File pipelineFile = new File(pipelinePath);
        String baseName = pipelineFile.getName();
        if (baseName.endsWith(".json")) {
            baseName = baseName.substring(0, baseName.length() - 5);
        }
        return baseName;
    }

    /**
     * Check if a saved label is an old-style default that should be updated.
     * Old files might have short labels like "Webcam" instead of "Webcam Source".
     */
    private static boolean isOldStyleDefaultLabel(String savedLabel, String type) {
        // Map of old-style labels to their node types
        // These are labels that were used as defaults but have since been renamed
        if (savedLabel == null) return false;

        switch (type) {
            case "WebcamSource":
                return "Webcam".equals(savedLabel);
            case "FileSource":
                return "File".equals(savedLabel) || "File Source".equals(savedLabel);
            // Blur nodes were renamed to have "Blur" suffix
            case "BilateralFilter":
                return "Bilateral".equals(savedLabel);
            case "MeanShift":
                return "Mean Shift".equals(savedLabel);
            // Add more mappings here as needed when node labels are updated
            default:
                return false;
        }
    }

    /**
     * Convert a Base64-encoded PNG string to JavaFX Image.
     */
    private static Image base64ToImage(String base64) {
        if (base64 == null || base64.isEmpty()) return null;
        try {
            byte[] imageBytes = Base64.getDecoder().decode(base64);
            ByteArrayInputStream bais = new ByteArrayInputStream(imageBytes);
            BufferedImage bufferedImage = ImageIO.read(bais);
            if (bufferedImage == null) return null;
            return SwingFXUtils.toFXImage(bufferedImage, null);
        } catch (Exception e) {
            System.err.println("Failed to decode thumbnail: " + e.getMessage());
            return null;
        }
    }

    /**
     * Serialize a list of nodes to a JsonArray.
     */
    private static JsonArray serializeNodes(List<FXNode> nodes) {
        JsonArray nodesArray = new JsonArray();
        for (int i = 0; i < nodes.size(); i++) {
            FXNode node = nodes.get(i);
            JsonObject nodeJson = new JsonObject();

            nodeJson.addProperty("id", node.id);
            nodeJson.addProperty("index", i);
            nodeJson.addProperty("type", node.nodeType);
            nodeJson.addProperty("label", node.label);
            nodeJson.addProperty("x", node.x);
            nodeJson.addProperty("y", node.y);
            // Don't save width/height - derived from node type
            nodeJson.addProperty("enabled", node.enabled);
            nodeJson.addProperty("workUnitsCompleted", node.workUnitsCompleted);
            nodeJson.addProperty("threadPriority", node.threadPriority);
            nodeJson.addProperty("inputCount", node.inputCount);
            nodeJson.addProperty("inputCount2", node.inputCount2);
            nodeJson.addProperty("outputCount1", node.outputCount1);
            nodeJson.addProperty("outputCount2", node.outputCount2);
            nodeJson.addProperty("outputCount3", node.outputCount3);
            nodeJson.addProperty("outputCount4", node.outputCount4);
            nodeJson.addProperty("hasInput", node.hasInput);
            nodeJson.addProperty("hasDualInput", node.hasDualInput);
            nodeJson.addProperty("queuesInSync", node.queuesInSync);
            nodeJson.addProperty("outputCount", node.outputCount);
            nodeJson.addProperty("isBoundaryNode", node.isBoundaryNode);

            if (node.backgroundColor != null) {
                nodeJson.addProperty("bgColorR", node.backgroundColor.getRed());
                nodeJson.addProperty("bgColorG", node.backgroundColor.getGreen());
                nodeJson.addProperty("bgColorB", node.backgroundColor.getBlue());
            }

            // Save generic properties map (for processing nodes like Gain, FFT, BitPlanes, etc.)
            saveNodeProperties(nodeJson, node);

            // Note: Thumbnails for inner nodes are not saved - they use the cache system at load time

            // Container-specific properties (for nested containers)
            nodeJson.addProperty("isContainer", node.isContainer);
            if (node.isContainer) {
                // Save pipeline file path if set (external sub-diagram file)
                if (node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty()) {
                    nodeJson.addProperty("pipelineFile", node.pipelineFilePath);
                }
                // Also save inline inner nodes if present (recursively)
                if (!node.innerNodes.isEmpty()) {
                    JsonArray innerNodesArray = serializeNodes(node.innerNodes);
                    nodeJson.add("innerNodes", innerNodesArray);

                    JsonArray innerConnectionsArray = serializeConnections(node.innerConnections, node.innerNodes);
                    nodeJson.add("innerConnections", innerConnectionsArray);
                }
            }

            nodesArray.add(nodeJson);
        }
        return nodesArray;
    }

    /**
     * Serialize a list of connections to a JsonArray.
     */
    private static JsonArray serializeConnections(List<FXConnection> connections, List<FXNode> nodes) {
        JsonArray connectionsArray = new JsonArray();
        for (FXConnection conn : connections) {
            JsonObject connJson = new JsonObject();

            // Use indexOf to find actual position in list, like main branch does
            int sourceIndex = conn.source != null ? nodes.indexOf(conn.source) : -1;
            connJson.addProperty("sourceIndex", sourceIndex);
            connJson.addProperty("sourceOutputIndex", conn.sourceOutputIndex);

            int targetIndex = conn.target != null ? nodes.indexOf(conn.target) : -1;
            connJson.addProperty("targetIndex", targetIndex);
            connJson.addProperty("targetInputIndex", conn.targetInputIndex);

            connectionsArray.add(connJson);
        }
        return connectionsArray;
    }

    /**
     * Deserialize a JsonArray into a list of nodes.
     */
    private static List<FXNode> deserializeNodes(JsonArray nodesArray) {
        return deserializeNodes(nodesArray, null);
    }

    /**
     * Deserialize a JsonArray into a list of nodes with parent path for external file resolution.
     */
    private static List<FXNode> deserializeNodes(JsonArray nodesArray, String parentPath) {
        List<FXNode> nodes = new ArrayList<>();
        for (JsonElement elem : nodesArray) {
            JsonObject nodeJson = elem.getAsJsonObject();

            String type = nodeJson.get("type").getAsString();
            double x = nodeJson.has("x") ? nodeJson.get("x").getAsDouble() : 0;
            double y = nodeJson.has("y") ? nodeJson.get("y").getAsDouble() : 0;

            FXNode node = FXNodeFactory.createFXNode(type, (int) x, (int) y);

            // Restore node ID if present (for inner node ID consistency)
            if (nodeJson.has("id")) {
                node.id = nodeJson.get("id").getAsInt();
            }

            if (nodeJson.has("label")) {
                String savedLabel = nodeJson.get("label").getAsString();
                if (!savedLabel.equals(type) && !savedLabel.equals(node.label)) {
                    if (!isOldStyleDefaultLabel(savedLabel, type)) {
                        node.label = savedLabel;
                    }
                }
            }

            node.x = x;
            node.y = y;

            if (nodeJson.has("enabled")) {
                node.enabled = nodeJson.get("enabled").getAsBoolean();
            }
            if (nodeJson.has("workUnitsCompleted")) {
                node.workUnitsCompleted = nodeJson.get("workUnitsCompleted").getAsLong();
            }
            if (nodeJson.has("threadPriority")) {
                node.threadPriority = nodeJson.get("threadPriority").getAsInt();
            }
            if (nodeJson.has("inputCount")) {
                node.inputCount = nodeJson.get("inputCount").getAsInt();
            }
            if (nodeJson.has("inputCount2")) {
                node.inputCount2 = nodeJson.get("inputCount2").getAsInt();
            }
            if (nodeJson.has("outputCount1")) {
                node.outputCount1 = nodeJson.get("outputCount1").getAsInt();
            }
            if (nodeJson.has("outputCount2")) {
                node.outputCount2 = nodeJson.get("outputCount2").getAsInt();
            }
            if (nodeJson.has("outputCount3")) {
                node.outputCount3 = nodeJson.get("outputCount3").getAsInt();
            }
            if (nodeJson.has("outputCount4")) {
                node.outputCount4 = nodeJson.get("outputCount4").getAsInt();
            }
            if (nodeJson.has("hasInput")) {
                node.hasInput = nodeJson.get("hasInput").getAsBoolean();
            }
            if (nodeJson.has("hasDualInput")) {
                node.hasDualInput = nodeJson.get("hasDualInput").getAsBoolean();
            }
            if (nodeJson.has("queuesInSync")) {
                node.queuesInSync = nodeJson.get("queuesInSync").getAsBoolean();
            }
            if (nodeJson.has("outputCount")) {
                node.outputCount = nodeJson.get("outputCount").getAsInt();
            }
            if (nodeJson.has("isBoundaryNode")) {
                node.isBoundaryNode = nodeJson.get("isBoundaryNode").getAsBoolean();
            }

            if (nodeJson.has("bgColorR") && nodeJson.has("bgColorG") && nodeJson.has("bgColorB")) {
                node.backgroundColor = Color.color(
                    nodeJson.get("bgColorR").getAsDouble(),
                    nodeJson.get("bgColorG").getAsDouble(),
                    nodeJson.get("bgColorB").getAsDouble()
                );
            }

            // Ensure boundary nodes use the standard container/boundary color
            if (node.isBoundaryNode) {
                node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
            }

            // Load generic properties for processing nodes
            loadNodeProperties(nodeJson, node);

            // For backwards compatibility: load embedded Base64 thumbnail from old files
            // New saves don't include embedded thumbnails
            if (nodeJson.has("thumbnail")) {
                String thumbnailBase64 = nodeJson.get("thumbnail").getAsString();
                node.thumbnail = base64ToImage(thumbnailBase64);
            }

            // Load container-specific properties (for nested containers)
            // Check both isContainer flag and nodeType for backwards compatibility with older files
            if (nodeJson.has("isContainer")) {
                node.isContainer = nodeJson.get("isContainer").getAsBoolean();
            }
            // Also mark as container if nodeType is "Container" (for files without isContainer flag)
            if ("Container".equals(type)) {
                node.isContainer = true;
            }
            if (node.isContainer) {
                node.backgroundColor = NodeRenderer.COLOR_CONTAINER_NODE;
                // Load pipeline file path if present
                if (nodeJson.has("pipelineFile")) {
                    node.pipelineFilePath = nodeJson.get("pipelineFile").getAsString();
                }
                // Load inner nodes: prioritize inline data (which may have thumbnails) over external file
                boolean loadedInline = false;
                if (nodeJson.has("innerNodes")) {
                    node.innerNodes = deserializeNodes(nodeJson.getAsJsonArray("innerNodes"), parentPath);
                    if (nodeJson.has("innerConnections")) {
                        node.innerConnections = deserializeConnections(nodeJson.getAsJsonArray("innerConnections"), node.innerNodes);
                    }
                    reassignInnerNodeIds(node.innerNodes);
                    loadedInline = true;
                    System.out.println("Loaded inline inner nodes for nested container: " + node.label +
                                       " (" + node.innerNodes.size() + " nodes)");
                }
                // Fall back to external pipeline file if no inline data and we have a parent path
                if (!loadedInline && node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty() && parentPath != null) {
                    // Resolve relative paths against the parent document's directory
                    String resolvedPath = node.pipelineFilePath;
                    File pipelineFile = new File(resolvedPath);
                    if (!pipelineFile.isAbsolute()) {
                        // Relative path - resolve against parent document's directory
                        File parentDir = new File(parentPath).getParentFile();
                        if (parentDir != null) {
                            pipelineFile = new File(parentDir, node.pipelineFilePath);
                            resolvedPath = pipelineFile.getAbsolutePath();
                        }
                    }
                    if (pipelineFile.exists()) {
                        try {
                            PipelineDocument externalDoc = load(resolvedPath);
                            node.innerNodes = externalDoc.nodes;
                            node.innerConnections = externalDoc.connections;
                            // Reassign IDs to inner nodes to avoid collisions with outer nodes
                            reassignInnerNodeIds(node.innerNodes);
                            System.out.println("Loaded external pipeline for nested container: " + resolvedPath);
                        } catch (IOException e) {
                            System.err.println("Failed to load external pipeline " + resolvedPath + ": " + e.getMessage());
                        }
                    } else {
                        System.err.println("External pipeline file not found: " + resolvedPath +
                            " (original: " + node.pipelineFilePath + ")");
                    }
                }
            }

            nodes.add(node);
        }
        return nodes;
    }

    /**
     * Deserialize a JsonArray into a list of connections.
     */
    private static List<FXConnection> deserializeConnections(JsonArray connectionsArray, List<FXNode> nodes) {
        List<FXConnection> connections = new ArrayList<>();
        for (JsonElement elem : connectionsArray) {
            JsonObject connJson = elem.getAsJsonObject();

            int sourceIndex = connJson.get("sourceIndex").getAsInt();
            int targetIndex = connJson.get("targetIndex").getAsInt();
            int sourceOutputIndex = connJson.has("sourceOutputIndex") ? connJson.get("sourceOutputIndex").getAsInt() : 0;
            int targetInputIndex = connJson.has("targetInputIndex") ? connJson.get("targetInputIndex").getAsInt() : 0;

            FXNode sourceNode = (sourceIndex >= 0 && sourceIndex < nodes.size()) ? nodes.get(sourceIndex) : null;
            FXNode targetNode = (targetIndex >= 0 && targetIndex < nodes.size()) ? nodes.get(targetIndex) : null;

            if (sourceNode != null && targetNode != null) {
                FXConnection conn = new FXConnection(sourceNode, sourceOutputIndex, targetNode, targetInputIndex);
                connections.add(conn);
            }
        }
        return connections;
    }

    /**
     * Reassign unique IDs to inner nodes to avoid collisions with outer pipeline nodes.
     * This is necessary because inner nodes loaded from external files may have IDs
     * that overlap with IDs already used in the main pipeline.
     *
     * @param innerNodes List of inner nodes to reassign IDs
     * @param existingNodes List of outer nodes to check for ID collisions (can be null)
     */
    private static void reassignInnerNodeIds(List<FXNode> innerNodes) {
        for (FXNode node : innerNodes) {
            node.reassignId();
            // Recursively handle nested containers
            if (node.isContainer && node.innerNodes != null && !node.innerNodes.isEmpty()) {
                reassignInnerNodeIds(node.innerNodes);
            }
        }
    }

    /**
     * Ensure all node IDs are unique by advancing the global ID counter past all existing IDs.
     * Call this after loading a pipeline to prevent ID collisions.
     */
    public static void ensureUniqueIds(List<FXNode> nodes) {
        int maxId = findMaxId(nodes);
        // Advance the global counter past the maximum ID
        while (FXNode.generateNewId() <= maxId) {
            // Keep generating until we're past the max
        }
    }

    /**
     * Find the maximum ID used in a list of nodes, including inner nodes.
     */
    private static int findMaxId(List<FXNode> nodes) {
        int maxId = 0;
        for (FXNode node : nodes) {
            if (node.id > maxId) {
                maxId = node.id;
            }
            // Check inner nodes recursively
            if (node.innerNodes != null && !node.innerNodes.isEmpty()) {
                int innerMax = findMaxId(node.innerNodes);
                if (innerMax > maxId) {
                    maxId = innerMax;
                }
            }
        }
        return maxId;
    }

    /**
     * Find the maximum ID used in a JSON array of nodes.
     * This is a pre-load scan to find all IDs before deserializing.
     */
    private static int findMaxIdInJson(JsonArray nodesArray) {
        int maxId = 0;
        for (JsonElement elem : nodesArray) {
            JsonObject nodeJson = elem.getAsJsonObject();
            if (nodeJson.has("id")) {
                int id = nodeJson.get("id").getAsInt();
                if (id > maxId) {
                    maxId = id;
                }
            }
            // Check inline inner nodes
            if (nodeJson.has("innerNodes")) {
                int innerMax = findMaxIdInJson(nodeJson.getAsJsonArray("innerNodes"));
                if (innerMax > maxId) {
                    maxId = innerMax;
                }
            }
        }
        return maxId;
    }

    /**
     * Save node-specific properties from FXNode.properties map into JSON.
     * This handles all processing node properties like radius, smoothness, gain, bitEnabled, etc.
     */
    private static void saveNodeProperties(JsonObject nodeJson, FXNode node) {
        if (node.properties == null || node.properties.isEmpty()) {
            return;
        }

        for (Map.Entry<String, Object> entry : node.properties.entrySet()) {
            String key = entry.getKey();
            Object value = entry.getValue();

            // Skip temporary UI controls (prefixed with underscore)
            if (key.startsWith("_")) {
                continue;
            }

            if (value instanceof Number) {
                if (value instanceof Double) {
                    nodeJson.addProperty(key, (Double) value);
                } else if (value instanceof Integer) {
                    nodeJson.addProperty(key, (Integer) value);
                } else if (value instanceof Long) {
                    nodeJson.addProperty(key, (Long) value);
                } else {
                    nodeJson.addProperty(key, ((Number) value).doubleValue());
                }
            } else if (value instanceof Boolean) {
                nodeJson.addProperty(key, (Boolean) value);
            } else if (value instanceof String) {
                nodeJson.addProperty(key, (String) value);
            } else if (value instanceof boolean[]) {
                boolean[] arr = (boolean[]) value;
                JsonArray jsonArr = new JsonArray();
                for (boolean b : arr) {
                    jsonArr.add(b);
                }
                nodeJson.add(key, jsonArr);
            } else if (value instanceof double[]) {
                double[] arr = (double[]) value;
                JsonArray jsonArr = new JsonArray();
                for (double d : arr) {
                    jsonArr.add(d);
                }
                nodeJson.add(key, jsonArr);
            } else if (value instanceof int[]) {
                int[] arr = (int[]) value;
                JsonArray jsonArr = new JsonArray();
                for (int i : arr) {
                    jsonArr.add(i);
                }
                nodeJson.add(key, jsonArr);
            }
        }
    }

    /**
     * Load node-specific properties from JSON into FXNode.properties map.
     * This handles all processing node properties like radius, smoothness, gain, bitEnabled, etc.
     */
    private static void loadNodeProperties(JsonObject nodeJson, FXNode node) {
        // List of known structural fields to skip (not processing properties)
        java.util.Set<String> skipFields = new java.util.HashSet<>(java.util.Arrays.asList(
            "id", "index", "type", "x", "y", "label", "enabled", "hasInput", "hasDualInput", "outputCount",
            "isBoundaryNode", "isContainer", "bgColorR", "bgColorG", "bgColorB", "thumbnail",
            "cameraIndex", "filePath", "fps", "pipelineFile", "innerNodes", "innerConnections",
            "threadPriority", "workUnitsCompleted", "inputReads1", "inputReads2", "name", "customName",
            "numOutputs", "containerName", "boundaryInputY", "boundaryOutputY",
            "inputCount", "inputCount2", "outputCount1", "outputCount2", "outputCount3", "outputCount4",
            "queuesInSync"
        ));

        for (java.util.Map.Entry<String, JsonElement> entry : nodeJson.entrySet()) {
            String key = entry.getKey();
            if (skipFields.contains(key)) {
                continue;
            }

            JsonElement value = entry.getValue();
            if (value.isJsonPrimitive()) {
                com.google.gson.JsonPrimitive prim = value.getAsJsonPrimitive();
                if (prim.isNumber()) {
                    // Store as Double for consistency
                    node.properties.put(key, prim.getAsDouble());
                } else if (prim.isBoolean()) {
                    node.properties.put(key, prim.getAsBoolean());
                } else if (prim.isString()) {
                    node.properties.put(key, prim.getAsString());
                }
            } else if (value.isJsonArray()) {
                // Handle arrays (like redBitEnabled, redBitGain, etc.)
                JsonArray arr = value.getAsJsonArray();
                if (arr.size() > 0) {
                    JsonElement first = arr.get(0);
                    if (first.isJsonPrimitive()) {
                        com.google.gson.JsonPrimitive prim = first.getAsJsonPrimitive();
                        if (prim.isBoolean()) {
                            // Boolean array
                            boolean[] boolArr = new boolean[arr.size()];
                            for (int i = 0; i < arr.size(); i++) {
                                boolArr[i] = arr.get(i).getAsBoolean();
                            }
                            node.properties.put(key, boolArr);
                        } else if (prim.isNumber()) {
                            // Double array
                            double[] doubleArr = new double[arr.size()];
                            for (int i = 0; i < arr.size(); i++) {
                                doubleArr[i] = arr.get(i).getAsDouble();
                            }
                            node.properties.put(key, doubleArr);
                        }
                    }
                }
            }
        }
    }
}
