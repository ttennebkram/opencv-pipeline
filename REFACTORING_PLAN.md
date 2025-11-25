# Serialization Refactoring Plan

## Goals

1. Each node class handles its own serialization/deserialization
2. Node classes declare their own metadata (aliases, category, name) via annotations
3. Nodes are discovered at runtime via classpath scanning (pure Java, no external deps)
4. Base class auto-generates "Untitled: NodeName" alias
5. Package renamed from `com.example.pipeline` to `com.ttennebkram.pipeline`
6. PipelineEditor reduced by ~900+ lines

---

## Phase 1: Package Rename

Rename all packages from `com.example.pipeline` to `com.ttennebkram.pipeline`:

```
src/main/java/com/example/pipeline/  →  src/main/java/com/ttennebkram/pipeline/
```

Files affected:
- `PipelineEditor.java`
- `registry/NodeRegistry.java`
- `nodes/*.java` (all 60 node files)
- `Connection.java`, `DanglingConnection.java`, etc.

Update `pom.xml` to reflect new groupId and main class.

---

## Phase 2: Create Annotations & Interfaces

### 2.1 Create `@NodeInfo` Annotation

`src/main/java/com/ttennebkram/pipeline/registry/NodeInfo.java`:

```java
package com.ttennebkram.pipeline.registry;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for pipeline node classes to declare their metadata.
 * Used for auto-registration and serialization.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface NodeInfo {
    /** Canonical name used for serialization (e.g., "GaussianBlur") */
    String name();

    /** Category for toolbar grouping (e.g., "Blur", "Edge Detection") */
    String category();

    /** Additional aliases for backward compatibility (e.g., {"Gaussian Blur"}) */
    String[] aliases() default {};
}
```

### 2.2 Create `NodeSerializable` Interface

`src/main/java/com/ttennebkram/pipeline/serialization/NodeSerializable.java`:

```java
package com.ttennebkram.pipeline.serialization;

import com.google.gson.JsonObject;

/**
 * Interface for nodes that can serialize/deserialize their properties.
 */
public interface NodeSerializable {

    /**
     * Serialize node-specific properties to JSON.
     * Common properties (x, y, threadPriority, etc.) are handled by base class.
     */
    void serializeProperties(JsonObject json);

    /**
     * Deserialize node-specific properties from JSON.
     * Common properties (x, y, threadPriority, etc.) are handled by base class.
     */
    void deserializeProperties(JsonObject json);

    /**
     * Get the canonical type name for serialization.
     * Default implementation reads from @NodeInfo annotation.
     */
    default String getSerializationType() {
        NodeInfo info = getClass().getAnnotation(NodeInfo.class);
        return info != null ? info.name() : getClass().getSimpleName().replace("Node", "");
    }
}
```

---

## Phase 3: Create Node Scanner (Pure Java)

`src/main/java/com/ttennebkram/pipeline/registry/NodeScanner.java`:

```java
package com.ttennebkram.pipeline.registry;

import com.ttennebkram.pipeline.nodes.PipelineNode;

import java.io.File;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.util.*;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Scans the classpath at runtime to discover all node classes annotated with @NodeInfo.
 * Works from both filesystem (IDE) and JAR (production).
 */
public class NodeScanner {

    private static final String NODES_PACKAGE = "com.ttennebkram.pipeline.nodes";

    /**
     * Find all concrete node classes annotated with @NodeInfo.
     */
    public static Set<Class<? extends PipelineNode>> findNodeClasses() {
        Set<Class<? extends PipelineNode>> nodeClasses = new LinkedHashSet<>();
        String path = NODES_PACKAGE.replace('.', '/');
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();

        try {
            Enumeration<URL> resources = classLoader.getResources(path);
            while (resources.hasMoreElements()) {
                URL resource = resources.nextElement();
                String protocol = resource.getProtocol();

                if ("file".equals(protocol)) {
                    // Running from filesystem (IDE/development)
                    scanDirectory(new File(resource.toURI()), NODES_PACKAGE, nodeClasses);
                } else if ("jar".equals(protocol)) {
                    // Running from JAR (production)
                    scanJar(resource, path, nodeClasses);
                }
            }
        } catch (Exception e) {
            System.err.println("Error scanning for node classes: " + e.getMessage());
            e.printStackTrace();
        }

        return nodeClasses;
    }

    private static void scanDirectory(File directory, String packageName,
                                       Set<Class<? extends PipelineNode>> result) {
        if (!directory.exists()) return;

        File[] files = directory.listFiles();
        if (files == null) return;

        for (File file : files) {
            if (file.isDirectory()) {
                scanDirectory(file, packageName + "." + file.getName(), result);
            } else if (file.getName().endsWith("Node.class")) {
                String className = packageName + "." + file.getName().replace(".class", "");
                tryLoadNodeClass(className, result);
            }
        }
    }

    private static void scanJar(URL jarUrl, String packagePath,
                                 Set<Class<? extends PipelineNode>> result) {
        try {
            // Extract JAR path from URL like "jar:file:/path/to.jar!/com/..."
            String jarPath = jarUrl.getPath();
            jarPath = jarPath.substring(5, jarPath.indexOf("!"));

            try (JarFile jarFile = new JarFile(jarPath)) {
                Enumeration<JarEntry> entries = jarFile.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    String name = entry.getName();

                    if (name.startsWith(packagePath) && name.endsWith("Node.class")) {
                        String className = name.replace('/', '.').replace(".class", "");
                        tryLoadNodeClass(className, result);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error scanning JAR: " + e.getMessage());
        }
    }

    private static void tryLoadNodeClass(String className,
                                          Set<Class<? extends PipelineNode>> result) {
        try {
            Class<?> clazz = Class.forName(className);

            // Must be a PipelineNode subclass
            if (!PipelineNode.class.isAssignableFrom(clazz)) return;

            // Must not be abstract
            if (Modifier.isAbstract(clazz.getModifiers())) return;

            // Must have @NodeInfo annotation
            if (!clazz.isAnnotationPresent(NodeInfo.class)) return;

            @SuppressWarnings("unchecked")
            Class<? extends PipelineNode> nodeClass = (Class<? extends PipelineNode>) clazz;
            result.add(nodeClass);

        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            // Skip classes that can't be loaded
        }
    }
}
```

---

## Phase 4: Update Base Classes

### 4.1 Update `PipelineNode`

Add to `PipelineNode.java`:

```java
public abstract class PipelineNode implements NodeSerializable {

    // ... existing fields ...

    /**
     * Serialize common properties shared by all nodes.
     * Called by PipelineSerializer before serializeProperties().
     */
    public void serializeCommon(JsonObject json) {
        json.addProperty("type", getSerializationType());
        json.addProperty("x", x);
        json.addProperty("y", y);
        json.addProperty("threadPriority", threadPriority);
        json.addProperty("workUnitsCompleted", workUnitsCompleted);
    }

    /**
     * Deserialize common properties shared by all nodes.
     * Called by PipelineSerializer before deserializeProperties().
     */
    public void deserializeCommon(JsonObject json) {
        if (json.has("x")) x = json.get("x").getAsInt();
        if (json.has("y")) y = json.get("y").getAsInt();
        if (json.has("threadPriority")) {
            threadPriority = json.get("threadPriority").getAsInt();
            originalPriority = threadPriority;
            lastRunningPriority = threadPriority;
        }
        if (json.has("workUnitsCompleted")) {
            workUnitsCompleted = json.get("workUnitsCompleted").getAsLong();
        }
    }

    // Default implementation - subclasses override if they have properties
    @Override
    public void serializeProperties(JsonObject json) {
        // No additional properties by default
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        // No additional properties by default
    }
}
```

### 4.2 Update `ProcessingNode`

Add `name` to common serialization:

```java
@Override
public void serializeCommon(JsonObject json) {
    super.serializeCommon(json);
    json.addProperty("name", name);
}

@Override
public void deserializeCommon(JsonObject json) {
    super.deserializeCommon(json);
    // name is set by constructor, no need to deserialize
}
```

---

## Phase 5: Refactor NodeRegistry

Replace manual registration with auto-discovery:

```java
package com.ttennebkram.pipeline.registry;

import com.ttennebkram.pipeline.nodes.PipelineNode;
import com.ttennebkram.pipeline.nodes.ProcessingNode;
import org.eclipse.swt.widgets.Canvas;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 * Registry for discovering and creating pipeline nodes.
 * Auto-discovers nodes via classpath scanning at initialization.
 */
public class NodeRegistry {

    private static final List<NodeRegistration> registeredNodes = new ArrayList<>();
    private static final Map<String, NodeRegistration> nodesByName = new HashMap<>();
    private static boolean initialized = false;

    public static class NodeRegistration {
        public final String name;
        public final String category;
        public final Class<? extends PipelineNode> nodeClass;
        public final String[] aliases;

        public NodeRegistration(String name, String category,
                                Class<? extends PipelineNode> nodeClass, String[] aliases) {
            this.name = name;
            this.category = category;
            this.nodeClass = nodeClass;
            this.aliases = aliases;
        }
    }

    /**
     * Initialize the registry by scanning for node classes.
     * Safe to call multiple times - only initializes once.
     */
    public static synchronized void initialize() {
        if (initialized) return;

        Set<Class<? extends PipelineNode>> nodeClasses = NodeScanner.findNodeClasses();

        for (Class<? extends PipelineNode> nodeClass : nodeClasses) {
            NodeInfo info = nodeClass.getAnnotation(NodeInfo.class);
            if (info == null) continue;

            // Build complete alias list
            List<String> allAliases = new ArrayList<>();
            allAliases.addAll(Arrays.asList(info.aliases()));
            allAliases.add("Untitled: " + info.name());  // Auto-generated
            allAliases.add("Unknown: " + info.name());   // Legacy support

            NodeRegistration registration = new NodeRegistration(
                info.name(),
                info.category(),
                nodeClass,
                allAliases.toArray(new String[0])
            );

            registeredNodes.add(registration);

            // Register canonical name
            nodesByName.put(info.name(), registration);

            // Register without spaces for compatibility
            String noSpaces = info.name().replace(" ", "");
            if (!noSpaces.equals(info.name())) {
                nodesByName.put(noSpaces, registration);
            }

            // Register all aliases
            for (String alias : allAliases) {
                nodesByName.put(alias, registration);
            }
        }

        initialized = true;
        System.out.println("NodeRegistry: Discovered " + registeredNodes.size() + " node types");
    }

    /**
     * Get all registered nodes.
     */
    public static List<NodeRegistration> getAllNodes() {
        initialize();
        return Collections.unmodifiableList(registeredNodes);
    }

    /**
     * Get nodes by category.
     */
    public static List<NodeRegistration> getNodesByCategory(String category) {
        initialize();
        List<NodeRegistration> result = new ArrayList<>();
        for (NodeRegistration reg : registeredNodes) {
            if (reg.category.equals(category)) {
                result.add(reg);
            }
        }
        return result;
    }

    /**
     * Get all categories in order of first appearance.
     */
    public static List<String> getCategories() {
        initialize();
        Set<String> categories = new LinkedHashSet<>();
        for (NodeRegistration reg : registeredNodes) {
            categories.add(reg.category);
        }
        return new ArrayList<>(categories);
    }

    /**
     * Create a ProcessingNode by type name.
     */
    public static ProcessingNode createProcessingNode(String type, Display display, Shell shell, int x, int y) {
        initialize();
        NodeRegistration reg = nodesByName.get(type);
        if (reg == null) {
            System.err.println("Unknown node type: " + type);
            return null;
        }

        if (!ProcessingNode.class.isAssignableFrom(reg.nodeClass)) {
            System.err.println("Not a ProcessingNode: " + type);
            return null;
        }

        try {
            Constructor<? extends PipelineNode> constructor =
                reg.nodeClass.getConstructor(Display.class, Shell.class, int.class, int.class);
            return (ProcessingNode) constructor.newInstance(display, shell, x, y);
        } catch (Exception e) {
            System.err.println("Failed to create node: " + type + " - " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Create a SourceNode by type name.
     */
    public static PipelineNode createSourceNode(String type, Shell shell, Display display, Canvas canvas, int x, int y) {
        initialize();
        NodeRegistration reg = nodesByName.get(type);
        if (reg == null) {
            System.err.println("Unknown source node type: " + type);
            return null;
        }

        try {
            // Try different constructor signatures
            try {
                Constructor<? extends PipelineNode> constructor =
                    reg.nodeClass.getConstructor(Shell.class, Display.class, Canvas.class, int.class, int.class);
                return constructor.newInstance(shell, display, canvas, x, y);
            } catch (NoSuchMethodException e) {
                // Try alternate signature
                Constructor<? extends PipelineNode> constructor =
                    reg.nodeClass.getConstructor(Shell.class, Display.class, int.class, int.class);
                return constructor.newInstance(shell, display, x, y);
            }
        } catch (Exception e) {
            System.err.println("Failed to create source node: " + type + " - " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Check if a node type is registered.
     */
    public static boolean isRegistered(String type) {
        initialize();
        return nodesByName.containsKey(type);
    }
}
```

---

## Phase 6: Create PipelineSerializer

`src/main/java/com/ttennebkram/pipeline/serialization/PipelineSerializer.java`:

```java
package com.ttennebkram.pipeline.serialization;

import com.google.gson.*;
import com.ttennebkram.pipeline.Connection;
import com.ttennebkram.pipeline.DanglingConnection;
import com.ttennebkram.pipeline.FreeConnection;
import com.ttennebkram.pipeline.ReverseDanglingConnection;
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
        for (PipelineNode node : nodes) {
            JsonObject nodeJson = new JsonObject();

            // Common properties (includes type)
            node.serializeCommon(nodeJson);

            // Node-specific properties
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
            if (conn.queue != null) {
                connJson.addProperty("queueCapacity", conn.queue.remainingCapacity() + conn.queue.size());
                connJson.addProperty("queueSize", conn.queue.size());
            }
            connectionsArray.add(connJson);
        }
        root.add("connections", connectionsArray);

        // Serialize dangling connections
        JsonArray danglingArray = new JsonArray();
        for (DanglingConnection dc : danglingConnections) {
            JsonObject dcJson = new JsonObject();
            dcJson.addProperty("sourceId", nodes.indexOf(dc.source));
            dcJson.addProperty("endX", dc.endX);
            dcJson.addProperty("endY", dc.endY);
            danglingArray.add(dcJson);
        }
        root.add("danglingConnections", danglingArray);

        // Serialize reverse dangling connections
        JsonArray reverseDanglingArray = new JsonArray();
        for (ReverseDanglingConnection rdc : reverseDanglingConnections) {
            JsonObject rdcJson = new JsonObject();
            rdcJson.addProperty("targetId", nodes.indexOf(rdc.target));
            rdcJson.addProperty("inputIndex", rdc.inputIndex);
            rdcJson.addProperty("startX", rdc.startX);
            rdcJson.addProperty("startY", rdc.startY);
            reverseDanglingArray.add(rdcJson);
        }
        root.add("reverseDanglingConnections", reverseDanglingArray);

        // Serialize free connections
        JsonArray freeArray = new JsonArray();
        for (FreeConnection fc : freeConnections) {
            JsonObject fcJson = new JsonObject();
            fcJson.addProperty("startX", fc.startX);
            fcJson.addProperty("startY", fc.startY);
            fcJson.addProperty("endX", fc.endX);
            fcJson.addProperty("endY", fc.endY);
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

                PipelineNode node = createNode(type, display, shell, canvas, x, y);

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
                int inputIndex = connJson.has("inputIndex") ? connJson.get("inputIndex").getAsInt() : 0;

                if (sourceId >= 0 && sourceId < nodes.size() &&
                    targetId >= 0 && targetId < nodes.size()) {

                    Connection conn = new Connection(nodes.get(sourceId), nodes.get(targetId), inputIndex);

                    // Restore queue capacity if specified
                    if (connJson.has("queueCapacity")) {
                        int capacity = connJson.get("queueCapacity").getAsInt();
                        conn.setQueueCapacity(capacity);
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
                int endX = dcJson.get("endX").getAsInt();
                int endY = dcJson.get("endY").getAsInt();

                if (sourceId >= 0 && sourceId < nodes.size()) {
                    danglingConnections.add(new DanglingConnection(nodes.get(sourceId), endX, endY));
                }
            }
        }

        // Deserialize reverse dangling connections
        if (root.has("reverseDanglingConnections")) {
            for (JsonElement elem : root.getAsJsonArray("reverseDanglingConnections")) {
                JsonObject rdcJson = elem.getAsJsonObject();
                int targetId = rdcJson.get("targetId").getAsInt();
                int inputIndex = rdcJson.has("inputIndex") ? rdcJson.get("inputIndex").getAsInt() : 0;
                int startX = rdcJson.get("startX").getAsInt();
                int startY = rdcJson.get("startY").getAsInt();

                if (targetId >= 0 && targetId < nodes.size()) {
                    reverseDanglingConnections.add(
                        new ReverseDanglingConnection(nodes.get(targetId), inputIndex, startX, startY));
                }
            }
        }

        // Deserialize free connections
        if (root.has("freeConnections")) {
            for (JsonElement elem : root.getAsJsonArray("freeConnections")) {
                JsonObject fcJson = elem.getAsJsonObject();
                int startX = fcJson.get("startX").getAsInt();
                int startY = fcJson.get("startY").getAsInt();
                int endX = fcJson.get("endX").getAsInt();
                int endY = fcJson.get("endY").getAsInt();

                freeConnections.add(new FreeConnection(startX, startY, endX, endY));
            }
        }

        return new PipelineDocument(nodes, connections, danglingConnections,
                                    reverseDanglingConnections, freeConnections);
    }

    /**
     * Create a node by type, handling both source and processing nodes.
     */
    private static PipelineNode createNode(String type, Display display, Shell shell, Canvas canvas, int x, int y) {
        // Check for source node types
        if ("FileSource".equals(type) || "FileSourceNode".equals(type)) {
            return new FileSourceNode(shell, display, canvas, x, y);
        } else if ("WebcamSource".equals(type) || "WebcamSourceNode".equals(type)) {
            return new WebcamSourceNode(shell, display, canvas, x, y);
        } else if ("BlankSource".equals(type) || "BlankSourceNode".equals(type)) {
            return new BlankSourceNode(shell, display, x, y);
        }

        // Try as processing node
        return NodeRegistry.createProcessingNode(type, display, shell, x, y);
    }
}
```

---

## Phase 7: Annotate All Node Classes

### 7.1 Example Transformations

**GaussianBlurNode.java:**

```java
@NodeInfo(
    name = "GaussianBlur",
    category = "Blur",
    aliases = {"Gaussian Blur"}
)
public class GaussianBlurNode extends ProcessingNode {
    private int kernelSizeX = 7;
    private int kernelSizeY = 7;
    private double sigmaX = 0.0;

    // ... existing constructor, process(), showPropertiesDialog() ...

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("kernelSizeX", kernelSizeX);
        json.addProperty("kernelSizeY", kernelSizeY);
        json.addProperty("sigmaX", sigmaX);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("kernelSizeX")) kernelSizeX = json.get("kernelSizeX").getAsInt();
        if (json.has("kernelSizeY")) kernelSizeY = json.get("kernelSizeY").getAsInt();
        if (json.has("sigmaX")) sigmaX = json.get("sigmaX").getAsDouble();
    }
}
```

**FileSourceNode.java:**

```java
@NodeInfo(
    name = "FileSource",
    category = "Source",
    aliases = {"File Source", "Image Source"}
)
public class FileSourceNode extends SourceNode {
    // ... existing fields ...

    @Override
    public void serializeProperties(JsonObject json) {
        if (imagePath != null) json.addProperty("imagePath", imagePath);
        json.addProperty("fpsMode", fpsMode);
        json.addProperty("loopVideo", loopVideo);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("imagePath")) {
            imagePath = json.get("imagePath").getAsString();
            loadMedia(imagePath);
        }
        if (json.has("fpsMode")) fpsMode = json.get("fpsMode").getAsInt();
        if (json.has("loopVideo")) loopVideo = json.get("loopVideo").getAsBoolean();
    }
}
```

### 7.2 Complete Node Annotation List

| Node Class | name | category | aliases |
|------------|------|----------|---------|
| GaussianBlurNode | GaussianBlur | Blur | Gaussian Blur |
| BoxBlurNode | BoxBlur | Blur | Box Blur |
| MedianBlurNode | MedianBlur | Blur | Median Blur |
| BilateralFilterNode | BilateralFilter | Blur | Bilateral Filter |
| MeanShiftFilterNode | MeanShift | Blur | Mean Shift, Mean Shift Blur |
| CannyEdgeNode | CannyEdge | Edge Detection | Canny Edge, Canny Edges |
| LaplacianNode | Laplacian | Edge Detection | Laplacian Edges |
| SobelNode | Sobel | Edge Detection | Sobel Edges |
| ScharrNode | Scharr | Edge Detection | Scharr Edges |
| GrayscaleNode | Grayscale | Basic | Grayscale/Color Convert |
| ThresholdNode | Threshold | Basic | |
| AdaptiveThresholdNode | AdaptiveThreshold | Basic | Adaptive Threshold |
| InvertNode | Invert | Basic | |
| GainNode | Gain | Basic | |
| CLAHENode | CLAHE | Basic | |
| BitPlanesColorNode | BitPlanesColor | Basic | Bit Planes Color |
| BitPlanesGrayscaleNode | BitPlanesGrayscale | Basic | Bit Planes Grayscale |
| ContoursNode | Contours | Detection | |
| BlobDetectorNode | BlobDetector | Detection | Blob Detector |
| HoughCirclesNode | HoughCircles | Detection | Hough Circles |
| HoughLinesNode | HoughLines | Detection | Hough Lines |
| HarrisCornersNode | HarrisCorners | Detection | Harris Corners |
| ShiTomasiCornersNode | ShiTomasi | Detection | Shi-Tomasi, Shi-Tomasi Corners |
| ConnectedComponentsNode | ConnectedComponents | Detection | Connected Components |
| ORBFeaturesNode | ORBFeatures | Detection | ORB Features |
| SIFTFeaturesNode | SIFTFeatures | Detection | SIFT Features |
| MatchTemplateNode | MatchTemplate | Detection | Match Template |
| ErodeNode | Erode | Morphology | |
| DilateNode | Dilate | Morphology | |
| MorphOpenNode | MorphOpen | Morphology | Morph Open |
| MorphCloseNode | MorphClose | Morphology | Morph Close |
| MorphologyExNode | MorphologyEx | Morphology | Morphology Ex |
| BitwiseNotNode | BitwiseNot | Filter | Bitwise NOT |
| ColorInRangeNode | ColorInRange | Filter | Color In Range |
| Filter2DNode | Filter2D | Filter | Filter 2D |
| FFTHighPassFilterNode | FFTHighPass | Filter | FFT High-Pass Filter |
| FFTLowPassFilterNode | FFTLowPass | Filter | FFT Low-Pass Filter |
| CropNode | Crop | Transform | |
| WarpAffineNode | WarpAffine | Transform | Warp Affine |
| RectangleNode | Rectangle | Content | |
| CircleNode | Circle | Content | |
| EllipseNode | Ellipse | Content | |
| LineNode | Line | Content | |
| ArrowNode | Arrow | Content | |
| TextNode | Text | Content | |
| HistogramNode | Histogram | Visualization | |
| AddClampNode | AddClamp | Dual Input | Add Clamp |
| AddWeightedNode | AddWeighted | Dual Input | Add Weighted |
| SubtractClampNode | SubtractClamp | Dual Input | Subtract Clamp |
| BitwiseAndNode | BitwiseAnd | Dual Input | Bitwise And |
| BitwiseOrNode | BitwiseOr | Dual Input | Bitwise Or |
| BitwiseXorNode | BitwiseXor | Dual Input | Bitwise Xor |
| FileSourceNode | FileSource | Source | File Source |
| WebcamSourceNode | WebcamSource | Source | Webcam Source |
| BlankSourceNode | BlankSource | Source | Blank Source |

---

## Phase 8: Update PipelineEditor

### 8.1 Remove Old Code

Delete from PipelineEditor.java:
- `registerNodes()` method (~120 lines)
- Static initializer calling `registerNodes()`
- `loadDiagramFromPath()` implementation (~500 lines) - replace with delegation
- `saveDiagramToPath()` implementation (~375 lines) - replace with delegation

### 8.2 New Implementation

```java
public class PipelineEditor {

    // Remove: static { registerNodes(); }

    // Add: Initialize registry on startup
    static {
        NodeRegistry.initialize();
    }

    // New simplified load
    private void loadDiagramFromPath(String path) {
        try {
            PipelineSerializer.PipelineDocument doc =
                PipelineSerializer.load(path, display, shell, canvas);

            // Clear existing
            clearPipeline();

            // Apply loaded data
            nodes.addAll(doc.nodes);
            connections.addAll(doc.connections);
            danglingConnections.addAll(doc.danglingConnections);
            reverseDanglingConnections.addAll(doc.reverseDanglingConnections);
            freeConnections.addAll(doc.freeConnections);

            // Load thumbnails from cache
            String cacheDir = getCacheDir(path);
            for (int i = 0; i < nodes.size(); i++) {
                PipelineNode node = nodes.get(i);
                if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).loadThumbnailFromCache(cacheDir, i);
                } else if (node instanceof WebcamSourceNode) {
                    ((WebcamSourceNode) node).loadThumbnailFromCache(cacheDir);
                }
            }

            canvas.redraw();
            updateTitle(path);

        } catch (IOException e) {
            showError("Failed to load pipeline: " + e.getMessage());
        }
    }

    // New simplified save
    private void saveDiagramToPath(String path) {
        try {
            PipelineSerializer.save(path, nodes, connections,
                danglingConnections, reverseDanglingConnections, freeConnections);

            // Save thumbnails to cache
            String cacheDir = getCacheDir(path);
            for (int i = 0; i < nodes.size(); i++) {
                PipelineNode node = nodes.get(i);
                if (node instanceof ProcessingNode) {
                    ((ProcessingNode) node).saveThumbnailToCache(cacheDir, i);
                }
            }

            updateTitle(path);

        } catch (IOException e) {
            showError("Failed to save pipeline: " + e.getMessage());
        }
    }
}
```

---

## Implementation Order

| Step | Task | Files | Impact |
|------|------|-------|--------|
| 1 | Rename packages to com.ttennebkram | All | Foundation |
| 2 | Create @NodeInfo annotation | 1 new file | Foundation |
| 3 | Create NodeSerializable interface | 1 new file | Foundation |
| 4 | Create NodeScanner | 1 new file | Foundation |
| 5 | Update PipelineNode base class | 1 file | Foundation |
| 6 | Update ProcessingNode base class | 1 file | Foundation |
| 7 | Update SourceNode base class | 1 file | Foundation |
| 8 | Annotate + add ser/deser to all 52 ProcessingNodes | 52 files | Bulk work |
| 9 | Annotate + add ser/deser to 3 SourceNodes | 3 files | Bulk work |
| 10 | Refactor NodeRegistry for auto-discovery | 1 file | Integration |
| 11 | Create PipelineSerializer | 1 new file | Integration |
| 12 | Update PipelineEditor to use new system | 1 file | Integration |
| 13 | Test loading existing .json files | - | Validation |
| 14 | Clean up: delete old code | 1 file | Cleanup |

---

## Backward Compatibility

Existing `.json` pipeline files will continue to work because:

1. **Type field preserved** - Same "type" values in JSON
2. **All aliases registered** - Including "Unknown: X" and "Untitled: X" patterns
3. **Property names unchanged** - Same JSON property names
4. **Only code organization changes** - Serialized format is identical

---

## Testing Checklist

- [ ] Load existing pipeline files saved with old code
- [ ] Save new pipelines and reload them
- [ ] Verify all 52 node types serialize/deserialize correctly
- [ ] Verify aliases work (e.g., "Gaussian Blur" loads as GaussianBlurNode)
- [ ] Test from IDE (filesystem scanning)
- [ ] Test from JAR (jar scanning)
- [ ] Verify toolbar still shows all nodes in correct categories
