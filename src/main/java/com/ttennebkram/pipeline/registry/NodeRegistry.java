package com.ttennebkram.pipeline.registry;

import com.ttennebkram.pipeline.nodes.PipelineNode;
import com.ttennebkram.pipeline.nodes.ProcessingNode;
import com.ttennebkram.pipeline.nodes.SourceNode;
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

    /**
     * Information about a registered node type.
     */
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

    private static final List<NodeRegistration> registeredNodes = new ArrayList<>();
    private static final Map<String, NodeRegistration> nodesByName = new HashMap<>();
    private static boolean initialized = false;

    /**
     * Initialize the registry by scanning for node classes.
     * Safe to call multiple times - only initializes once.
     */
    public static synchronized void initialize() {
        if (initialized) return;

        Set<Class<? extends PipelineNode>> nodeClasses = NodeScanner.findNodeClasses();

        for (Class<? extends PipelineNode> nodeClass : nodeClasses) {
            com.ttennebkram.pipeline.registry.NodeInfo info =
                nodeClass.getAnnotation(com.ttennebkram.pipeline.registry.NodeInfo.class);
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

        // Sort by category then name for consistent ordering
        registeredNodes.sort((a, b) -> {
            int catCmp = getCategoryOrder(a.category) - getCategoryOrder(b.category);
            if (catCmp != 0) return catCmp;
            return a.name.compareTo(b.name);
        });

        initialized = true;
        System.out.println("NodeRegistry: Discovered " + registeredNodes.size() + " node types");
    }

    /**
     * Get category display order (for toolbar).
     */
    private static int getCategoryOrder(String category) {
        switch (category) {
            case "Source": return 0;
            case "Basic": return 1;
            case "Blur": return 2;
            case "Edge Detection": return 3;
            case "Morphology": return 4;
            case "Filter": return 5;
            case "Transform": return 6;
            case "Detection": return 7;
            case "Content": return 8;
            case "Visualization": return 9;
            case "Dual Input": return 10;
            default: return 99;
        }
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
     * Get all categories in display order.
     */
    public static List<String> getCategories() {
        initialize();
        Set<String> categories = new LinkedHashSet<>();
        for (NodeRegistration reg : registeredNodes) {
            categories.add(reg.category);
        }
        List<String> result = new ArrayList<>(categories);
        result.sort((a, b) -> getCategoryOrder(a) - getCategoryOrder(b));
        return result;
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
    public static SourceNode createSourceNode(String type, Shell shell, Display display, Canvas canvas, int x, int y) {
        initialize();
        NodeRegistration reg = nodesByName.get(type);
        if (reg == null) {
            System.err.println("Unknown source node type: " + type);
            return null;
        }

        if (!SourceNode.class.isAssignableFrom(reg.nodeClass)) {
            System.err.println("Not a SourceNode: " + type);
            return null;
        }

        try {
            // Try constructor with Canvas (FileSourceNode, WebcamSourceNode)
            try {
                Constructor<? extends PipelineNode> constructor =
                    reg.nodeClass.getConstructor(Shell.class, Display.class, Canvas.class, int.class, int.class);
                return (SourceNode) constructor.newInstance(shell, display, canvas, x, y);
            } catch (NoSuchMethodException e) {
                // Try constructor without Canvas (BlankSourceNode)
                Constructor<? extends PipelineNode> constructor =
                    reg.nodeClass.getConstructor(Shell.class, Display.class, int.class, int.class);
                return (SourceNode) constructor.newInstance(shell, display, x, y);
            }
        } catch (Exception e) {
            System.err.println("Failed to create source node: " + type + " - " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Create any PipelineNode by type name (auto-detects Source vs Processing).
     */
    public static PipelineNode createNode(String type, Display display, Shell shell, Canvas canvas, int x, int y) {
        initialize();
        NodeRegistration reg = nodesByName.get(type);
        if (reg == null) {
            System.err.println("Unknown node type: " + type);
            return null;
        }

        if (SourceNode.class.isAssignableFrom(reg.nodeClass)) {
            return createSourceNode(type, shell, display, canvas, x, y);
        } else {
            return createProcessingNode(type, display, shell, x, y);
        }
    }

    /**
     * Check if a node type is registered.
     */
    public static boolean isRegistered(String type) {
        initialize();
        return nodesByName.containsKey(type);
    }

    /**
     * Get registration info for a type name.
     */
    public static NodeRegistration getRegistration(String type) {
        initialize();
        return nodesByName.get(type);
    }

    // Legacy compatibility methods

    /**
     * @deprecated Use getAllNodes() instead
     */
    @Deprecated
    public static class NodeInfo {
        public final String name;
        public final String category;
        public final Class<? extends ProcessingNode> nodeClass;

        @SuppressWarnings("unchecked")
        public NodeInfo(NodeRegistration reg) {
            this.name = reg.name;
            this.category = reg.category;
            this.nodeClass = (Class<? extends ProcessingNode>) reg.nodeClass;
        }
    }

    /**
     * @deprecated Use createProcessingNode() instead
     */
    @Deprecated
    public static ProcessingNode createNode(String type, Display display, Shell shell, int x, int y) {
        return createProcessingNode(type, display, shell, x, y);
    }
}
