package com.example.pipeline.registry;

import com.example.pipeline.nodes.ProcessingNode;
import org.eclipse.swt.widgets.Display;
import org.eclipse.swt.widgets.Shell;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 * Registry for discovering and creating pipeline nodes.
 */
public class NodeRegistry {

    // Node info for registration
    public static class NodeInfo {
        public final String name;
        public final String category;
        public final Class<? extends ProcessingNode> nodeClass;

        public NodeInfo(String name, String category, Class<? extends ProcessingNode> nodeClass) {
            this.name = name;
            this.category = category;
            this.nodeClass = nodeClass;
        }
    }

    private static final List<NodeInfo> registeredNodes = new ArrayList<>();
    private static final Map<String, NodeInfo> nodesByName = new HashMap<>();

    /**
     * Register a node type.
     */
    public static void register(String name, String category, Class<? extends ProcessingNode> nodeClass) {
        NodeInfo info = new NodeInfo(name, category, nodeClass);
        registeredNodes.add(info);
        nodesByName.put(name, info);
        // Also register without spaces for compatibility
        String noSpaces = name.replace(" ", "");
        if (!noSpaces.equals(name)) {
            nodesByName.put(noSpaces, info);
        }
    }

    /**
     * Register an alias for an existing node type (for backward compatibility).
     */
    public static void registerAlias(String alias, String existingName) {
        NodeInfo info = nodesByName.get(existingName);
        if (info != null) {
            nodesByName.put(alias, info);
        }
    }

    /**
     * Get all registered nodes.
     */
    public static List<NodeInfo> getAllNodes() {
        return Collections.unmodifiableList(registeredNodes);
    }

    /**
     * Get nodes by category.
     */
    public static List<NodeInfo> getNodesByCategory(String category) {
        List<NodeInfo> result = new ArrayList<>();
        for (NodeInfo info : registeredNodes) {
            if (info.category.equals(category)) {
                result.add(info);
            }
        }
        return result;
    }

    /**
     * Get all categories.
     */
    public static List<String> getCategories() {
        Set<String> categories = new LinkedHashSet<>();
        for (NodeInfo info : registeredNodes) {
            categories.add(info.category);
        }
        return new ArrayList<>(categories);
    }

    /**
     * Create a node instance by type name.
     */
    public static ProcessingNode createNode(String type, Display display, Shell shell, int x, int y) {
        NodeInfo info = nodesByName.get(type);
        if (info == null) {
            System.err.println("Unknown node type: " + type);
            return null;
        }

        try {
            Constructor<? extends ProcessingNode> constructor =
                info.nodeClass.getConstructor(Display.class, Shell.class, int.class, int.class);
            return constructor.newInstance(display, shell, x, y);
        } catch (Exception e) {
            System.err.println("Failed to create node: " + type + " - " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Check if a node type is registered.
     */
    public static boolean isRegistered(String type) {
        return nodesByName.containsKey(type);
    }
}
