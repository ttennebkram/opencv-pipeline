package com.ttennebkram.pipeline.fx;

import com.ttennebkram.pipeline.fx.processors.FXProcessorInfo;
import com.ttennebkram.pipeline.fx.processors.FXProcessorScanner;
import com.ttennebkram.pipeline.fx.processors.FXProcessor;

import java.util.*;

/**
 * Registry of available node types for the JavaFX editor.
 * Auto-discovers processor types via @FXProcessorInfo annotations at runtime.
 * Also includes non-processor node types (Container, ContainerInput, ContainerOutput).
 */
public class FXNodeRegistry {

    public static class NodeType {
        public final String name;
        public final String displayName;
        public final String buttonName;  // Shorter name for toolbar buttons (null = use displayName)
        public final String category;
        public final String description;  // Method signature / description
        public final boolean isSource;
        public final boolean isDualInput;
        public final boolean isContainer;
        public final int outputCount;
        public final boolean canBeDisabled;  // Whether this node can be disabled (show checkbox)

        public NodeType(String name, String displayName, String buttonName, String category, String description,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount, boolean canBeDisabled) {
            this.name = name;
            this.displayName = displayName;
            this.buttonName = buttonName;
            this.category = category;
            this.description = description;
            this.isSource = isSource;
            this.isDualInput = isDualInput;
            this.isContainer = isContainer;
            this.outputCount = outputCount;
            this.canBeDisabled = canBeDisabled;
        }

        public NodeType(String name, String displayName, String buttonName, String category, String description,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
            this(name, displayName, buttonName, category, description, isSource, isDualInput, isContainer, outputCount, true);
        }

        public NodeType(String name, String displayName, String category,
                        boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
            this(name, displayName, null, category, null, isSource, isDualInput, isContainer, outputCount, true);
        }

        public NodeType(String name, String displayName, String category) {
            this(name, displayName, null, category, null, false, false, false, 1, true);
        }

        /**
         * Get the name to display on toolbar buttons.
         */
        public String getButtonName() {
            return buttonName != null ? buttonName : displayName;
        }
    }

    private static final List<NodeType> nodeTypes = new ArrayList<>();
    private static final Map<String, List<NodeType>> byCategory = new LinkedHashMap<>();
    private static boolean initialized = false;

    // Define the preferred order of categories in the button bar
    private static final List<String> CATEGORY_ORDER = Arrays.asList(
        "Sources",
        "Basic",
        "Blur",
        "Content",
        "Edges",
        "Filter",
        "Morphology",
        "Transform",
        "Visualization",
        "Detection",
        "Dual Input",
        "Nested Pipelines",
        "Utility"
    );

    /**
     * Initialize the registry. Safe to call multiple times.
     */
    public static synchronized void initialize() {
        if (initialized) return;

        // First, auto-discover all processors via @FXProcessorInfo annotations
        Set<Class<? extends FXProcessor>> processorClasses = FXProcessorScanner.findProcessorClasses();
        for (Class<? extends FXProcessor> processorClass : processorClasses) {
            FXProcessorInfo info = processorClass.getAnnotation(FXProcessorInfo.class);
            if (info == null) continue;

            String name = info.nodeType();
            String displayName = info.displayName().isEmpty() ? name : info.displayName();
            String buttonName = info.buttonName().isEmpty() ? null : info.buttonName();
            String category = info.category();
            String description = info.description();
            boolean isSource = info.isSource();
            boolean isDualInput = info.dualInput();
            boolean isContainer = info.isContainer();
            int outputCount = info.outputCount();
            boolean canBeDisabled = info.canBeDisabled();

            NodeType type = new NodeType(name, displayName, buttonName, category, description,
                    isSource, isDualInput, isContainer, outputCount, canBeDisabled);
            nodeTypes.add(type);
            byCategory.computeIfAbsent(category, k -> new ArrayList<>()).add(type);
        }

        // Add non-processor node types (special nodes without processors)
        // Container (sub-pipeline) - special handling, not a processor
        register("Container", "Container/Sub-pipeline", "Nested Pipelines", "Container sub-pipeline\nEncapsulates a pipeline", false, false, true, 1);

        // Container I/O nodes - boundary nodes for containers
        register("ContainerInput", "Nested Pipeline Input", "Nested Pipelines", "Nested pipeline input\nReceives data from parent pipeline", false, false, 1);
        register("ContainerOutput", "Nested Pipeline Output", "Nested Pipelines", "Nested pipeline output\nSends data to parent pipeline");

        // Sort nodes within each category alphabetically by display name
        for (List<NodeType> nodes : byCategory.values()) {
            nodes.sort((a, b) -> a.displayName.compareToIgnoreCase(b.displayName));
        }

        initialized = true;
        System.out.println("[FXNodeRegistry] Initialized with " + nodeTypes.size() + " node types from " + byCategory.size() + " categories");
    }

    // Ensure initialized before any access
    private static void ensureInitialized() {
        if (!initialized) {
            initialize();
        }
    }

    private static void register(String name, String displayName, String category) {
        register(name, displayName, null, category, null, false, false, false, 1);
    }

    private static void register(String name, String displayName, String category, String description) {
        register(name, displayName, null, category, description, false, false, false, 1);
    }

    // Note: For buttonName with no description, use the 5-arg version with null description
    private static void register(String name, String displayName, String buttonName, String category, String description) {
        register(name, displayName, buttonName, category, description, false, false, false, 1);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, int outputCount) {
        register(name, displayName, null, category, null, isSource, isDualInput, false, outputCount);
    }

    private static void register(String name, String displayName, String category, String description,
                                  boolean isSource, boolean isDualInput, int outputCount) {
        register(name, displayName, null, category, description, isSource, isDualInput, false, outputCount);
    }

    private static void register(String name, String displayName, String category,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        register(name, displayName, null, category, null, isSource, isDualInput, isContainer, outputCount);
    }

    private static void register(String name, String displayName, String category, String description,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        register(name, displayName, null, category, description, isSource, isDualInput, isContainer, outputCount);
    }

    private static void register(String name, String displayName, String buttonName, String category, String description,
                                  boolean isSource, boolean isDualInput, boolean isContainer, int outputCount) {
        NodeType type = new NodeType(name, displayName, buttonName, category, description, isSource, isDualInput, isContainer, outputCount);
        nodeTypes.add(type);
        byCategory.computeIfAbsent(category, k -> new ArrayList<>()).add(type);
    }

    /**
     * Get all categories in display order.
     * Categories are sorted according to CATEGORY_ORDER, with any unknown categories at the end.
     */
    public static List<String> getCategories() {
        ensureInitialized();
        List<String> result = new ArrayList<>();

        // First add categories in the defined order
        for (String cat : CATEGORY_ORDER) {
            if (byCategory.containsKey(cat)) {
                result.add(cat);
            }
        }

        // Then add any remaining categories not in the defined order (alphabetically)
        List<String> remaining = new ArrayList<>();
        for (String cat : byCategory.keySet()) {
            if (!CATEGORY_ORDER.contains(cat)) {
                remaining.add(cat);
            }
        }
        Collections.sort(remaining);
        result.addAll(remaining);

        return result;
    }

    /**
     * Get categories excluding certain ones (e.g., exclude "Container I/O" from main editor).
     * Categories are sorted according to CATEGORY_ORDER.
     */
    public static List<String> getCategoriesExcluding(String... excludeCategories) {
        ensureInitialized();
        Set<String> excluded = new HashSet<>(Arrays.asList(excludeCategories));
        List<String> allCategories = getCategories();
        List<String> result = new ArrayList<>();
        for (String cat : allCategories) {
            if (!excluded.contains(cat)) {
                result.add(cat);
            }
        }
        return result;
    }

    /**
     * Get all node types in a category.
     */
    public static List<NodeType> getNodesInCategory(String category) {
        ensureInitialized();
        return byCategory.getOrDefault(category, Collections.emptyList());
    }

    /**
     * Get a node type by name.
     */
    public static NodeType getNodeType(String name) {
        ensureInitialized();
        for (NodeType type : nodeTypes) {
            if (type.name.equals(name) || type.displayName.equals(name)) {
                return type;
            }
        }
        return null;
    }

    /**
     * Get all node types.
     */
    public static List<NodeType> getAllNodeTypes() {
        ensureInitialized();
        return Collections.unmodifiableList(nodeTypes);
    }
}
