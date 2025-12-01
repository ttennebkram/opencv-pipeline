package com.ttennebkram.pipeline.fx.processors;

import com.ttennebkram.pipeline.fx.FXNode;

import java.util.*;
import java.util.function.Supplier;

/**
 * Registry for FXProcessor implementations.
 * Auto-discovers processors at runtime via classpath scanning.
 * Processors must be annotated with @FXProcessorInfo to be discovered.
 *
 * Usage:
 *   FXProcessor processor = FXProcessorRegistry.createProcessor("GaussianBlur");
 *   processor.syncFromFXNode(fxNode);  // Load properties from FXNode
 *   Mat output = processor.process(input);
 */
public class FXProcessorRegistry {

    // Map from node type to processor class
    private static final Map<String, Class<? extends FXProcessor>> processorClasses = new HashMap<>();

    // Map from node type to whether it's dual-input
    private static final Map<String, Boolean> dualInputTypes = new HashMap<>();

    // Initialization flag
    private static boolean initialized = false;

    /**
     * Initialize the registry by scanning for processor classes.
     * Safe to call multiple times - only initializes once.
     */
    public static synchronized void initialize() {
        if (initialized) return;

        Set<Class<? extends FXProcessor>> classes = FXProcessorScanner.findProcessorClasses();

        for (Class<? extends FXProcessor> processorClass : classes) {
            FXProcessorInfo info = processorClass.getAnnotation(FXProcessorInfo.class);
            if (info == null) continue;

            String nodeType = info.nodeType();
            boolean isDualInput = info.dualInput();

            processorClasses.put(nodeType, processorClass);
            dualInputTypes.put(nodeType, isDualInput);
        }

        initialized = true;
    }

    /**
     * Check if a processor exists for the given node type.
     */
    public static boolean hasProcessor(String nodeType) {
        initialize();
        return processorClasses.containsKey(nodeType);
    }

    /**
     * Check if a node type is a dual-input processor.
     */
    public static boolean isDualInput(String nodeType) {
        initialize();
        return dualInputTypes.getOrDefault(nodeType, false);
    }

    /**
     * Create a new processor instance for the given node type.
     * Returns null if no processor is registered for this type.
     */
    public static FXProcessor createProcessor(String nodeType) {
        initialize();
        Class<? extends FXProcessor> processorClass = processorClasses.get(nodeType);
        if (processorClass != null) {
            try {
                return processorClass.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                System.err.println("[FXProcessorRegistry] Failed to create processor for " + nodeType + ": " + e.getMessage());
                e.printStackTrace();
            }
        }
        return null;
    }

    /**
     * Create a processor and initialize it from an FXNode.
     * Returns null if no processor is registered for this type.
     */
    public static FXProcessor createProcessor(FXNode fxNode) {
        FXProcessor processor = createProcessor(fxNode.nodeType);
        if (processor != null) {
            processor.syncFromFXNode(fxNode);
        }
        return processor;
    }

    /**
     * Get all registered node types.
     */
    public static Set<String> getRegisteredTypes() {
        initialize();
        return Collections.unmodifiableSet(processorClasses.keySet());
    }

    /**
     * Get the count of registered processors.
     */
    public static int getRegisteredCount() {
        initialize();
        return processorClasses.size();
    }
}
