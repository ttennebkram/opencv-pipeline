package com.ttennebkram.pipeline.registry;

import com.ttennebkram.pipeline.nodes.PipelineNode;

import java.io.File;
import java.lang.reflect.Modifier;
import java.net.URI;
import java.net.URL;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;

/**
 * Scans the classpath at runtime to discover all node classes annotated with @NodeInfo.
 * Works from both filesystem (IDE/development) and JAR (production).
 * Uses only built-in Java libraries - no external dependencies.
 */
public class NodeScanner {

    private static final String NODES_PACKAGE = "com.ttennebkram.pipeline.nodes";

    /**
     * Find all concrete node classes annotated with @NodeInfo.
     *
     * @return set of discovered node classes
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

    /**
     * Scan a directory for node classes.
     */
    private static void scanDirectory(File directory, String packageName,
                                       Set<Class<? extends PipelineNode>> result) {
        if (!directory.exists()) return;

        File[] files = directory.listFiles();
        if (files == null) return;

        for (File file : files) {
            if (file.isDirectory()) {
                // Recurse into subdirectories
                scanDirectory(file, packageName + "." + file.getName(), result);
            } else if (file.getName().endsWith("Node.class")) {
                // Only check files ending with "Node.class" for efficiency
                String className = packageName + "." + file.getName().replace(".class", "");
                tryLoadNodeClass(className, result);
            }
        }
    }

    /**
     * Scan a JAR file for node classes.
     */
    private static void scanJar(URL jarUrl, String packagePath,
                                 Set<Class<? extends PipelineNode>> result) {
        try {
            // Extract JAR path from URL like "jar:file:/path/to.jar!/com/..."
            String urlPath = jarUrl.getPath();
            int bangIndex = urlPath.indexOf('!');
            if (bangIndex < 0) return;

            String jarPath = urlPath.substring(0, bangIndex);
            // Handle "file:" prefix
            if (jarPath.startsWith("file:")) {
                jarPath = new URI(jarPath).getPath();
            }

            try (JarFile jarFile = new JarFile(jarPath)) {
                Enumeration<JarEntry> entries = jarFile.entries();
                while (entries.hasMoreElements()) {
                    JarEntry entry = entries.nextElement();
                    String name = entry.getName();

                    // Check if entry is in our package and ends with Node.class
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

    /**
     * Try to load a class and add it to the result if it's a valid node class.
     */
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
            // Skip classes that can't be loaded (missing dependencies, etc.)
        }
    }
}
