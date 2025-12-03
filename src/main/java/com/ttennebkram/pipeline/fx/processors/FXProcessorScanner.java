package com.ttennebkram.pipeline.fx.processors;

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
 * Scans the classpath at runtime to discover all FXProcessor classes
 * annotated with @FXProcessorInfo.
 * Works from both filesystem (IDE/development) and JAR (production).
 */
public class FXProcessorScanner {

    private static final String PROCESSORS_PACKAGE = "com.ttennebkram.pipeline.fx.processors";

    /**
     * Find all concrete processor classes annotated with @FXProcessorInfo.
     *
     * @return set of discovered processor classes
     */
    public static Set<Class<? extends FXProcessor>> findProcessorClasses() {
        Set<Class<? extends FXProcessor>> processorClasses = new LinkedHashSet<>();
        String path = PROCESSORS_PACKAGE.replace('.', '/');
        ClassLoader classLoader = Thread.currentThread().getContextClassLoader();

        try {
            Enumeration<URL> resources = classLoader.getResources(path);
            while (resources.hasMoreElements()) {
                URL resource = resources.nextElement();
                String protocol = resource.getProtocol();

                if ("file".equals(protocol)) {
                    // Running from filesystem (IDE/development)
                    scanDirectory(new File(resource.toURI()), PROCESSORS_PACKAGE, processorClasses);
                } else if ("jar".equals(protocol)) {
                    // Running from JAR (production)
                    scanJar(resource, path, processorClasses);
                }
            }
        } catch (Exception e) {
            System.err.println("Error scanning for FXProcessor classes: " + e.getMessage());
            e.printStackTrace();
        }

        return processorClasses;
    }

    /**
     * Scan a directory for processor classes.
     */
    private static void scanDirectory(File directory, String packageName,
                                       Set<Class<? extends FXProcessor>> result) {
        if (!directory.exists()) return;

        File[] files = directory.listFiles();
        if (files == null) return;

        for (File file : files) {
            if (file.isDirectory()) {
                // Recurse into subdirectories
                scanDirectory(file, packageName + "." + file.getName(), result);
            } else if (file.getName().endsWith("Processor.class")) {
                // Only check files ending with "Processor.class" for efficiency
                String className = packageName + "." + file.getName().replace(".class", "");
                tryLoadProcessorClass(className, result);
            }
        }
    }

    /**
     * Scan a JAR file for processor classes.
     */
    private static void scanJar(URL jarUrl, String packagePath,
                                 Set<Class<? extends FXProcessor>> result) {
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

                    // Check if entry is in our package and ends with Processor.class
                    if (name.startsWith(packagePath) && name.endsWith("Processor.class")) {
                        String className = name.replace('/', '.').replace(".class", "");
                        tryLoadProcessorClass(className, result);
                    }
                }
            }
        } catch (Exception e) {
            System.err.println("Error scanning JAR for processors: " + e.getMessage());
        }
    }

    /**
     * Try to load a class and add it to the result if it's a valid processor class.
     */
    private static void tryLoadProcessorClass(String className,
                                               Set<Class<? extends FXProcessor>> result) {
        try {
            Class<?> clazz = Class.forName(className);

            // Must implement FXProcessor
            if (!FXProcessor.class.isAssignableFrom(clazz)) return;

            // Must not be abstract or interface
            if (Modifier.isAbstract(clazz.getModifiers())) return;
            if (clazz.isInterface()) return;

            // Must have @FXProcessorInfo annotation
            if (!clazz.isAnnotationPresent(FXProcessorInfo.class)) return;

            @SuppressWarnings("unchecked")
            Class<? extends FXProcessor> processorClass = (Class<? extends FXProcessor>) clazz;
            result.add(processorClass);

        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            // Skip classes that can't be loaded (missing dependencies, etc.)
        }
    }
}
