package com.ttennebkram.pipeline.util;

import org.opencv.core.Mat;

import java.io.PrintStream;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Utility for tracking OpenCV Mat allocations to help debug memory leaks.
 *
 * Usage:
 * 1. Enable tracking: MatTracker.setEnabled(true);
 * 2. Use MatTracker.track(mat) after creating Mats
 * 3. Use MatTracker.release(mat) instead of mat.release()
 * 4. Press Ctrl+Shift+M or call MatTracker.dumpLeaks() to see unreleased Mats
 *
 * When enabled, each tracked Mat stores the stack trace of where it was created,
 * making it easy to find the source of leaks.
 *
 * The tracker is automatically enabled at startup. Use Ctrl+M to toggle.
 */
public class MatTracker {

    // Default to enabled for debugging - toggle with Ctrl+M
    private static volatile boolean enabled = true;
    private static volatile boolean captureStackTraces = true;  // Can disable for less overhead
    private static final Map<Long, MatInfo> activeMats = new ConcurrentHashMap<>();
    private static final AtomicLong totalCreated = new AtomicLong(0);
    private static final AtomicLong totalReleased = new AtomicLong(0);

    // Install global exception handler on class load
    static {
        installExceptionHandler();
    }

    /**
     * Install global exception handler to catch "Too many open files" errors.
     */
    private static void installExceptionHandler() {
        Thread.UncaughtExceptionHandler existingHandler = Thread.getDefaultUncaughtExceptionHandler();
        Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
            // Check if this is a "too many open files" error
            if (isTooManyFilesError(throwable)) {
                System.err.println("\n!!! TOO MANY OPEN FILES ERROR DETECTED !!!");
                System.err.println("Dumping Mat tracking information...\n");
                printSummary(System.err);
                dumpLeaksByLocation(System.err);
                System.err.println("\nFull leak details:");
                dumpLeaks(System.err);
                // Print the original exception stack trace after the Mat info
                System.err.println("\n=== ORIGINAL EXCEPTION STACK TRACE ===");
                System.err.println("Exception in thread \"" + thread.getName() + "\"");
                throwable.printStackTrace(System.err);
                System.err.println("=== END STACK TRACE ===\n");
            }
            // Call existing handler if present (for non-file errors, or additional handling)
            if (existingHandler != null) {
                existingHandler.uncaughtException(thread, throwable);
            } else if (!isTooManyFilesError(throwable)) {
                // Default behavior for non-file errors (file errors already printed above)
                System.err.println("Uncaught exception in thread " + thread.getName());
                throwable.printStackTrace();
            }
        });
    }

    /**
     * Check if an exception is related to "too many open files".
     */
    private static boolean isTooManyFilesError(Throwable t) {
        while (t != null) {
            String msg = t.getMessage();
            if (msg != null && (msg.contains("Too many open files") ||
                               msg.contains("too many open files") ||
                               msg.contains("EMFILE") ||
                               msg.contains("ENFILE"))) {
                return true;
            }
            t = t.getCause();
        }
        return false;
    }

    /**
     * Info about a tracked Mat.
     */
    private static class MatInfo {
        final long id;
        final long nativeAddr;
        final String creationStack;
        final long creationTime;
        final String shortLocation;

        MatInfo(long id, long nativeAddr) {
            this.id = id;
            this.nativeAddr = nativeAddr;
            this.creationTime = System.currentTimeMillis();

            if (!captureStackTraces) {
                this.creationStack = "(stack traces disabled)";
                this.shortLocation = "unknown";
                return;
            }

            // Capture stack trace, skipping MatTracker frames
            StackTraceElement[] stack = Thread.currentThread().getStackTrace();
            StringBuilder sb = new StringBuilder();
            String shortLoc = "unknown";
            boolean foundCaller = false;

            for (int i = 0; i < stack.length; i++) {
                String className = stack[i].getClassName();
                // Skip Thread.getStackTrace and MatTracker methods
                if (className.equals("java.lang.Thread") ||
                    className.equals("com.ttennebkram.pipeline.util.MatTracker")) {
                    continue;
                }

                String line = String.format("  at %s.%s(%s:%d)",
                    className,
                    stack[i].getMethodName(),
                    stack[i].getFileName(),
                    stack[i].getLineNumber());
                sb.append(line).append("\n");

                // First non-MatTracker frame is the caller
                if (!foundCaller) {
                    shortLoc = String.format("%s:%d",
                        stack[i].getFileName(),
                        stack[i].getLineNumber());
                    foundCaller = true;
                }

                // Limit stack depth
                if (sb.length() > 2000) {
                    sb.append("  ... (truncated)\n");
                    break;
                }
            }

            this.creationStack = sb.toString();
            this.shortLocation = shortLoc;
        }
    }

    /**
     * Enable or disable Mat tracking.
     * When disabled, create() returns new Mat() directly with no tracking overhead.
     */
    public static void setEnabled(boolean enable) {
        enabled = enable;
        if (enable) {
            System.out.println("[MatTracker] Enabled - tracking Mat allocations");
        } else {
            System.out.println("[MatTracker] Disabled");
        }
    }

    /**
     * Check if tracking is enabled.
     */
    public static boolean isEnabled() {
        return enabled;
    }

    /**
     * Create a new Mat and track it.
     * Use this instead of new Mat().
     */
    public static Mat create() {
        Mat mat = new Mat();
        if (enabled) {
            track(mat);
        }
        return mat;
    }

    /**
     * Track an existing Mat.
     * Call this after creating a Mat with new Mat(...) or clone().
     */
    public static void track(Mat mat) {
        if (!enabled || mat == null) return;

        long addr = mat.getNativeObjAddr();
        long id = totalCreated.incrementAndGet();
        activeMats.put(addr, new MatInfo(id, addr));
    }

    /**
     * Release a Mat and remove it from tracking.
     * Use this instead of mat.release().
     */
    public static void release(Mat mat) {
        if (mat == null) return;

        if (enabled) {
            long addr = mat.getNativeObjAddr();
            if (activeMats.remove(addr) != null) {
                totalReleased.incrementAndGet();
            }
        }
        mat.release();
    }

    /**
     * Get count of currently active (unreleased) tracked Mats.
     */
    public static int getActiveCount() {
        return activeMats.size();
    }

    /**
     * Get total Mats created since tracking started.
     */
    public static long getTotalCreated() {
        return totalCreated.get();
    }

    /**
     * Get total Mats released since tracking started.
     */
    public static long getTotalReleased() {
        return totalReleased.get();
    }

    /**
     * Print summary of Mat allocations.
     */
    public static void printSummary() {
        printSummary(System.out);
    }

    /**
     * Print summary of Mat allocations to specified stream.
     */
    public static void printSummary(PrintStream out) {
        out.printf("[MatTracker] Created: %d, Released: %d, Active: %d%n",
            totalCreated.get(), totalReleased.get(), activeMats.size());
    }

    /**
     * Dump all active (potentially leaked) Mats with their creation locations.
     */
    public static void dumpLeaks() {
        dumpLeaks(System.out);
    }

    /**
     * Dump all active (potentially leaked) Mats to specified stream.
     */
    public static void dumpLeaks(PrintStream out) {
        if (activeMats.isEmpty()) {
            out.println("[MatTracker] No active Mats (no leaks detected)");
            return;
        }

        out.printf("[MatTracker] === %d ACTIVE MATS (potential leaks) ===%n", activeMats.size());

        long now = System.currentTimeMillis();
        int count = 0;

        for (MatInfo info : activeMats.values()) {
            count++;
            long ageMs = now - info.creationTime;
            out.printf("%n--- Mat #%d (age: %dms, addr: 0x%x) ---%n",
                info.id, ageMs, info.nativeAddr);
            out.printf("Created at: %s%n", info.shortLocation);
            out.println("Stack trace:");
            out.print(info.creationStack);

            // Limit output for very large leaks
            if (count >= 50) {
                out.printf("%n... and %d more (showing first 50)%n",
                    activeMats.size() - 50);
                break;
            }
        }

        out.println("\n[MatTracker] === END LEAK DUMP ===");
    }

    /**
     * Dump summary by creation location (grouped).
     * Useful for finding patterns in leaks.
     */
    public static void dumpLeaksByLocation() {
        dumpLeaksByLocation(System.out);
    }

    /**
     * Dump summary by creation location to specified stream.
     */
    public static void dumpLeaksByLocation(PrintStream out) {
        if (activeMats.isEmpty()) {
            out.println("[MatTracker] No active Mats (no leaks detected)");
            return;
        }

        // Group by short location
        Map<String, AtomicLong> locationCounts = new ConcurrentHashMap<>();
        for (MatInfo info : activeMats.values()) {
            locationCounts.computeIfAbsent(info.shortLocation, k -> new AtomicLong(0))
                .incrementAndGet();
        }

        out.printf("[MatTracker] === LEAKS BY LOCATION (%d total) ===%n", activeMats.size());

        // Sort by count descending
        locationCounts.entrySet().stream()
            .sorted((a, b) -> Long.compare(b.getValue().get(), a.getValue().get()))
            .forEach(e -> out.printf("  %5d : %s%n", e.getValue().get(), e.getKey()));

        out.println("[MatTracker] === END ===");
    }

    /**
     * Clear all tracking data and reset counters.
     */
    public static void reset() {
        activeMats.clear();
        totalCreated.set(0);
        totalReleased.set(0);
        System.out.println("[MatTracker] Reset - all tracking data cleared");
    }
}
