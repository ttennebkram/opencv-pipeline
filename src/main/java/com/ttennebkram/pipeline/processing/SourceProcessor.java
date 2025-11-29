package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;

import java.util.concurrent.BlockingQueue;

/**
 * Processor wrapper for source nodes (FileSource, WebcamSource, BlankSource).
 *
 * Sources don't have an ImageProcessor (they generate frames instead of processing them),
 * but they need to participate in the backpressure signaling chain. This class provides:
 * - A ThreadedProcessor that downstream nodes can reference as inputNode
 * - FPS-based slowdown (halving FPS when at min priority)
 * - Recovery logic matching SourceNode in the main branch
 *
 * The actual frame generation happens in FXPipelineExecutor.startSourceFeeder(),
 * which reads the effective FPS from this processor.
 */
public class SourceProcessor extends ThreadedProcessor {

    // FPS slowdown tracking (matches main branch SourceNode)
    private double originalFps = 1.0;
    private int fpsSlowdownLevel = 0;
    private long frameDelayMs = 1000;

    /** Max times to halve FPS (7 levels = 128x reduction) */
    private static final int MAX_FPS_SLOWDOWN_LEVELS = 7;
    /** Recovery interval for FPS - wait 60 seconds before raising FPS back up */
    private static final long FPS_RECOVERY_MS = 60000;

    public SourceProcessor(String name) {
        // Pass a no-op processor - source doesn't process, it generates
        super(name, input -> null);
        setIsSource(true);
    }

    /**
     * Set the original FPS for this source.
     */
    @Override
    public void setOriginalFps(double fps) {
        this.originalFps = fps;
        this.frameDelayMs = fps > 0 ? (long) (1000.0 / fps) : 0;
    }

    /**
     * Get the original configured FPS.
     */
    public double getOriginalFps() {
        return originalFps;
    }

    /**
     * Get the current effective FPS after slowdown adjustments.
     */
    @Override
    public double getEffectiveFps() {
        if (fpsSlowdownLevel <= 0) {
            return originalFps;
        }
        return originalFps / Math.pow(2, fpsSlowdownLevel);
    }

    /**
     * Get the current frame delay in milliseconds.
     */
    public long getFrameDelayMs() {
        return frameDelayMs;
    }

    /**
     * Get the current FPS slowdown level.
     */
    public int getFpsSlowdownLevel() {
        return fpsSlowdownLevel;
    }

    /**
     * Override receiveSlowdownSignal to implement FPS-based slowdown for sources.
     * First reduces thread priority, then halves FPS when at MIN_PRIORITY.
     */
    @Override
    public synchronized void receiveSlowdownSignal() {
        long now = System.currentTimeMillis();
        setLastSlowdownReceivedTime(now);
        setInSlowdownMode(true);

        int currentPriority = getThreadPriority();

        if (currentPriority > Thread.MIN_PRIORITY) {
            // First, reduce priority like other nodes
            int newPriority = currentPriority - 1;
            System.out.println("[SourceProcessor] " + getName() + " RECEIVED SLOWDOWN, " +
                "lowering priority: " + currentPriority + " -> " + newPriority);
            setThreadPriority(newPriority);
            incrementSlowdownPriorityReduction();
        } else if (fpsSlowdownLevel < MAX_FPS_SLOWDOWN_LEVELS) {
            // Already at minimum priority, reduce FPS instead
            fpsSlowdownLevel++;
            double effectiveFps = getEffectiveFps();
            frameDelayMs = effectiveFps > 0 ? (long) (1000.0 / effectiveFps) : 0;
            System.out.println("[SourceProcessor] " + getName() + " RECEIVED SLOWDOWN at min priority, " +
                "reducing FPS: " + String.format("%.3f", originalFps) + " -> " + String.format("%.3f", effectiveFps) +
                " (level " + fpsSlowdownLevel + "/" + MAX_FPS_SLOWDOWN_LEVELS + ")");

            // Update FXNode stats for UI display
            FXNode node = getFXNode();
            if (node != null) {
                // threadPriority field shows effective FPS level (10 = full, lower = reduced)
                node.threadPriority = (int) Math.round((effectiveFps / originalFps) * 10);
            }
        } else {
            System.out.println("[SourceProcessor] " + getName() + " RECEIVED SLOWDOWN but already at max slowdown");
        }
    }

    /**
     * Override checkSlowdownRecovery for source-specific recovery.
     * Recovers FPS first (with longer interval), then priority.
     */
    @Override
    protected synchronized void checkSlowdownRecovery() {
        if (!isInSlowdownMode()) {
            return;
        }

        long now = System.currentTimeMillis();
        long timeSinceSlowdown = now - getLastSlowdownReceivedTime();

        // First recover FPS if reduced (uses longer FPS_RECOVERY_MS interval)
        if (fpsSlowdownLevel > 0) {
            if (timeSinceSlowdown >= FPS_RECOVERY_MS) {
                double oldFps = getEffectiveFps();
                fpsSlowdownLevel--;
                double newFps = getEffectiveFps();
                frameDelayMs = newFps > 0 ? (long) (1000.0 / newFps) : 0;
                System.out.println("[SourceProcessor] " + getName() + " SLOWDOWN RECOVERY, " +
                    "raising FPS: " + String.format("%.3f", oldFps) + " -> " + String.format("%.3f", newFps) +
                    " (level " + fpsSlowdownLevel + "/" + MAX_FPS_SLOWDOWN_LEVELS + ")");
                setLastSlowdownReceivedTime(now); // Reset timer for next recovery step

                // Update FXNode stats for UI display
                FXNode node = getFXNode();
                if (node != null) {
                    node.threadPriority = (int) Math.round((newFps / originalFps) * 10);
                }
            }
            return; // Don't recover priority until FPS is fully restored
        }

        // Then use parent's priority recovery
        super.checkSlowdownRecovery();
    }

    /**
     * Override processingLoop - sources don't process from queues,
     * they generate frames in a separate feeder thread.
     * This processor just needs to exist for backpressure signaling.
     */
    @Override
    protected void processingLoop() {
        // Source processors don't have their own processing loop.
        // The actual frame generation happens in FXPipelineExecutor.startSourceFeeder().
        // This method exists so the processor can be started (for stats tracking)
        // but doesn't need to do anything.
        System.out.println("[SourceProcessor] " + getName() + " processingLoop started (stats/recovery thread)");

        while (isRunning()) {
            try {
                // Check recovery periodically
                checkSlowdownRecovery();

                // Update stats for UI
                FXNode node = getFXNode();
                if (node != null) {
                    node.workUnitsCompleted = getWorkUnitsCompleted();
                    // Keep threadPriority updated based on effective FPS
                    if (originalFps > 0) {
                        node.threadPriority = (int) Math.round((getEffectiveFps() / originalFps) * 10);
                    }
                }

                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }

        System.out.println("[SourceProcessor] " + getName() + " processingLoop exiting");
    }

    /**
     * Increment work units (called by source feeder when frame is generated).
     */
    public void incrementWorkUnits() {
        incrementWorkUnitsCompleted();
    }
}
