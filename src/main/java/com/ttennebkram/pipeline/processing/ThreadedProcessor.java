package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
import org.opencv.core.Mat;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;

/**
 * Base class for threaded image processors.
 * Runs in its own thread, consuming from input queue and producing to output queue.
 * Implements backpressure system with cascading slowdown signals.
 * Each processor updates its associated FXNode's stats directly.
 */
public class ThreadedProcessor {

    private final String name;
    private final ImageProcessor processor;

    // Reference to associated FXNode for direct stats updates
    private FXNode fxNode;

    private Thread processingThread;
    private final AtomicBoolean running = new AtomicBoolean(false);

    private BlockingQueue<Mat> inputQueue;
    private BlockingQueue<Mat> outputQueue;
    private BlockingQueue<Mat> inputQueue2; // For dual-input processors
    private BlockingQueue<Mat> outputQueue2; // For dual-output processors (e.g., Clone)

    private int threadPriority = Thread.NORM_PRIORITY;
    private int originalPriority = Thread.NORM_PRIORITY;
    private int lastRunningPriority = Thread.NORM_PRIORITY;
    private boolean enabled = true;
    private boolean isDualInput = false;
    private boolean isSource = false;

    // Statistics
    private long inputReads1 = 0;
    private long inputReads2 = 0;
    private long workUnitsCompleted = 0;
    private long outputWrites1 = 0;
    private long outputWrites2 = 0;

    // Backpressure timing constants (from main branch)
    /** Cooldown between priority reductions (going down) - 1 second per step */
    private static final long PRIORITY_LOWER_COOLDOWN_MS = 1000;
    /** Cooldown between priority increases (going up) - 10 seconds per step */
    private static final long PRIORITY_RAISE_COOLDOWN_MS = 10000;
    /** How long after last slowdown signal before starting recovery */
    private static final long SLOWDOWN_RECOVERY_MS = 10000;
    private long lastPriorityAdjustmentTime = 0;

    // Slowdown signaling (cascading backpressure)
    private volatile long lastSlowdownReceivedTime = 0;
    private volatile boolean inSlowdownMode = false;
    private int slowdownPriorityReduction = 0;

    // Source node FPS slowdown
    private double originalFps = 0;
    private int fpsSlowdownLevel = 0;
    /** Max times to halve FPS (7 levels = 128x reduction, e.g. 10fps -> 0.08fps) */
    private static final int MAX_FPS_SLOWDOWN_LEVELS = 7;
    /** Recovery interval for FPS - wait 1 minute before raising FPS back up */
    private static final long FPS_RECOVERY_MS = 60000;
    private volatile long frameDelayMs = 0;

    // Upstream node references for signaling slowdown
    private ThreadedProcessor inputNode;
    private ThreadedProcessor inputNode2;

    // Parent container reference for boundary nodes (ContainerInput/ContainerOutput)
    // When a ContainerOutput receives backpressure/slowdown, it forwards to the container
    private ContainerProcessor parentContainer;

    // Callbacks
    private Consumer<Mat> onFrameCallback;

    // Timestamp formatter for logging
    private static final SimpleDateFormat LOG_TIME_FORMAT = new SimpleDateFormat("HH:mm:ss.SSS");

    public ThreadedProcessor(String name, ImageProcessor processor) {
        this.name = name;
        this.processor = processor;
    }

    public String getName() {
        return name;
    }

    /**
     * Set the associated FXNode for direct stats updates.
     * Each processor updates its FXNode's stats directly in its own thread.
     */
    public void setFXNode(FXNode fxNode) {
        this.fxNode = fxNode;
    }

    public FXNode getFXNode() {
        return fxNode;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public boolean isEnabled() {
        return enabled;
    }

    public void setDualInput(boolean dualInput) {
        this.isDualInput = dualInput;
    }

    public void setIsSource(boolean isSource) {
        this.isSource = isSource;
    }

    public boolean isSource() {
        return isSource;
    }

    public void setInputQueue(BlockingQueue<Mat> queue) {
        this.inputQueue = queue;
    }

    public void setInputQueue2(BlockingQueue<Mat> queue) {
        this.inputQueue2 = queue;
    }

    public void setOutputQueue(BlockingQueue<Mat> queue) {
        this.outputQueue = queue;
    }

    public void setOutputQueue2(BlockingQueue<Mat> queue) {
        this.outputQueue2 = queue;
    }

    public BlockingQueue<Mat> getOutputQueue2() {
        return outputQueue2;
    }

    public BlockingQueue<Mat> getInputQueue() {
        return inputQueue;
    }

    public BlockingQueue<Mat> getOutputQueue() {
        return outputQueue;
    }

    public void setInputNode(ThreadedProcessor node) {
        this.inputNode = node;
    }

    public void setInputNode2(ThreadedProcessor node) {
        this.inputNode2 = node;
    }

    /**
     * Set the parent container for boundary nodes (ContainerInput/ContainerOutput).
     * When a ContainerOutput detects backpressure or receives slowdown signal,
     * it forwards to the parent container.
     */
    public void setParentContainer(ContainerProcessor container) {
        this.parentContainer = container;
    }

    public ContainerProcessor getParentContainer() {
        return parentContainer;
    }

    public void setThreadPriority(int priority) {
        this.threadPriority = Math.max(Thread.MIN_PRIORITY, Math.min(Thread.MAX_PRIORITY, priority));
        this.originalPriority = this.threadPriority;
        this.lastRunningPriority = this.threadPriority;
        if (processingThread != null && processingThread.isAlive()) {
            processingThread.setPriority(threadPriority);
        }
    }

    public int getThreadPriority() {
        if (processingThread != null && processingThread.isAlive()) {
            int currentPriority = processingThread.getPriority();
            lastRunningPriority = currentPriority;
            return currentPriority;
        }
        return lastRunningPriority;
    }

    public int getOriginalPriority() {
        return originalPriority;
    }

    public long getInputReads1() {
        return inputReads1;
    }

    public long getInputReads2() {
        return inputReads2;
    }

    public long getWorkUnitsCompleted() {
        return workUnitsCompleted;
    }

    public long getOutputWrites1() {
        return outputWrites1;
    }

    public void resetStats() {
        inputReads1 = 0;
        inputReads2 = 0;
        workUnitsCompleted = 0;
        outputWrites1 = 0;
    }

    public void setOnFrameCallback(Consumer<Mat> callback) {
        this.onFrameCallback = callback;
    }

    /**
     * Set the original FPS for source nodes (before slowdown).
     */
    public void setOriginalFps(double fps) {
        this.originalFps = fps;
        this.frameDelayMs = fps > 0 ? (long) (1000.0 / fps) : 0;
    }

    /**
     * Get the effective FPS after slowdown adjustments.
     * Each slowdown level halves the FPS.
     */
    public double getEffectiveFps() {
        if (fpsSlowdownLevel <= 0) {
            return originalFps;
        }
        return originalFps / Math.pow(2, fpsSlowdownLevel);
    }

    // Protected accessors for subclasses (SourceProcessor)
    protected void setLastSlowdownReceivedTime(long time) {
        this.lastSlowdownReceivedTime = time;
    }

    protected long getLastSlowdownReceivedTime() {
        return lastSlowdownReceivedTime;
    }

    protected void setInSlowdownMode(boolean mode) {
        this.inSlowdownMode = mode;
    }

    protected boolean isInSlowdownMode() {
        return inSlowdownMode;
    }

    protected void incrementSlowdownPriorityReduction() {
        slowdownPriorityReduction++;
    }

    protected int getSlowdownPriorityReduction() {
        return slowdownPriorityReduction;
    }

    /**
     * Get a timestamp string for logging.
     */
    private static String timestamp() {
        return LOG_TIME_FORMAT.format(new Date());
    }

    /**
     * Format a number with commas for display.
     */
    public static String formatNumber(long number) {
        return String.format("%,d", number);
    }

    /**
     * Check output queue and apply progressive backpressure.
     * When this node's output queue backs up, progressively lower THIS node's priority.
     * - Queue size 5+: reduce by 1
     *
     * If already at minimum priority and queue still backed up, signal upstream to slow down.
     */
    protected void checkBackpressure() {
        // First, check if we should recover from slowdown mode
        checkSlowdownRecovery();

        // Calculate queue size from output queue
        int queueSize = outputQueue != null ? outputQueue.size() : 0;

        // Also check input queue if connected to a source (source has no natural blocking)
        // Input queue backing up means we're not keeping up with the source
        int inputQueueSize = 0;
        if (inputQueue != null && inputNode != null && inputNode.isSource()) {
            inputQueueSize = inputQueue.size();
        }

        // Use the max of output and input queue sizes for backpressure decision
        int effectiveQueueSize = Math.max(queueSize, inputQueueSize);

        // If no queues to monitor, nothing to do
        if (outputQueue == null && inputQueueSize == 0) {
            return;
        }

        if (processingThread != null && processingThread.isAlive()) {
            int currentPriority = processingThread.getPriority();
            long now = System.currentTimeMillis();

            // If in slowdown mode, don't raise priority - let checkSlowdownRecovery() handle it
            if (inSlowdownMode) {
                if (effectiveQueueSize >= 5) {
                    if (currentPriority > Thread.MIN_PRIORITY) {
                        // Lower by 1 if cooldown has passed
                        if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                            int newPriority = currentPriority - 1;
                            System.out.println("[" + timestamp() + "] [Processor " + name + "] LOWERING priority: " +
                                currentPriority + " -> " + newPriority + " (outQ=" + queueSize + ", inQ=" + inputQueueSize + ", inSlowdownMode)");
                            processingThread.setPriority(newPriority);
                            lastPriorityAdjustmentTime = now;
                            lastRunningPriority = newPriority;
                        }
                    } else {
                        // At min priority, signal upstream
                        if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                            signalUpstreamSlowdown();
                            lastPriorityAdjustmentTime = now;
                        }
                    }
                }
                return; // Don't raise priority - let slowdown recovery handle it
            }

            // Normal backpressure logic (not in slowdown mode)
            if (effectiveQueueSize >= 5) {
                // Queue backed up - lower priority by 1 if cooldown has passed
                if (currentPriority > Thread.MIN_PRIORITY) {
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                        int newPriority = currentPriority - 1;
                        System.out.println("[" + timestamp() + "] [Processor " + name + "] LOWERING priority: " +
                            currentPriority + " -> " + newPriority + " (outQ=" + queueSize + ", inQ=" + inputQueueSize + ")");
                        processingThread.setPriority(newPriority);
                        lastPriorityAdjustmentTime = now;
                        lastRunningPriority = newPriority;
                    }
                } else {
                    // At min priority and still backed up - signal upstream
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                        signalUpstreamSlowdown();
                        lastPriorityAdjustmentTime = now;
                    }
                }
            } else if (effectiveQueueSize == 0) {
                // Queue empty - raise priority by 1 toward original if cooldown has passed
                if (currentPriority < originalPriority) {
                    if (now - lastPriorityAdjustmentTime >= PRIORITY_RAISE_COOLDOWN_MS) {
                        int newPriority = currentPriority + 1;
                        System.out.println("[" + timestamp() + "] [Processor " + name + "] RAISING priority: " +
                            currentPriority + " -> " + newPriority + " (outQ=" + queueSize + ", inQ=" + inputQueueSize + ")");
                        processingThread.setPriority(newPriority);
                        lastPriorityAdjustmentTime = now;
                        lastRunningPriority = newPriority;
                    }
                }
            }
            // If queue is 1-4, do nothing - hold current priority
        }
    }

    /**
     * Signal upstream nodes to slow down because this node is overwhelmed.
     * For boundary nodes (ContainerOutput), also signals the parent container.
     */
    private void signalUpstreamSlowdown() {
        System.out.println("[" + timestamp() + "] [Processor " + name + "] SIGNALING SLOWDOWN to upstream nodes (inputNode=" +
                           (inputNode != null ? inputNode.getName() : "null") + ", parentContainer=" +
                           (parentContainer != null ? parentContainer.getName() : "null") + ")");
        if (inputNode != null) {
            inputNode.receiveSlowdownSignal();
        }
        if (inputNode2 != null) {
            inputNode2.receiveSlowdownSignal();
        }
        // For boundary nodes (ContainerOutput), signal the parent container
        if (parentContainer != null) {
            System.out.println("[" + timestamp() + "] [Processor " + name + "] SIGNALING SLOWDOWN to parent container: " + parentContainer.getName());
            parentContainer.receiveSlowdownSignal();
        }
    }

    /**
     * Receive a slowdown signal from a downstream node.
     * For processing nodes: reduces priority and enters slowdown mode.
     * For source nodes: reduces priority first, then FPS if at min priority.
     */
    public synchronized void receiveSlowdownSignal() {
        System.out.println("[" + timestamp() + "] [Processor " + name + "] receiveSlowdownSignal() called");
        long now = System.currentTimeMillis();
        lastSlowdownReceivedTime = now;
        inSlowdownMode = true;

        Thread pt = processingThread;
        if (pt != null && pt.isAlive()) {
            int currentPriority = pt.getPriority();

            if (currentPriority > Thread.MIN_PRIORITY) {
                // Reduce priority by 1
                int newPriority = currentPriority - 1;
                System.out.println("[" + timestamp() + "] [Processor " + name + "] RECEIVED SLOWDOWN, " +
                    "lowering priority: " + currentPriority + " -> " + newPriority);
                pt.setPriority(newPriority);
                lastRunningPriority = newPriority;
                slowdownPriorityReduction++;
            } else if (isSource && fpsSlowdownLevel < MAX_FPS_SLOWDOWN_LEVELS) {
                // Source node at min priority - reduce FPS instead
                fpsSlowdownLevel++;
                double effectiveFps = getEffectiveFps();
                frameDelayMs = effectiveFps > 0 ? (long) (1000.0 / effectiveFps) : 0;
                System.out.println("[" + timestamp() + "] [Processor " + name + "] RECEIVED SLOWDOWN at min priority, " +
                    "reducing FPS: " + String.format("%.3f", originalFps) + " -> " + String.format("%.3f", effectiveFps) +
                    " (level " + fpsSlowdownLevel + "/" + MAX_FPS_SLOWDOWN_LEVELS + ")");
            } else {
                // Already at minimum priority (and min FPS for sources), cascade upstream
                System.out.println("[" + timestamp() + "] [Processor " + name + "] RECEIVED SLOWDOWN at min priority, cascading upstream");
                signalUpstreamSlowdown();
            }
        }
    }

    /**
     * Check if we should recover from slowdown mode.
     * For source nodes: recovers FPS first (with longer interval), then priority.
     * For processing nodes: recovers priority after 10 seconds without slowdown signal.
     */
    protected synchronized void checkSlowdownRecovery() {
        if (!inSlowdownMode) {
            return;
        }

        long now = System.currentTimeMillis();
        long timeSinceSlowdown = now - lastSlowdownReceivedTime;

        // For source nodes, recover FPS first (with longer interval)
        if (isSource && fpsSlowdownLevel > 0) {
            if (timeSinceSlowdown >= FPS_RECOVERY_MS) {
                double oldFps = getEffectiveFps();
                fpsSlowdownLevel--;
                double newFps = getEffectiveFps();
                frameDelayMs = newFps > 0 ? (long) (1000.0 / newFps) : 0;
                System.out.println("[" + timestamp() + "] [Processor " + name + "] SLOWDOWN RECOVERY, " +
                    "raising FPS: " + String.format("%.3f", oldFps) + " -> " + String.format("%.3f", newFps) +
                    " (level " + fpsSlowdownLevel + "/" + MAX_FPS_SLOWDOWN_LEVELS + ")");
                lastSlowdownReceivedTime = now; // Reset timer for next recovery step
            }
            return; // Don't recover priority until FPS is fully restored
        }

        // Recover priority if reduced
        if (timeSinceSlowdown >= SLOWDOWN_RECOVERY_MS) {
            if (slowdownPriorityReduction > 0) {
                Thread pt = processingThread;
                if (pt != null && pt.isAlive()) {
                    int currentPriority = pt.getPriority();
                    int maxAllowedPriority = originalPriority;

                    if (currentPriority < maxAllowedPriority) {
                        int newPriority = currentPriority + 1;
                        System.out.println("[" + timestamp() + "] [Processor " + name + "] SLOWDOWN RECOVERY, " +
                            "raising priority: " + currentPriority + " -> " + newPriority);
                        pt.setPriority(newPriority);
                        lastRunningPriority = newPriority;
                        slowdownPriorityReduction--;
                        lastSlowdownReceivedTime = now; // Reset timer for next recovery step
                        return; // Only recover one thing per check
                    }
                }
            }

            // Fully recovered
            if (slowdownPriorityReduction <= 0 && fpsSlowdownLevel <= 0) {
                inSlowdownMode = false;
                System.out.println("[" + timestamp() + "] [Processor " + name + "] EXITED SLOWDOWN MODE");
            }
        }
    }

    /**
     * Start processing thread.
     */
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0;
        fpsSlowdownLevel = 0;
        slowdownPriorityReduction = 0;
        inSlowdownMode = false;

        processingThread = new Thread(this::processingLoop, "Processor-" + name);
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Main processing loop. Subclasses can override this to customize behavior.
     */
    protected void processingLoop() {
        System.out.println("[ThreadedProcessor] " + name + " starting processingLoop, inputQueue=" + (inputQueue != null ? "set" : "NULL") + ", outputQueue=" + (outputQueue != null ? "set" : "NULL"));
        long lastDebugTime = System.currentTimeMillis();
        while (running.get()) {
            try {
                if (inputQueue == null) {
                    Thread.sleep(100);
                    continue;
                }

                // Check for slowdown recovery (for processing nodes)
                if (!isSource) {
                    checkSlowdownRecovery();
                }

                Mat input = inputQueue.take();
                inputReads1++;

                // Update FXNode stats directly (each thread updates its own node)
                if (fxNode != null) {
                    fxNode.inputCount = (int) inputReads1;
                }

                // Debug every 2 seconds
                if (System.currentTimeMillis() - lastDebugTime > 2000) {
                    System.out.println("[ThreadedProcessor] " + name + " processed " + inputReads1 + " frames, outputWrites=" + outputWrites1);
                    lastDebugTime = System.currentTimeMillis();
                }

                if (input == null) {
                    continue;
                }

                Mat output;
                if (!enabled) {
                    // Bypass mode: pass through unchanged
                    output = input.clone();
                } else {
                    // Process the frame
                    output = processor.process(input);
                }

                workUnitsCompleted++;

                // Check backpressure after processing
                checkBackpressure();

                if (output != null) {
                    // Notify callback with a clone
                    if (onFrameCallback != null) {
                        Mat callbackCopy = output.clone();
                        onFrameCallback.accept(callbackCopy);
                    }

                    // Put on output queues (support dual output)
                    if (outputQueue != null) {
                        outputQueue.put(output.clone());
                        outputWrites1++;
                    }
                    if (outputQueue2 != null) {
                        outputQueue2.put(output.clone());
                        outputWrites2++;
                    }

                    // Update FXNode output stats directly
                    if (fxNode != null) {
                        fxNode.outputCount1 = (int) outputWrites1;
                        fxNode.outputCount2 = (int) outputWrites2;
                        fxNode.workUnitsCompleted = workUnitsCompleted;
                        fxNode.threadPriority = getThreadPriority();
                        fxNode.effectiveFps = getEffectiveFps();
                    }

                    output.release();
                }

                input.release();

            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    // === Protected helper methods for subclasses ===

    protected void incrementInputReads1() {
        inputReads1++;
        if (fxNode != null) {
            fxNode.inputCount = (int) inputReads1;
        }
    }

    protected void incrementWorkUnitsCompleted() {
        workUnitsCompleted++;
        if (fxNode != null) {
            fxNode.workUnitsCompleted = workUnitsCompleted;
            fxNode.threadPriority = getThreadPriority();
            fxNode.effectiveFps = getEffectiveFps();
        }
    }

    protected void incrementOutputWrites1() {
        outputWrites1++;
        if (fxNode != null) {
            fxNode.outputCount1 = (int) outputWrites1;
        }
    }

    protected void incrementOutputWrites2() {
        outputWrites2++;
        if (fxNode != null) {
            fxNode.outputCount2 = (int) outputWrites2;
        }
    }

    protected void notifyCallback(Mat frame) {
        if (onFrameCallback != null) {
            onFrameCallback.accept(frame);
        }
    }

    /**
     * Stop processing thread.
     */
    public void stopProcessing() {
        running.set(false);

        if (processingThread != null) {
            processingThread.interrupt();
            try {
                processingThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            processingThread = null;
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    public boolean hasActiveThread() {
        return processingThread != null && processingThread.isAlive();
    }
}
