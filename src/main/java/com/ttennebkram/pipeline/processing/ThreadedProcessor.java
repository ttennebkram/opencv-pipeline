package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.util.MatTracker;
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
    private BlockingQueue<Mat> outputQueue3; // For quad-output processors (e.g., FFT4)
    private BlockingQueue<Mat> outputQueue4; // For quad-output processors (e.g., FFT4)

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

    // Backpressure detection state - tracks if WE have signaled upstream
    private volatile boolean sentBackpressureSignal = false;

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

    // Throttle callbacks to avoid overwhelming JavaFX (max ~30 fps per node)
    private long lastCallbackTime = 0;
    private static final long MIN_CALLBACK_INTERVAL_MS = 33; // ~30 fps max

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
        // Read from FXNode for live updates when available
        if (fxNode != null) {
            return fxNode.enabled;
        }
        return enabled;
    }

    public void setDualInput(boolean dualInput) {
        this.isDualInput = dualInput;
    }

    public void setIsDualInput(boolean dualInput) {
        this.isDualInput = dualInput;
    }

    public boolean isDualInput() {
        return isDualInput;
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

    public void setOutputQueue3(BlockingQueue<Mat> queue) {
        this.outputQueue3 = queue;
    }

    public BlockingQueue<Mat> getOutputQueue3() {
        return outputQueue3;
    }

    public void setOutputQueue4(BlockingQueue<Mat> queue) {
        this.outputQueue4 = queue;
    }

    public BlockingQueue<Mat> getOutputQueue4() {
        return outputQueue4;
    }

    /**
     * Get output queue by index (0-based).
     */
    public BlockingQueue<Mat> getOutputQueue(int index) {
        switch (index) {
            case 0: return outputQueue;
            case 1: return outputQueue2;
            case 2: return outputQueue3;
            case 3: return outputQueue4;
            default: return null;
        }
    }

    /**
     * Set output queue by index (0-based).
     */
    public void setOutputQueue(int index, BlockingQueue<Mat> queue) {
        switch (index) {
            case 0: outputQueue = queue; break;
            case 1: outputQueue2 = queue; break;
            case 2: outputQueue3 = queue; break;
            case 3: outputQueue4 = queue; break;
        }
    }

    public BlockingQueue<Mat> getInputQueue() {
        return inputQueue;
    }

    public BlockingQueue<Mat> getInputQueue2() {
        return inputQueue2;
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
     * Check OUTPUT queues and handle backpressure.
     * If ANY output queue is backed up (>=5), it means DOWNSTREAM can't keep up.
     * We lower OUR OWN priority (to slow our processing rate) with cooldown.
     * If we're already at min priority, we signal UPSTREAM to slow down.
     *
     * Key behavior: Keep lowering priority with cooldown while queue stays >= 5.
     * When ALL queues drop below 5, recovery timer starts counting.
     */
    protected void checkBackpressure() {
        // First, check if we should recover from slowdown mode
        checkSlowdownRecovery();

        // Check BOTH output queues - if either is backing up, downstream can't keep up
        int outputQueueSize = outputQueue != null ? outputQueue.size() : 0;
        int outputQueue2Size = outputQueue2 != null ? outputQueue2.size() : 0;
        int maxQueueSize = Math.max(outputQueueSize, outputQueue2Size);

        // If no output queues to monitor, nothing to do
        if (outputQueue == null && outputQueue2 == null) {
            return;
        }

        long now = System.currentTimeMillis();

        // If any output queue is backed up (>=5), we need to slow down
        if (maxQueueSize >= 5) {
            // Respect cooldown - only lower priority once per second
            if (now - lastPriorityAdjustmentTime >= PRIORITY_LOWER_COOLDOWN_MS) {
                // Try to lower our own priority
                Thread pt = processingThread;
                if (pt != null && pt.isAlive()) {
                    int currentPriority = pt.getPriority();
                    if (currentPriority > Thread.MIN_PRIORITY) {
                        int newPriority = currentPriority - 1;
                        pt.setPriority(newPriority);
                        lastRunningPriority = newPriority;
                        inSlowdownMode = true;
                        lastSlowdownReceivedTime = now;
                        lastPriorityAdjustmentTime = now;
                    } else {
                        // Already at min priority, signal upstream to slow down
                        if (inputNode != null) {
                            inputNode.receiveSlowdownSignal();
                        }
                        lastSlowdownReceivedTime = now;
                        lastPriorityAdjustmentTime = now;
                    }
                }
            }
            // NOTE: Don't reset lastSlowdownReceivedTime here on every check - only when we take action
            // This allows recovery to start when queue drops below 5
        }
    }

    /**
     * Signal upstream nodes to slow down because this node is overwhelmed.
     * For boundary nodes (ContainerOutput), also signals the parent container.
     */
    private void signalUpstreamSlowdown() {
        if (inputNode != null) {
            inputNode.receiveSlowdownSignal();
        }
        if (inputNode2 != null) {
            inputNode2.receiveSlowdownSignal();
        }
        // For boundary nodes (ContainerOutput), signal the parent container
        if (parentContainer != null) {
            parentContainer.receiveSlowdownSignal();
        }
    }

    /**
     * Receive a slowdown signal from a downstream node.
     * For processing nodes: reduces priority and enters slowdown mode.
     * For source nodes: reduces priority first, then FPS if at min priority.
     */
    public synchronized void receiveSlowdownSignal() {
        long now = System.currentTimeMillis();
        lastSlowdownReceivedTime = now;
        inSlowdownMode = true;

        Thread pt = processingThread;
        if (pt != null && pt.isAlive()) {
            int currentPriority = pt.getPriority();

            if (currentPriority > Thread.MIN_PRIORITY) {
                // Reduce priority by 1
                int newPriority = currentPriority - 1;
                pt.setPriority(newPriority);
                lastRunningPriority = newPriority;
                slowdownPriorityReduction++;
            } else if (isSource && fpsSlowdownLevel < MAX_FPS_SLOWDOWN_LEVELS) {
                // Source node at min priority - reduce FPS instead
                fpsSlowdownLevel++;
                double effectiveFps = getEffectiveFps();
                frameDelayMs = effectiveFps > 0 ? (long) (1000.0 / effectiveFps) : 0;
            } else {
                // Already at minimum priority (and min FPS for sources), cascade upstream
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
                fpsSlowdownLevel--;
                double newFps = getEffectiveFps();
                frameDelayMs = newFps > 0 ? (long) (1000.0 / newFps) : 0;
                lastSlowdownReceivedTime = now; // Reset timer for next recovery step
            }
            return; // Don't recover priority until FPS is fully restored
        }

        // Recover priority if below original
        if (timeSinceSlowdown >= SLOWDOWN_RECOVERY_MS) {
            Thread pt = processingThread;
            if (pt != null && pt.isAlive()) {
                int currentPriority = pt.getPriority();

                if (currentPriority < originalPriority) {
                    // Raise priority by 1
                    int newPriority = currentPriority + 1;
                    pt.setPriority(newPriority);
                    lastRunningPriority = newPriority;
                    lastSlowdownReceivedTime = now; // Reset timer for next recovery step
                    return; // Only recover one step per check
                }
            }

            // Fully recovered - priority is back to original and FPS is not reduced
            if (fpsSlowdownLevel <= 0) {
                inSlowdownMode = false;
            }
        }
    }

    /**
     * Start processing thread.
     * Boundary nodes (ContainerInput/ContainerOutput) don't start their threads
     * when at root level (not embedded inside a container).
     */
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        // Boundary nodes should not run their threads at root level
        if (fxNode != null && isBoundaryNode() && !fxNode.isEmbedded) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0;
        fpsSlowdownLevel = 0;
        slowdownPriorityReduction = 0;
        inSlowdownMode = false;
        sentBackpressureSignal = false;

        processingThread = new Thread(this::processingLoop, "Processor-" + name);
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Check if this processor is for a boundary node (ContainerInput/ContainerOutput).
     */
    private boolean isBoundaryNode() {
        if (fxNode == null) return false;
        String nodeType = fxNode.nodeType;
        return "ContainerInput".equals(nodeType) || "ContainerOutput".equals(nodeType);
    }

    /**
     * Main processing loop. Subclasses can override this to customize behavior.
     */
    protected void processingLoop() {
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

                if (input == null) {
                    continue;
                }

                Mat output;
                if (!isEnabled()) {
                    // Bypass mode: pass through unchanged
                    output = input.clone();
                    MatTracker.track(output);
                } else {
                    // Process the frame
                    output = processor.process(input);
                    MatTracker.track(output);
                }

                workUnitsCompleted++;

                if (output != null) {
                    // Notify callback with a clone (throttled to avoid overwhelming JavaFX)
                    if (onFrameCallback != null) {
                        long now = System.currentTimeMillis();
                        if (now - lastCallbackTime >= MIN_CALLBACK_INTERVAL_MS) {
                            lastCallbackTime = now;
                            Mat callbackCopy = output.clone();
                            MatTracker.track(callbackCopy);
                            onFrameCallback.accept(callbackCopy);
                        }
                    }

                    // Put on output queues (support dual output)
                    if (outputQueue != null) {
                        Mat queueCopy = output.clone();
                        MatTracker.track(queueCopy);
                        outputQueue.put(queueCopy);
                        outputWrites1++;
                    }
                    if (outputQueue2 != null) {
                        Mat queueCopy2 = output.clone();
                        MatTracker.track(queueCopy2);
                        outputQueue2.put(queueCopy2);
                        outputWrites2++;
                    }

                    // Check backpressure AFTER putting on output queues
                    // This is when we can detect if queues are backing up
                    checkBackpressure();

                    // Update FXNode output stats directly
                    if (fxNode != null) {
                        fxNode.outputCount1 = (int) outputWrites1;
                        fxNode.outputCount2 = (int) outputWrites2;
                        fxNode.workUnitsCompleted = workUnitsCompleted;
                        fxNode.threadPriority = getThreadPriority();
                        fxNode.effectiveFps = getEffectiveFps();
                    }

                    MatTracker.release(output);
                }

                MatTracker.release(input);

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

    protected void incrementInputReads2() {
        inputReads2++;
        if (fxNode != null) {
            fxNode.inputCount2 = (int) inputReads2;
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
            long now = System.currentTimeMillis();
            if (now - lastCallbackTime >= MIN_CALLBACK_INTERVAL_MS) {
                lastCallbackTime = now;
                onFrameCallback.accept(frame);
            } else {
                // Throttled - release the frame since we're not using it
                MatTracker.release(frame);
            }
        } else {
            // No callback - release the frame
            MatTracker.release(frame);
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

        // Drain and release any remaining Mats in queues to prevent memory leaks
        drainQueues();
    }

    /**
     * Drain all queues and release any remaining Mats.
     */
    private void drainQueues() {
        // Drain input queues
        drainQueue(inputQueue);
        drainQueue(inputQueue2);

        // Drain output queues
        drainQueue(outputQueue);
        drainQueue(outputQueue2);
        drainQueue(outputQueue3);
        drainQueue(outputQueue4);
    }

    private void drainQueue(BlockingQueue<Mat> queue) {
        if (queue != null) {
            Mat mat;
            while ((mat = queue.poll()) != null) {
                MatTracker.release(mat);
            }
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    public boolean hasActiveThread() {
        return processingThread != null && processingThread.isAlive();
    }
}
