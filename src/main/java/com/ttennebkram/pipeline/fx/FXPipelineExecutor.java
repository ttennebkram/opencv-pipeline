package com.ttennebkram.pipeline.fx;

import com.ttennebkram.pipeline.processing.ContainerProcessor;
import com.ttennebkram.pipeline.processing.ProcessorFactory;
import com.ttennebkram.pipeline.processing.SourceProcessor;
import com.ttennebkram.pipeline.processing.ThreadedProcessor;
import javafx.application.Platform;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;

/**
 * Executes a pipeline of connected nodes, passing frames from sources through processors.
 * This is the JavaFX-native implementation with per-node threading.
 *
 * Architecture:
 * - Source nodes (Webcam, File, Blank) run in source feeder threads
 * - Processing nodes use ThreadedProcessor instances with their own threads
 * - Nodes communicate via BlockingQueues
 * - Each node runs its own OpenCV processing code in its own thread
 */
public class FXPipelineExecutor {

    private final List<FXNode> nodes;
    private final List<FXConnection> connections;
    private final Map<Integer, FXWebcamSource> webcamSources;

    // Per-node threading support
    private ProcessorFactory processorFactory;
    private final Map<Integer, Thread> sourceFeederThreads = new HashMap<>();
    private final Map<Integer, BlockingQueue<Mat>> sourceOutputQueues = new HashMap<>();

    private Thread executionThread;
    private AtomicBoolean running = new AtomicBoolean(false);

    // Callback for updating node thumbnails
    private BiConsumer<FXNode, Mat> onNodeOutput;

    // Frame rate control
    private double targetFps = 10.0;

    // FileSource state: cached images and last output times for FPS throttling
    private final Map<Integer, Mat> fileSourceCache = new HashMap<>();
    private final Map<Integer, Long> fileSourceLastOutput = new HashMap<>();

    // Flag to use per-node threading (can be toggled for comparison/debugging)
    private boolean usePerNodeThreading = true;

    // Base path for resolving relative pipeline file paths (for nested containers)
    private String basePath = null;

    public FXPipelineExecutor(List<FXNode> nodes, List<FXConnection> connections,
                              Map<Integer, FXWebcamSource> webcamSources) {
        this.nodes = nodes;
        this.connections = connections;
        this.webcamSources = webcamSources;
    }

    /**
     * Set the base path for resolving relative pipeline file paths.
     * This should be the path to the current pipeline document file.
     */
    public void setBasePath(String basePath) {
        this.basePath = basePath;
    }

    /**
     * Enable or disable per-node threading.
     * When disabled, uses the original single-threaded execution loop.
     */
    public void setUsePerNodeThreading(boolean enabled) {
        this.usePerNodeThreading = enabled;
    }

    /**
     * Set callback for when a node produces output.
     * Called on JavaFX Application Thread.
     */
    public void setOnNodeOutput(BiConsumer<FXNode, Mat> callback) {
        this.onNodeOutput = callback;
    }

    /**
     * Start pipeline execution.
     */
    public void start() {
        if (running.get()) return;

        running.set(true);

        if (usePerNodeThreading) {
            startPerNodeThreading();
            int processors = processorFactory.getProcessorCount();
            int totalThreads = processors + 1; // +1 for JavaFX
            System.out.println("[FXPipelineExecutor] Started multi-threaded mode: " + processors + " processors, " + totalThreads + " threads (1 for JavaFX)");
        } else {
            System.out.println("[FXPipelineExecutor] Started single-threaded mode: 2 threads (1 for JavaFX)");
            // Original single-threaded execution
            executionThread = new Thread(this::executionLoop, "FXPipelineExecutor");
            executionThread.setDaemon(true);
            executionThread.start();
        }
    }

    /**
     * Start per-node threading mode.
     * Creates ThreadedProcessor for each processing node,
     * wires them together with queues, and starts all threads.
     */
    private void startPerNodeThreading() {
        // Create processor factory and set up output callback
        processorFactory = new ProcessorFactory();
        processorFactory.setOnNodeOutput(output -> {
            if (onNodeOutput != null) {
                Platform.runLater(() -> {
                    onNodeOutput.accept(output.node, output.output);
                });
            }
        });

        // Build execution order
        List<FXNode> executionOrder = buildExecutionOrder();

        // Create processors for all nodes (including sources for backpressure signaling)
        for (FXNode node : executionOrder) {
            ThreadedProcessor tp = processorFactory.createProcessor(node);
            if (tp != null) {
                tp.setEnabled(node.enabled);
            }

            // For Container nodes, set up internal processors and wiring
            if ("Container".equals(node.nodeType) || node.isContainer) {
                boolean hasInnerNodes = node.innerNodes != null && !node.innerNodes.isEmpty();
                boolean hasExternalFile = node.pipelineFilePath != null && !node.pipelineFilePath.isEmpty();
                if (hasInnerNodes || hasExternalFile) {
                    setupContainerInternals(node);
                }
            }
        }

        // Wire up connections - create queues between nodes
        for (FXConnection conn : connections) {
            if (conn.source != null && conn.target != null) {
                if (isSourceNode(conn.source)) {
                    // Source -> Processing: create source output queue
                    BlockingQueue<Mat> queue = sourceOutputQueues.computeIfAbsent(
                        conn.source.id, k -> new LinkedBlockingQueue<>());
                    ThreadedProcessor targetProc = processorFactory.getProcessor(conn.target);
                    ThreadedProcessor sourceProc = processorFactory.getProcessor(conn.source);
                    if (targetProc != null) {
                        if (conn.targetInputIndex == 1) {
                            targetProc.setInputQueue2(queue);
                            // Wire upstream reference for backpressure signaling
                            if (sourceProc != null) {
                                targetProc.setInputNode2(sourceProc);
                            }
                        } else {
                            targetProc.setInputQueue(queue);
                            // Wire upstream reference for backpressure signaling
                            if (sourceProc != null) {
                                targetProc.setInputNode(sourceProc);
                            }
                        }
                    }
                } else {
                    // Processing -> Processing: wire via ProcessorFactory
                    processorFactory.wireConnection(conn.source, conn.target,
                        conn.sourceOutputIndex, conn.targetInputIndex);
                }
            }
        }

        // Start source feeder threads
        for (FXNode node : nodes) {
            if (isSourceNode(node)) {
                startSourceFeeder(node);
            }
        }

        // Start all processors (including container internals)
        processorFactory.startAll();
    }

    /**
     * Resolve a pipeline file path, handling both absolute and relative paths.
     * Relative paths are resolved against the basePath (parent document directory).
     */
    private String resolvePipelinePath(String pipelinePath) {
        if (pipelinePath == null || pipelinePath.isEmpty()) {
            return pipelinePath;
        }
        java.io.File file = new java.io.File(pipelinePath);
        // If it's already absolute, use it as-is
        if (file.isAbsolute()) {
            return pipelinePath;
        }
        // Relative path - resolve against basePath
        if (basePath != null && !basePath.isEmpty()) {
            java.io.File baseDir = new java.io.File(basePath);
            if (baseDir.isFile()) {
                baseDir = baseDir.getParentFile();
            }
            if (baseDir != null) {
                java.io.File resolved = new java.io.File(baseDir, pipelinePath);
                return resolved.getAbsolutePath();
            }
        }
        // No basePath available, return as-is
        return pipelinePath;
    }

    /**
     * Load a container's inner nodes from an external pipeline file if specified.
     * ALWAYS reloads from the external file at execution time to ensure we have the
     * complete pipeline (saved inline data may be incomplete/corrupted).
     */
    private void loadContainerFromExternalFile(FXNode containerNode) {
        if (containerNode.pipelineFilePath == null || containerNode.pipelineFilePath.isEmpty()) {
            return;
        }

        try {
            String resolvedPath = resolvePipelinePath(containerNode.pipelineFilePath);
            java.io.File pipelineFile = new java.io.File(resolvedPath);
            if (pipelineFile.exists()) {
                FXPipelineSerializer.PipelineDocument doc = FXPipelineSerializer.load(resolvedPath);

                // Instead of replacing nodes wholesale (which breaks editor references),
                // update existing nodes in place when possible, or add fresh nodes to the existing list.
                if (containerNode.innerNodes == null) {
                    containerNode.innerNodes = new java.util.ArrayList<>();
                }
                if (containerNode.innerConnections == null) {
                    containerNode.innerConnections = new java.util.ArrayList<>();
                }

                // Build a map from nodeType+label to existing nodes for matching
                Map<String, FXNode> existingByKey = new HashMap<>();
                for (FXNode existing : containerNode.innerNodes) {
                    String key = existing.nodeType + ":" + existing.label;
                    existingByKey.put(key, existing);
                }

                // Update or add nodes from the loaded document
                List<FXNode> updatedNodes = new ArrayList<>();
                for (FXNode loadedNode : doc.nodes) {
                    String key = loadedNode.nodeType + ":" + loadedNode.label;
                    FXNode existing = existingByKey.get(key);
                    if (existing != null) {
                        // Update existing node in place to preserve editor references
                        updateNodeInPlace(existing, loadedNode);
                        updatedNodes.add(existing);
                        existingByKey.remove(key); // Mark as used
                    } else {
                        // New node, add to list with fresh ID
                        loadedNode.reassignId();
                        // Also reassign IDs for nested containers
                        if (loadedNode.isContainer && loadedNode.innerNodes != null) {
                            reassignNodeIds(loadedNode.innerNodes);
                        }
                        updatedNodes.add(loadedNode);
                    }
                }

                // Clear and repopulate with updated/new nodes
                containerNode.innerNodes.clear();
                containerNode.innerNodes.addAll(updatedNodes);

                // Connections need to be rebuilt to reference the correct nodes
                // Build ID mapping from loaded doc nodes to actual nodes
                Map<Integer, FXNode> loadedIdToActual = new HashMap<>();
                for (int i = 0; i < doc.nodes.size(); i++) {
                    loadedIdToActual.put(doc.nodes.get(i).id, updatedNodes.get(i));
                }

                // Rebuild connections using actual nodes
                containerNode.innerConnections.clear();
                for (FXConnection loadedConn : doc.connections) {
                    FXNode actualSource = loadedIdToActual.get(loadedConn.source != null ? loadedConn.source.id : -1);
                    FXNode actualTarget = loadedIdToActual.get(loadedConn.target != null ? loadedConn.target.id : -1);
                    if (actualSource != null && actualTarget != null) {
                        FXConnection newConn = new FXConnection(actualSource, loadedConn.sourceOutputIndex,
                                                                 actualTarget, loadedConn.targetInputIndex);
                        containerNode.innerConnections.add(newConn);
                    }
                }

            } else {
                System.err.println("[FXPipelineExecutor] External pipeline file not found: " + resolvedPath);
            }
        } catch (Exception e) {
            System.err.println("[FXPipelineExecutor] Error loading external pipeline file: " + e.getMessage());
        }
    }

    /**
     * Update an existing FXNode in place with data from a loaded node.
     * This preserves the existing object reference so that any editors or
     * callbacks bound to this node continue to work.
     */
    private void updateNodeInPlace(FXNode existing, FXNode loaded) {
        // Update position and size
        existing.x = loaded.x;
        existing.y = loaded.y;
        existing.width = loaded.width;
        existing.height = loaded.height;

        // Update properties
        existing.enabled = loaded.enabled;
        existing.properties.clear();
        existing.properties.putAll(loaded.properties);

        // Update container-specific fields
        if (existing.isContainer && loaded.isContainer) {
            existing.pipelineFilePath = loaded.pipelineFilePath;
            // For nested containers, recursively update inner nodes in place to preserve editor references
            if (loaded.innerNodes != null && !loaded.innerNodes.isEmpty()) {
                if (existing.innerNodes == null) {
                    existing.innerNodes = new ArrayList<>();
                }
                // Use the same in-place update strategy recursively
                updateInnerNodesInPlace(existing.innerNodes, loaded.innerNodes);
            }
            // Rebuild connections to reference the actual nodes
            if (loaded.innerConnections != null && !loaded.innerConnections.isEmpty()) {
                if (existing.innerConnections == null) {
                    existing.innerConnections = new ArrayList<>();
                }
                // Build ID mapping from loaded nodes to existing nodes
                Map<Integer, FXNode> loadedIdToExisting = new HashMap<>();
                for (int i = 0; i < loaded.innerNodes.size() && i < existing.innerNodes.size(); i++) {
                    loadedIdToExisting.put(loaded.innerNodes.get(i).id, existing.innerNodes.get(i));
                }
                existing.innerConnections.clear();
                for (FXConnection conn : loaded.innerConnections) {
                    FXNode srcNode = loadedIdToExisting.get(conn.source != null ? conn.source.id : -1);
                    FXNode tgtNode = loadedIdToExisting.get(conn.target != null ? conn.target.id : -1);
                    if (srcNode != null && tgtNode != null) {
                        existing.innerConnections.add(new FXConnection(srcNode, conn.sourceOutputIndex,
                                                                        tgtNode, conn.targetInputIndex));
                    }
                }
            }
        }
    }

    /**
     * Recursively reassign unique IDs to all nodes in a list to avoid collisions
     * with outer pipeline nodes. This is needed when loading container internals
     * from external files.
     */
    private void reassignNodeIds(java.util.List<FXNode> nodes) {
        for (FXNode node : nodes) {
            node.reassignId();
            // Recursively handle nested containers
            if (node.isContainer && node.innerNodes != null && !node.innerNodes.isEmpty()) {
                reassignNodeIds(node.innerNodes);
            }
        }
    }

    /**
     * Update existing nodes in place from loaded nodes, preserving object identity.
     * This is critical for nested containers where editors may be bound to nodes.
     * Matches nodes by nodeType+label key.
     */
    private void updateInnerNodesInPlace(List<FXNode> existing, List<FXNode> loaded) {
        // Build a map from nodeType+label to existing nodes
        Map<String, FXNode> existingByKey = new HashMap<>();
        for (FXNode node : existing) {
            String key = node.nodeType + ":" + node.label;
            existingByKey.put(key, node);
        }

        // Update or add nodes
        List<FXNode> updatedNodes = new ArrayList<>();
        for (FXNode loadedNode : loaded) {
            String key = loadedNode.nodeType + ":" + loadedNode.label;
            FXNode existingNode = existingByKey.get(key);
            if (existingNode != null) {
                // Update existing node in place - recursively call updateNodeInPlace
                updateNodeInPlace(existingNode, loadedNode);
                updatedNodes.add(existingNode);
                existingByKey.remove(key);
            } else {
                // New node, add with fresh ID
                loadedNode.reassignId();
                if (loadedNode.isContainer && loadedNode.innerNodes != null) {
                    reassignNodeIds(loadedNode.innerNodes);
                }
                updatedNodes.add(loadedNode);
            }
        }

        // Replace the contents of the existing list
        existing.clear();
        existing.addAll(updatedNodes);
    }

    /**
     * Set up the internal processors for a Container node.
     * Creates processors for all inner nodes including boundary nodes,
     * wires internal connections, and connects boundary nodes to the container.
     */
    private void setupContainerInternals(FXNode containerNode) {
        // If container has an external pipeline file, load from it first
        if (containerNode.pipelineFilePath != null && !containerNode.pipelineFilePath.isEmpty()) {
            loadContainerFromExternalFile(containerNode);
        }

        if (containerNode.innerNodes == null || containerNode.innerNodes.isEmpty()) {
            System.err.println("[FXPipelineExecutor] Container " + containerNode.label + " has no inner nodes after loading!");
            return;
        }

        // Find boundary nodes
        FXNode boundaryInput = null;
        FXNode boundaryOutput = null;
        for (FXNode innerNode : containerNode.innerNodes) {
            if ("ContainerInput".equals(innerNode.nodeType)) {
                boundaryInput = innerNode;
            } else if ("ContainerOutput".equals(innerNode.nodeType)) {
                boundaryOutput = innerNode;
            }
        }

        if (boundaryInput == null || boundaryOutput == null) {
            System.err.println("Container " + containerNode.label + " missing boundary nodes");
            return;
        }

        // Create processors for all inner nodes (including boundary nodes)
        for (FXNode innerNode : containerNode.innerNodes) {
            ThreadedProcessor tp = processorFactory.createProcessor(innerNode);
            if (tp != null) {
                tp.setEnabled(innerNode.enabled);
            }

            // Recursively set up nested containers
            // Check for pipelineFilePath too, since we may need to load from external file
            if ("Container".equals(innerNode.nodeType) || innerNode.isContainer) {
                boolean hasInnerNodes = innerNode.innerNodes != null && !innerNode.innerNodes.isEmpty();
                boolean hasExternalFile = innerNode.pipelineFilePath != null && !innerNode.pipelineFilePath.isEmpty();
                if (hasInnerNodes || hasExternalFile) {
                    setupContainerInternals(innerNode);
                }
            }
        }

        // Wire internal connections between inner nodes
        if (containerNode.innerConnections != null) {
            for (FXConnection conn : containerNode.innerConnections) {
                if (conn.source != null && conn.target != null) {
                    processorFactory.wireConnection(conn.source, conn.target,
                        conn.sourceOutputIndex, conn.targetInputIndex);
                }
            }
        }

        // Get the container's processor and boundary processors
        ThreadedProcessor containerProc = processorFactory.getProcessor(containerNode);
        ThreadedProcessor inputProc = processorFactory.getProcessor(boundaryInput);
        ThreadedProcessor outputProc = processorFactory.getProcessor(boundaryOutput);

        if (containerProc == null || inputProc == null || outputProc == null) {
            System.err.println("Failed to get processors for container wiring");
            return;
        }

        // Create internal queues to connect container thread to boundary nodes:
        // Container -> ContainerInput queue
        BlockingQueue<Mat> containerToInputQueue = new LinkedBlockingQueue<>();
        // ContainerOutput -> Container queue
        BlockingQueue<Mat> outputToContainerQueue = new LinkedBlockingQueue<>();

        // Wire: Container's output -> ContainerInput's input
        // The container processor will put frames into this queue
        inputProc.setInputQueue(containerToInputQueue);

        // Wire: ContainerOutput's output -> Container can read from this
        outputProc.setOutputQueue(outputToContainerQueue);

        // Store the queues in the container processor so it can use them
        processorFactory.setContainerQueues(containerNode, containerToInputQueue, outputToContainerQueue);

        // Wire backpressure signaling for container:
        // When internal pipeline backs up, ContainerInput's outputQueue fills.
        // ContainerInput signals parentContainer (ContainerProcessor), which then
        // signals upstream in the main diagram.
        // NOTE: We do NOT wire cp.setContainerOutputProcessor() because that would create
        // a cycle: downstream slowdown -> Container -> ContainerOutput -> internal pipeline
        // -> ContainerInput -> parentContainer (Container) -> LOOP
        if (containerProc instanceof ContainerProcessor) {
            ContainerProcessor cp = (ContainerProcessor) containerProc;
            inputProc.setParentContainer(cp);
        } else {
            System.err.println("WARNING: containerProc for " + containerNode.label + " is NOT a ContainerProcessor: " +
                               (containerProc != null ? containerProc.getClass().getSimpleName() : "null"));
        }
    }

    /**
     * Start a feeder thread for a source node.
     * This thread reads from the source (webcam, file, blank) and feeds
     * frames into the output queue at the FPS controlled by the SourceProcessor.
     * Backpressure is handled via the SourceProcessor's receiveSlowdownSignal().
     */
    private void startSourceFeeder(FXNode sourceNode) {
        BlockingQueue<Mat> outputQueue = sourceOutputQueues.get(sourceNode.id);
        if (outputQueue == null) {
            // No connections from this source, skip
            return;
        }

        // Get the SourceProcessor for this node (created by ProcessorFactory)
        ThreadedProcessor tp = processorFactory.getProcessor(sourceNode);
        if (tp == null) {
            System.err.println("[FXPipelineExecutor] Source " + sourceNode.label + " (id=" + sourceNode.id +
                               ") has NO processor! Check if it was in executionOrder.");
            return;
        }
        if (!(tp instanceof SourceProcessor)) {
            System.err.println("[FXPipelineExecutor] Source " + sourceNode.label + " (id=" + sourceNode.id +
                               ") has wrong processor type: " + tp.getClass().getSimpleName() + " (expected SourceProcessor)");
            return;
        }
        SourceProcessor sourceProc = (SourceProcessor) tp;

        Thread feederThread = new Thread(() -> {
            while (running.get()) {
                long startTime = System.currentTimeMillis();

                // Get effective FPS from SourceProcessor (may be reduced due to backpressure)
                double effectiveFps = sourceProc.getEffectiveFps();
                long frameDelayMs = effectiveFps > 0 ? (long) (1000.0 / effectiveFps) : 1000;

                try {
                    Mat frame = getSourceFrame(sourceNode);
                    if (frame != null) {
                        // Update stats via SourceProcessor
                        sourceProc.incrementWorkUnits();
                        sourceNode.outputCount1++;

                        // Send thumbnail update
                        if (onNodeOutput != null) {
                            Mat copy = frame.clone();
                            Platform.runLater(() -> {
                                onNodeOutput.accept(sourceNode, copy);
                            });
                        }

                        // Put frame on output queue
                        outputQueue.put(frame);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                } catch (Exception e) {
                    System.err.println("Source feeder error for " + sourceNode.label + ": " + e.getMessage());
                }

                // Maintain frame rate (using effective FPS from SourceProcessor)
                long elapsed = System.currentTimeMillis() - startTime;
                long sleepTime = frameDelayMs - elapsed;
                if (sleepTime > 0) {
                    try {
                        Thread.sleep(sleepTime);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
        }, "SourceFeeder-" + sourceNode.label);

        feederThread.setDaemon(true);
        feederThread.start();
        sourceFeederThreads.put(sourceNode.id, feederThread);
    }

    /**
     * Get a frame from a source node.
     */
    private Mat getSourceFrame(FXNode sourceNode) {
        String type = sourceNode.nodeType;

        if ("WebcamSource".equals(type)) {
            FXWebcamSource webcam = webcamSources.get(sourceNode.id);
            if (webcam != null) {
                return webcam.getLastFrameClone();
            }
        } else if ("FileSource".equals(type)) {
            if (sourceNode.filePath != null && !sourceNode.filePath.isEmpty()) {
                Mat cached = fileSourceCache.get(sourceNode.id);
                if (cached == null || cached.empty()) {
                    cached = org.opencv.imgcodecs.Imgcodecs.imread(sourceNode.filePath);
                    if (!cached.empty()) {
                        fileSourceCache.put(sourceNode.id, cached);
                    }
                }
                if (cached != null && !cached.empty()) {
                    return cached.clone();
                }
            }
        } else if ("BlankSource".equals(type)) {
            return new Mat(480, 640, org.opencv.core.CvType.CV_8UC3, new Scalar(255, 255, 255));
        }

        return null;
    }

    /**
     * Stop pipeline execution.
     */
    public void stop() {
        running.set(false);

        // Stop per-node threading if active
        if (processorFactory != null) {
            processorFactory.stopAll();
            processorFactory.clear();
            processorFactory = null;
        }

        // Stop source feeder threads
        for (Thread feederThread : sourceFeederThreads.values()) {
            feederThread.interrupt();
            try {
                feederThread.join(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        sourceFeederThreads.clear();
        sourceOutputQueues.clear();

        // Stop single-threaded executor if active
        if (executionThread != null) {
            executionThread.interrupt();
            try {
                executionThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            executionThread = null;
        }

        // Clean up FileSource cached images
        for (Mat cached : fileSourceCache.values()) {
            if (cached != null && !cached.empty()) {
                cached.release();
            }
        }
        fileSourceCache.clear();
        fileSourceLastOutput.clear();
    }

    /**
     * Check if pipeline is running.
     */
    public boolean isRunning() {
        return running.get();
    }

    /**
     * Get the number of processing threads currently active.
     */
    public int getThreadCount() {
        if (processorFactory != null) {
            return processorFactory.getProcessorCount();
        }
        return usePerNodeThreading ? 0 : 1;
    }

    /**
     * Invalidate the cached image for a FileSource node.
     * Call this when the file path changes while the pipeline is running.
     */
    public void invalidateFileSourceCache(int nodeId) {
        Mat cached = fileSourceCache.remove(nodeId);
        if (cached != null && !cached.empty()) {
            cached.release();
        }
        fileSourceLastOutput.remove(nodeId);
    }

    /**
     * Sync stats from all processors back to their FXNodes.
     * Call this periodically from UI thread to update displayed stats.
     */
    public void syncAllStats() {
        if (processorFactory != null) {
            for (FXNode node : nodes) {
                // Sync stats for all nodes including sources
                // SourceProcessor updates threadPriority, workUnitsCompleted, and effectiveFps
                processorFactory.syncStats(node);

                // Recursively sync inner nodes for Container nodes
                if (("Container".equals(node.nodeType) || node.isContainer) && node.innerNodes != null) {
                    syncContainerInnerStats(node);
                }
            }
            // Sync queue sizes for connections
            syncConnectionQueues();
        }
    }

    /**
     * Recursively sync stats for all inner nodes of a container, including nested containers.
     */
    private void syncContainerInnerStats(FXNode containerNode) {
        if (containerNode.innerNodes == null) return;

        for (FXNode innerNode : containerNode.innerNodes) {
            processorFactory.syncStats(innerNode);

            // Recursively sync nested containers
            if (("Container".equals(innerNode.nodeType) || innerNode.isContainer) && innerNode.innerNodes != null) {
                syncContainerInnerStats(innerNode);
            }
        }

        // Sync inner connection queue stats
        if (containerNode.innerConnections != null) {
            syncInnerConnectionQueues(containerNode.innerConnections);
        }
    }

    /**
     * Sync queue sizes for inner connections of a Container node.
     */
    private void syncInnerConnectionQueues(List<FXConnection> innerConnections) {
        for (FXConnection conn : innerConnections) {
            if (conn.source != null && conn.target != null) {
                ThreadedProcessor sourceProc = processorFactory.getProcessor(conn.source);
                if (sourceProc != null) {
                    BlockingQueue<Mat> queue = conn.sourceOutputIndex == 1 ?
                        sourceProc.getOutputQueue2() : sourceProc.getOutputQueue();
                    if (queue != null) {
                        conn.queueSize = queue.size();
                    }
                    conn.totalFrames = sourceProc.getWorkUnitsCompleted();
                }
            }
        }
    }

    /**
     * Sync queue sizes from actual BlockingQueues to FXConnection objects.
     */
    private void syncConnectionQueues() {
        for (FXConnection conn : connections) {
            if (conn.source != null && conn.target != null) {
                if (isSourceNode(conn.source)) {
                    // Source -> Processing: get queue from sourceOutputQueues
                    BlockingQueue<Mat> queue = sourceOutputQueues.get(conn.source.id);
                    if (queue != null) {
                        conn.queueSize = queue.size();
                        conn.totalFrames = conn.source.outputCount1;
                    }
                } else {
                    // Processing -> Processing: get queue from processor
                    ThreadedProcessor sourceProc = processorFactory.getProcessor(conn.source);
                    if (sourceProc != null) {
                        BlockingQueue<Mat> queue = sourceProc.getOutputQueue();
                        if (queue != null) {
                            conn.queueSize = queue.size();
                        }
                        conn.totalFrames = sourceProc.getWorkUnitsCompleted();
                    }
                }
            }
        }
    }

    /**
     * Main execution loop.
     */
    private void executionLoop() {
        long frameDelayMs = (long) (1000.0 / targetFps);

        // Build execution order (topological sort from sources)
        List<FXNode> executionOrder = buildExecutionOrder();

        while (running.get()) {
            long startTime = System.currentTimeMillis();

            // Map to store current frame for each node
            Map<Integer, Mat> nodeOutputs = new HashMap<>();

            try {
                // Process nodes in order
                for (FXNode node : executionOrder) {
                    if (!node.enabled) {
                        // Pass through from input if disabled
                        Mat input = getInputForNode(node, nodeOutputs);
                        if (input != null) {
                            // Store at output index 0 using compound key
                            nodeOutputs.put(outputKey(node.id, 0), input.clone());
                        }
                        continue;
                    }

                    // Check if this is a multi-output node
                    if (isMultiOutputNode(node.nodeType)) {
                        Mat[] outputs = processMultiOutputNode(node, nodeOutputs);
                        if (outputs != null) {
                            // Store each output with its compound key
                            for (int i = 0; i < outputs.length; i++) {
                                if (outputs[i] != null) {
                                    nodeOutputs.put(outputKey(node.id, i), outputs[i]);
                                }
                            }
                            // Notify output callback with primary output (filtered image)
                            if (onNodeOutput != null && outputs[0] != null) {
                                Mat outputCopy = outputs[0].clone();
                                Platform.runLater(() -> {
                                    onNodeOutput.accept(node, outputCopy);
                                });
                            }
                        }
                    } else {
                        Mat output = processNode(node, nodeOutputs);
                        if (output != null) {
                            // Store at output index 0 using compound key
                            nodeOutputs.put(outputKey(node.id, 0), output);

                            // Notify output callback
                            if (onNodeOutput != null) {
                                Mat outputCopy = output.clone();
                                Platform.runLater(() -> {
                                    onNodeOutput.accept(node, outputCopy);
                                    // Note: callback is responsible for releasing
                                });
                            }
                        }
                    }
                }

                // Release all mats
                for (Mat mat : nodeOutputs.values()) {
                    mat.release();
                }
                nodeOutputs.clear();

            } catch (Exception e) {
                System.err.println("Pipeline execution error: " + e.getMessage());
                e.printStackTrace();
            }

            // Maintain frame rate
            long elapsed = System.currentTimeMillis() - startTime;
            long sleepTime = frameDelayMs - elapsed;
            if (sleepTime > 0) {
                try {
                    Thread.sleep(sleepTime);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
    }

    /**
     * Build execution order using topological sort.
     */
    private List<FXNode> buildExecutionOrder() {
        List<FXNode> order = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();

        // Find source nodes (no input connections)
        for (FXNode node : nodes) {
            if (!node.hasInput || isSourceNode(node)) {
                visitNode(node, visited, order);
            }
        }

        // Add any remaining nodes (disconnected)
        for (FXNode node : nodes) {
            if (!visited.contains(node.id)) {
                visitNode(node, visited, order);
            }
        }

        return order;
    }

    private void visitNode(FXNode node, Set<Integer> visited, List<FXNode> order) {
        if (visited.contains(node.id)) return;
        visited.add(node.id);

        // First visit all nodes this one depends on
        for (FXConnection conn : connections) {
            if (conn.target == node && conn.source != null) {
                visitNode(conn.source, visited, order);
            }
        }

        order.add(node);

        // Then visit nodes that depend on this one
        for (FXConnection conn : connections) {
            if (conn.source == node && conn.target != null) {
                visitNode(conn.target, visited, order);
            }
        }
    }

    private boolean isSourceNode(FXNode node) {
        return "WebcamSource".equals(node.nodeType) ||
               "FileSource".equals(node.nodeType) ||
               "BlankSource".equals(node.nodeType);
    }

    /**
     * Get input frame for a node from its connected source (input index 0).
     * Also updates connection statistics when data flows through.
     */
    private Mat getInputForNode(FXNode node, Map<Integer, Mat> nodeOutputs) {
        return getInputForNodeByIndex(node, nodeOutputs, 0);
    }

    /**
     * Create a compound key for storing multiple outputs per node.
     * Uses nodeId * 10 + outputIndex to allow up to 10 outputs per node.
     */
    private static int outputKey(int nodeId, int outputIndex) {
        return nodeId * 10 + outputIndex;
    }

    /**
     * Get input frame for a node from a specific input index.
     * Used for dual-input nodes that have multiple input connections.
     * Also updates connection statistics when data flows through.
     */
    private Mat getInputForNodeByIndex(FXNode node, Map<Integer, Mat> nodeOutputs, int inputIndex) {
        for (FXConnection conn : connections) {
            if (conn.target == node && conn.targetInputIndex == inputIndex && conn.source != null) {
                // Look up source output using compound key (nodeId * 10 + outputIndex)
                Mat sourceOutput = nodeOutputs.get(outputKey(conn.source.id, conn.sourceOutputIndex));
                if (sourceOutput != null) {
                    // Update connection statistics
                    conn.totalFrames++;
                    // Update source node's output counter based on which output port this connection uses
                    switch (conn.sourceOutputIndex) {
                        case 0: conn.source.outputCount1++; break;
                        case 1: conn.source.outputCount2++; break;
                        case 2: conn.source.outputCount3++; break;
                        case 3: conn.source.outputCount4++; break;
                    }
                    return sourceOutput.clone();
                }
            }
        }
        return null;
    }

    /**
     * Process a multi-output node and return all outputs.
     * Currently supports FFTHighPass4 and FFTLowPass4.
     */
    private Mat[] processMultiOutputNode(FXNode node, Map<Integer, Mat> nodeOutputs) {
        String type = node.nodeType;
        System.out.println("[FXPipelineExecutor] processMultiOutputNode called for type: " + type);

        // Get input for processing nodes
        Mat input = getInputForNode(node, nodeOutputs);
        if (input == null) return null;

        // Increment input counter
        node.inputCount++;

        // Get radius and smoothness from node properties
        int radius = 30;  // Default
        int smoothness = 0;  // Default (hard edge)

        Object radiusProp = node.properties.get("radius");
        if (radiusProp instanceof Number) {
            radius = ((Number) radiusProp).intValue();
        }

        Object smoothnessProp = node.properties.get("smoothness");
        if (smoothnessProp instanceof Number) {
            smoothness = ((Number) smoothnessProp).intValue();
        }

        Mat[] outputs;
        try {
            if ("FFTHighPass4".equals(type)) {
                outputs = processFFTHighPass4(input, radius, smoothness);
            } else if ("FFTLowPass4".equals(type)) {
                outputs = processFFTLowPass4(input, radius, smoothness);
            } else {
                // Fallback - shouldn't happen
                outputs = new Mat[] { input.clone(), null, null, null };
            }

            // Update output counters
            if (outputs != null) {
                if (outputs[0] != null) node.outputCount1++;
                if (outputs[1] != null) node.outputCount2++;
                if (outputs[2] != null) node.outputCount3++;
                if (outputs[3] != null) node.outputCount4++;
            }
        } catch (Exception e) {
            System.err.println("Error processing multi-output node " + type + ": " + e.getMessage());
            outputs = new Mat[] { input.clone(), null, null, null };
        }

        input.release();
        return outputs;
    }

    /**
     * Process a single node and return its output.
     */
    private Mat processNode(FXNode node, Map<Integer, Mat> nodeOutputs) {
        String type = node.nodeType;

        // Source nodes
        if ("WebcamSource".equals(type)) {
            FXWebcamSource webcam = webcamSources.get(node.id);
            if (webcam != null) {
                Mat frame = webcam.getLastFrameClone();
                if (frame != null) {
                    node.outputCount1++;
                }
                return frame;
            }
            return null;
        }

        if ("FileSource".equals(type)) {
            // Load image from file path with FPS throttling
            if (node.filePath != null && !node.filePath.isEmpty()) {
                // Determine effective FPS: -1 means automatic (1 FPS for images)
                double effectiveFps = node.fps;
                if (effectiveFps < 0) {
                    effectiveFps = 1.0;  // Default 1 FPS for static images
                }

                // Check if enough time has passed since last output
                long now = System.currentTimeMillis();
                Long lastOutput = fileSourceLastOutput.get(node.id);
                long frameIntervalMs = (long) (1000.0 / effectiveFps);

                if (lastOutput != null && (now - lastOutput) < frameIntervalMs) {
                    // Not time for a new frame yet - return null to skip
                    return null;
                }

                // Time for a new frame - use cached image if available
                Mat cached = fileSourceCache.get(node.id);
                if (cached == null || cached.empty()) {
                    // Load and cache the image
                    cached = org.opencv.imgcodecs.Imgcodecs.imread(node.filePath);
                    if (!cached.empty()) {
                        fileSourceCache.put(node.id, cached);
                    }
                }

                if (cached != null && !cached.empty()) {
                    fileSourceLastOutput.put(node.id, now);
                    node.outputCount1++;
                    return cached.clone();  // Return a clone so the cached version isn't modified
                }
            }
            return null;
        }

        if ("BlankSource".equals(type)) {
            // Create a blank 640x480 white image
            Mat blank = new Mat(480, 640, org.opencv.core.CvType.CV_8UC3, new Scalar(255, 255, 255));
            node.outputCount1++;
            return blank;
        }

        // Get input for processing nodes
        Mat input = getInputForNode(node, nodeOutputs);
        if (input == null) return null;

        // Increment input counter
        node.inputCount++;

        // Process based on node type
        Mat output = new Mat();

        try {
            switch (type) {
                case "Grayscale":
                    Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    break;

                case "Invert":
                    Core.bitwise_not(input, output);
                    break;

                case "GaussianBlur":
                    Imgproc.GaussianBlur(input, output, new org.opencv.core.Size(15, 15), 0);
                    break;

                case "MedianBlur":
                    Imgproc.medianBlur(input, output, 15);
                    break;

                case "CannyEdge":
                    Mat gray = new Mat();
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.Canny(gray, output, 50, 150);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    gray.release();
                    break;

                case "Sobel":
                    Mat grayS = new Mat();
                    Imgproc.cvtColor(input, grayS, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.Sobel(grayS, output, org.opencv.core.CvType.CV_8U, 1, 1);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayS.release();
                    break;

                case "Laplacian":
                    Mat grayL = new Mat();
                    Imgproc.cvtColor(input, grayL, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.Laplacian(grayL, output, org.opencv.core.CvType.CV_8U);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayL.release();
                    break;

                case "Threshold":
                    Mat grayT = new Mat();
                    Imgproc.cvtColor(input, grayT, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.threshold(grayT, output, 127, 255, Imgproc.THRESH_BINARY);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayT.release();
                    break;

                case "Erode":
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.erode(input, output, kernel);
                    kernel.release();
                    break;

                case "Dilate":
                    Mat kernelD = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.dilate(input, output, kernelD);
                    kernelD.release();
                    break;

                case "BilateralFilter":
                    Imgproc.bilateralFilter(input, output, 9, 75, 75);
                    break;

                case "BoxBlur":
                    Imgproc.blur(input, output, new org.opencv.core.Size(15, 15));
                    break;

                case "AdaptiveThreshold":
                    Mat grayAT = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayAT, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayAT);
                    }
                    Mat threshAT = new Mat();
                    Imgproc.adaptiveThreshold(grayAT, threshAT, 255,
                        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
                    Imgproc.cvtColor(threshAT, output, Imgproc.COLOR_GRAY2BGR);
                    grayAT.release();
                    threshAT.release();
                    break;

                case "Gain":
                    double gainValue = 1.0;
                    if (node.properties.containsKey("gain")) {
                        Object gainObj = node.properties.get("gain");
                        if (gainObj instanceof Number) {
                            gainValue = ((Number) gainObj).doubleValue();
                        }
                    }
                    input.convertTo(output, -1, gainValue, 0);
                    break;

                case "CLAHE":
                    Mat grayCL = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayCL, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayCL);
                    }
                    org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(2.0, new org.opencv.core.Size(8, 8));
                    Mat claheDst = new Mat();
                    clahe.apply(grayCL, claheDst);
                    Imgproc.cvtColor(claheDst, output, Imgproc.COLOR_GRAY2BGR);
                    grayCL.release();
                    claheDst.release();
                    break;

                case "Scharr":
                    Mat graySch = new Mat();
                    Imgproc.cvtColor(input, graySch, Imgproc.COLOR_BGR2GRAY);
                    Mat scharrX = new Mat();
                    Mat scharrY = new Mat();
                    Imgproc.Scharr(graySch, scharrX, org.opencv.core.CvType.CV_16S, 1, 0);
                    Imgproc.Scharr(graySch, scharrY, org.opencv.core.CvType.CV_16S, 0, 1);
                    Core.convertScaleAbs(scharrX, scharrX);
                    Core.convertScaleAbs(scharrY, scharrY);
                    Core.addWeighted(scharrX, 0.5, scharrY, 0.5, 0, output);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    graySch.release();
                    scharrX.release();
                    scharrY.release();
                    break;

                case "MorphOpen":
                    Mat kernelOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_OPEN, kernelOpen);
                    kernelOpen.release();
                    break;

                case "MorphClose":
                    Mat kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, kernelClose);
                    kernelClose.release();
                    break;

                case "MorphologyEx":
                    // Default to gradient operation
                    Mat kernelMorph = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_GRADIENT, kernelMorph);
                    kernelMorph.release();
                    break;

                case "BitwiseNot":
                    Core.bitwise_not(input, output);
                    break;

                case "ColorInRange":
                    // Default HSV range for blue color detection
                    Mat hsv = new Mat();
                    Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
                    Mat mask = new Mat();
                    Core.inRange(hsv, new Scalar(100, 50, 50), new Scalar(130, 255, 255), mask);
                    Imgproc.cvtColor(mask, output, Imgproc.COLOR_GRAY2BGR);
                    hsv.release();
                    mask.release();
                    break;

                case "MeanShift":
                    Imgproc.pyrMeanShiftFiltering(input, output, 21, 51);
                    break;

                case "Clone":
                case "Monitor":
                    // Pass through - these just duplicate/monitor the signal
                    input.copyTo(output);
                    break;

                case "Contours":
                    // Detect and draw contours
                    Mat grayCont = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayCont, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayCont);
                    }
                    Mat binary = new Mat();
                    Imgproc.threshold(grayCont, binary, 127, 255, Imgproc.THRESH_BINARY);
                    java.util.List<org.opencv.core.MatOfPoint> contours = new java.util.ArrayList<>();
                    Mat hierarchy = new Mat();
                    Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                    input.copyTo(output);
                    Imgproc.drawContours(output, contours, -1, new Scalar(0, 255, 0), 2);
                    grayCont.release();
                    binary.release();
                    hierarchy.release();
                    break;

                case "HoughCircles":
                    // Detect and draw circles
                    Mat grayHC = new Mat();
                    Imgproc.cvtColor(input, grayHC, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.GaussianBlur(grayHC, grayHC, new org.opencv.core.Size(9, 9), 2);
                    Mat circles = new Mat();
                    Imgproc.HoughCircles(grayHC, circles, Imgproc.HOUGH_GRADIENT, 1, 50, 100, 30, 10, 100);
                    input.copyTo(output);
                    for (int i = 0; i < circles.cols(); i++) {
                        double[] c = circles.get(0, i);
                        org.opencv.core.Point center = new org.opencv.core.Point(c[0], c[1]);
                        int radius = (int) Math.round(c[2]);
                        Imgproc.circle(output, center, radius, new Scalar(0, 255, 0), 2);
                        Imgproc.circle(output, center, 3, new Scalar(0, 0, 255), -1);
                    }
                    grayHC.release();
                    circles.release();
                    break;

                case "HoughLines":
                    // Detect and draw lines
                    Mat grayHL = new Mat();
                    Imgproc.cvtColor(input, grayHL, Imgproc.COLOR_BGR2GRAY);
                    Mat edges = new Mat();
                    Imgproc.Canny(grayHL, edges, 50, 150);
                    Mat lines = new Mat();
                    Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, 50, 10);
                    input.copyTo(output);
                    for (int i = 0; i < lines.rows(); i++) {
                        double[] l = lines.get(i, 0);
                        Imgproc.line(output, new org.opencv.core.Point(l[0], l[1]),
                            new org.opencv.core.Point(l[2], l[3]), new Scalar(0, 0, 255), 2);
                    }
                    grayHL.release();
                    edges.release();
                    lines.release();
                    break;

                case "HarrisCorners":
                    // Detect Harris corners
                    Mat grayHarris = new Mat();
                    Imgproc.cvtColor(input, grayHarris, Imgproc.COLOR_BGR2GRAY);
                    Mat dst = new Mat();
                    Imgproc.cornerHarris(grayHarris, dst, 2, 3, 0.04);
                    Mat dstNorm = new Mat();
                    Core.normalize(dst, dstNorm, 0, 255, Core.NORM_MINMAX);
                    input.copyTo(output);
                    for (int j = 0; j < dstNorm.rows(); j++) {
                        for (int i = 0; i < dstNorm.cols(); i++) {
                            if (dstNorm.get(j, i)[0] > 200) {
                                Imgproc.circle(output, new org.opencv.core.Point(i, j), 5, new Scalar(0, 0, 255), 2);
                            }
                        }
                    }
                    grayHarris.release();
                    dst.release();
                    dstNorm.release();
                    break;

                case "ShiTomasi":
                    // Shi-Tomasi corner detection
                    Mat grayST = new Mat();
                    Imgproc.cvtColor(input, grayST, Imgproc.COLOR_BGR2GRAY);
                    org.opencv.core.MatOfPoint corners = new org.opencv.core.MatOfPoint();
                    Imgproc.goodFeaturesToTrack(grayST, corners, 100, 0.01, 10);
                    input.copyTo(output);
                    org.opencv.core.Point[] cornerArray = corners.toArray();
                    for (org.opencv.core.Point pt : cornerArray) {
                        Imgproc.circle(output, pt, 5, new Scalar(0, 255, 0), -1);
                    }
                    grayST.release();
                    break;

                case "Histogram":
                    // Draw histogram overlay
                    int histHeight = input.rows();
                    int histWidth = input.cols();
                    Mat hist = new Mat();
                    java.util.List<Mat> bgr = new java.util.ArrayList<>();
                    Core.split(input, bgr);
                    input.copyTo(output);
                    Scalar[] colors = {new Scalar(255, 0, 0), new Scalar(0, 255, 0), new Scalar(0, 0, 255)};
                    for (int ch = 0; ch < Math.min(3, bgr.size()); ch++) {
                        Imgproc.calcHist(java.util.Collections.singletonList(bgr.get(ch)),
                            new org.opencv.core.MatOfInt(0), new Mat(), hist,
                            new org.opencv.core.MatOfInt(256),
                            new org.opencv.core.MatOfFloat(0, 256));
                        Core.normalize(hist, hist, 0, histHeight * 0.4, Core.NORM_MINMAX);
                        int binW = Math.max(1, histWidth / 256);
                        for (int i = 1; i < 256; i++) {
                            Imgproc.line(output,
                                new org.opencv.core.Point(binW * (i - 1), histHeight - hist.get(i - 1, 0)[0]),
                                new org.opencv.core.Point(binW * i, histHeight - hist.get(i, 0)[0]),
                                colors[ch], 2);
                        }
                    }
                    for (Mat m : bgr) m.release();
                    hist.release();
                    break;

                // ===== BitPlanes nodes =====
                case "BitPlanesGrayscale":
                    Mat grayBP = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayBP, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayBP);
                    }
                    // Reconstruct from all 8 bit planes (default behavior)
                    Mat resultBP = Mat.zeros(grayBP.rows(), grayBP.cols(), org.opencv.core.CvType.CV_32F);
                    byte[] grayData = new byte[grayBP.rows() * grayBP.cols()];
                    grayBP.get(0, 0, grayData);
                    float[] resultData = new float[grayBP.rows() * grayBP.cols()];
                    for (int j = 0; j < grayData.length; j++) {
                        resultData[j] = (grayData[j] & 0xFF);
                    }
                    resultBP.put(0, 0, resultData);
                    resultBP.convertTo(output, org.opencv.core.CvType.CV_8U);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayBP.release();
                    resultBP.release();
                    break;

                case "BitPlanesColor":
                    // Color bit plane processing - all 8 bits enabled with gain 1.0 (default)
                    // This reconstructs the original image with ability to enable/disable bits
                    output = applyColorBitPlanes(input);
                    break;

                // ===== Content/Drawing nodes =====
                case "Rectangle":
                    input.copyTo(output);
                    Imgproc.rectangle(output, new org.opencv.core.Point(50, 50),
                        new org.opencv.core.Point(200, 150), new Scalar(0, 255, 0), 2);
                    break;

                case "Circle":
                    input.copyTo(output);
                    Imgproc.circle(output, new org.opencv.core.Point(100, 100), 50,
                        new Scalar(0, 255, 0), 2);
                    break;

                case "Ellipse":
                    input.copyTo(output);
                    Imgproc.ellipse(output, new org.opencv.core.Point(100, 100),
                        new org.opencv.core.Size(100, 50), 0, 0, 360, new Scalar(0, 255, 0), 2);
                    break;

                case "Line":
                    input.copyTo(output);
                    Imgproc.line(output, new org.opencv.core.Point(50, 50),
                        new org.opencv.core.Point(200, 150), new Scalar(0, 255, 0), 2);
                    break;

                case "Arrow":
                    input.copyTo(output);
                    Imgproc.arrowedLine(output, new org.opencv.core.Point(50, 50),
                        new org.opencv.core.Point(200, 150), new Scalar(0, 255, 0), 2);
                    break;

                case "Text":
                    input.copyTo(output);
                    Imgproc.putText(output, "Hello", new org.opencv.core.Point(50, 100),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    break;

                // ===== Filter nodes =====
                case "Filter2D": {
                    // Get kernel size and values from properties
                    int kSize = 3;
                    int[] kValues = null;
                    Object ksObj = node.properties.get("kernelSize");
                    if (ksObj instanceof Number) {
                        kSize = ((Number) ksObj).intValue();
                    }
                    Object kvObj = node.properties.get("kernelValues");
                    if (kvObj instanceof int[]) {
                        kValues = (int[]) kvObj;
                    } else if (kvObj instanceof double[]) {
                        double[] darr = (double[]) kvObj;
                        kValues = new int[darr.length];
                        for (int ki = 0; ki < darr.length; ki++) {
                            kValues[ki] = (int) darr[ki];
                        }
                    }

                    // Build the kernel matrix
                    Mat kernel2D = new Mat(kSize, kSize, org.opencv.core.CvType.CV_32F);
                    if (kValues != null && kValues.length == kSize * kSize) {
                        for (int ki = 0; ki < kSize; ki++) {
                            for (int kj = 0; kj < kSize; kj++) {
                                kernel2D.put(ki, kj, (float) kValues[ki * kSize + kj]);
                            }
                        }
                    } else {
                        // Default identity kernel
                        kernel2D.setTo(new Scalar(0));
                        kernel2D.put(kSize / 2, kSize / 2, 1.0f);
                    }
                    Imgproc.filter2D(input, output, -1, kernel2D);
                    kernel2D.release();
                    break;
                }

                // ===== Transform nodes =====
                case "Crop":
                    // Default crop: 100x100 from top-left
                    int cropW = Math.min(100, input.cols());
                    int cropH = Math.min(100, input.rows());
                    org.opencv.core.Rect roi = new org.opencv.core.Rect(0, 0, cropW, cropH);
                    Mat submat = new Mat(input, roi);
                    submat.copyTo(output);
                    submat.release();
                    break;

                case "WarpAffine":
                    // Default: no transformation (identity)
                    double cx = input.cols() / 2.0;
                    double cy = input.rows() / 2.0;
                    Mat M = Imgproc.getRotationMatrix2D(new org.opencv.core.Point(cx, cy), 0, 1.0);
                    Imgproc.warpAffine(input, output, M, new org.opencv.core.Size(input.cols(), input.rows()));
                    M.release();
                    break;

                // ===== Detection nodes =====
                case "BlobDetector":
                    Mat grayBlob = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayBlob, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayBlob);
                    }
                    org.opencv.features2d.SimpleBlobDetector detector = org.opencv.features2d.SimpleBlobDetector.create();
                    org.opencv.core.MatOfKeyPoint keypoints = new org.opencv.core.MatOfKeyPoint();
                    detector.detect(grayBlob, keypoints);
                    input.copyTo(output);
                    org.opencv.features2d.Features2d.drawKeypoints(output, keypoints, output,
                        new Scalar(255, 0, 0), org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    grayBlob.release();
                    break;

                case "ConnectedComponents":
                    Mat grayCC = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayCC, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayCC);
                    }
                    Mat binaryCC = new Mat();
                    Imgproc.threshold(grayCC, binaryCC, 127, 255, Imgproc.THRESH_BINARY);
                    Mat labels = new Mat();
                    Mat stats = new Mat();
                    Mat centroids = new Mat();
                    int numLabels = Imgproc.connectedComponentsWithStats(binaryCC, labels, stats, centroids, 8, org.opencv.core.CvType.CV_32S);
                    // Create colored output
                    java.util.Random rand = new java.util.Random(42);
                    output = new Mat(input.rows(), input.cols(), org.opencv.core.CvType.CV_8UC3);
                    for (int row = 0; row < labels.rows(); row++) {
                        for (int col = 0; col < labels.cols(); col++) {
                            int label = (int) labels.get(row, col)[0];
                            if (label == 0) {
                                output.put(row, col, 0, 0, 0);
                            } else {
                                rand.setSeed(label * 42);
                                output.put(row, col, rand.nextInt(256), rand.nextInt(256), rand.nextInt(256));
                            }
                        }
                    }
                    grayCC.release();
                    binaryCC.release();
                    labels.release();
                    stats.release();
                    centroids.release();
                    break;

                case "SIFTFeatures":
                    Mat graySIFT = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, graySIFT, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(graySIFT);
                    }
                    org.opencv.features2d.SIFT sift = org.opencv.features2d.SIFT.create(500);
                    org.opencv.core.MatOfKeyPoint siftKp = new org.opencv.core.MatOfKeyPoint();
                    sift.detect(graySIFT, siftKp);
                    input.copyTo(output);
                    org.opencv.features2d.Features2d.drawKeypoints(output, siftKp, output,
                        new Scalar(0, 255, 0), org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    graySIFT.release();
                    break;

                case "ORBFeatures":
                    Mat grayORB = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayORB, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayORB);
                    }
                    org.opencv.features2d.ORB orb = org.opencv.features2d.ORB.create(500);
                    org.opencv.core.MatOfKeyPoint orbKp = new org.opencv.core.MatOfKeyPoint();
                    orb.detect(grayORB, orbKp);
                    input.copyTo(output);
                    org.opencv.features2d.Features2d.drawKeypoints(output, orbKp, output,
                        new Scalar(0, 255, 0), org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    grayORB.release();
                    break;

                case "MatchTemplate":
                    // MatchTemplate requires dual input - pass through for now
                    input.copyTo(output);
                    break;

                // ===== Dual Input nodes (require second input) =====
                case "AddClamp":
                case "SubtractClamp":
                case "AddWeighted":
                case "BitwiseAnd":
                case "BitwiseOr":
                case "BitwiseXor":
                    output = processDualInputNode(node, type, input, nodeOutputs);
                    break;

                // ===== FFT Filter nodes =====
                case "FFTLowPass":
                    // Default radius=100, smoothness=0 (hard edge)
                    output = applyFFTLowPass(input, 100, 0);
                    break;

                case "FFTHighPass":
                    // Default radius=30, smoothness=0 (hard edge)
                    output = applyFFTHighPass(input, 30, 0);
                    break;

                case "FFTLowPass4":
                case "FFTHighPass4":
                    // Multi-output FFT nodes - apply with default settings for primary output
                    if (type.equals("FFTLowPass4")) {
                        output = applyFFTLowPass(input, 100, 0);
                    } else {
                        output = applyFFTHighPass(input, 30, 0);
                    }
                    break;

                case "Container":
                    // Execute the container's internal subdiagram
                    Mat containerResult = executeContainer(node, input);
                    if (containerResult != null) {
                        containerResult.copyTo(output);
                        containerResult.release();
                    } else {
                        input.copyTo(output);
                    }
                    break;

                default:
                    // Pass through for unimplemented nodes
                    input.copyTo(output);
                    break;
            }
        } catch (Exception e) {
            System.err.println("Error processing " + type + ": " + e.getMessage());
            input.copyTo(output);
        }

        // Note: output counters are now incremented in getInputForNode when data actually
        // flows through a connection, so we can track per-output-port statistics for
        // multi-output nodes like Clone.

        input.release();
        return output;
    }

    public void setTargetFps(double fps) {
        this.targetFps = fps;
    }

    /**
     * Execute a container's internal subdiagram.
     * Passes the input through the subdiagram and returns the output.
     */
    private Mat executeContainer(FXNode containerNode, Mat input) {
        if (containerNode.innerNodes == null || containerNode.innerNodes.isEmpty()) {
            return input.clone();  // No subdiagram, pass through
        }

        // Find ContainerInput and ContainerOutput nodes
        FXNode inputNode = null;
        FXNode outputNode = null;
        for (FXNode node : containerNode.innerNodes) {
            if ("ContainerInput".equals(node.nodeType)) {
                inputNode = node;
            } else if ("ContainerOutput".equals(node.nodeType)) {
                outputNode = node;
            }
        }

        if (inputNode == null || outputNode == null) {
            System.out.println("  -> Missing boundary nodes (input=" + inputNode + ", output=" + outputNode + ")");
            return input.clone();  // Missing boundary nodes, pass through
        }

        System.out.println("  -> Found boundary nodes, executing " + containerNode.innerNodes.size() + " inner nodes");

        // Build execution order for inner nodes
        List<FXNode> executionOrder = buildInnerExecutionOrder(containerNode.innerNodes, containerNode.innerConnections);

        // Map to store outputs from each inner node
        Map<Integer, Mat> innerOutputs = new HashMap<>();

        // Seed the ContainerInput with the incoming frame
        Mat inputClone = input.clone();
        innerOutputs.put(inputNode.id, inputClone);
        inputNode.inputCount++;
        inputNode.outputCount1++;

        // Update thumbnail for ContainerInput node
        if (onNodeOutput != null) {
            Mat inputCopy = inputClone.clone();
            final FXNode capturedInputNode = inputNode;
            System.out.println("  executeContainer: updating thumbnail for ContainerInput id=" + capturedInputNode.id + " label=" + capturedInputNode.label);
            Platform.runLater(() -> {
                onNodeOutput.accept(capturedInputNode, inputCopy);
            });
        }

        // Process inner nodes in order
        for (FXNode innerNode : executionOrder) {
            if (innerNode == inputNode) {
                continue;  // Already seeded
            }

            // Get input for this node from its connected source
            Mat nodeInput = getInnerNodeInput(innerNode, containerNode.innerConnections, innerOutputs);

            if (nodeInput == null) {
                continue;  // No input available
            }

            innerNode.inputCount++;

            if ("ContainerOutput".equals(innerNode.nodeType)) {
                // This is the output node - its input becomes the container output
                innerOutputs.put(innerNode.id, nodeInput);
                innerNode.inputCount++;

                // Update thumbnail for output node
                if (onNodeOutput != null) {
                    Mat outputCopy = nodeInput.clone();
                    Platform.runLater(() -> {
                        onNodeOutput.accept(innerNode, outputCopy);
                    });
                }
            } else {
                // Process the node using the same logic as processNode
                Mat nodeOutput = processInnerNode(innerNode, nodeInput, containerNode.innerConnections, innerOutputs);
                if (nodeOutput != null) {
                    innerOutputs.put(innerNode.id, nodeOutput);

                    // Update thumbnail for this inner node
                    if (onNodeOutput != null) {
                        Mat outputCopy = nodeOutput.clone();
                        Platform.runLater(() -> {
                            onNodeOutput.accept(innerNode, outputCopy);
                        });
                    }
                }
                nodeInput.release();
            }
        }

        // Get the output from ContainerOutput
        Mat result = innerOutputs.get(outputNode.id);
        if (result != null) {
            result = result.clone();  // Clone so we can safely release all inner outputs
        }

        // Clean up inner outputs
        for (Mat mat : innerOutputs.values()) {
            if (mat != null && !mat.empty()) {
                mat.release();
            }
        }

        return result;
    }

    /**
     * Build execution order for inner nodes using topological sort.
     */
    private List<FXNode> buildInnerExecutionOrder(List<FXNode> innerNodes, List<FXConnection> innerConnections) {
        List<FXNode> order = new ArrayList<>();
        Set<Integer> visited = new HashSet<>();

        // Start from source nodes (ContainerInput or nodes with no inputs)
        for (FXNode node : innerNodes) {
            if ("ContainerInput".equals(node.nodeType) || !node.hasInput) {
                visitInnerNode(node, innerNodes, innerConnections, visited, order);
            }
        }

        // Add any remaining unvisited nodes
        for (FXNode node : innerNodes) {
            if (!visited.contains(node.id)) {
                visitInnerNode(node, innerNodes, innerConnections, visited, order);
            }
        }

        return order;
    }

    private void visitInnerNode(FXNode node, List<FXNode> allNodes, List<FXConnection> connections,
                                 Set<Integer> visited, List<FXNode> order) {
        if (visited.contains(node.id)) return;
        visited.add(node.id);

        // First visit all nodes this one depends on
        for (FXConnection conn : connections) {
            if (conn.target == node && conn.source != null && !visited.contains(conn.source.id)) {
                visitInnerNode(conn.source, allNodes, connections, visited, order);
            }
        }

        order.add(node);

        // Then visit nodes that depend on this one
        for (FXConnection conn : connections) {
            if (conn.source == node && conn.target != null && !visited.contains(conn.target.id)) {
                visitInnerNode(conn.target, allNodes, connections, visited, order);
            }
        }
    }

    /**
     * Get input for an inner node from its connected source (input index 0).
     * Also updates connection statistics when data flows through.
     */
    private Mat getInnerNodeInput(FXNode node, List<FXConnection> connections, Map<Integer, Mat> outputs) {
        return getInnerNodeInputByIndex(node, connections, outputs, 0);
    }

    /**
     * Get input for an inner node from a specific input index.
     * Used for dual-input nodes that have multiple input connections.
     * Also updates connection statistics when data flows through.
     */
    private Mat getInnerNodeInputByIndex(FXNode node, List<FXConnection> connections, Map<Integer, Mat> outputs, int inputIndex) {
        for (FXConnection conn : connections) {
            if (conn.target == node && conn.targetInputIndex == inputIndex && conn.source != null) {
                Mat sourceOutput = outputs.get(conn.source.id);
                if (sourceOutput != null) {
                    // Update connection statistics
                    conn.totalFrames++;
                    // Update source node's output counter based on which output port this connection uses
                    switch (conn.sourceOutputIndex) {
                        case 0: conn.source.outputCount1++; break;
                        case 1: conn.source.outputCount2++; break;
                        case 2: conn.source.outputCount3++; break;
                        case 3: conn.source.outputCount4++; break;
                    }
                    return sourceOutput.clone();
                }
            }
        }
        return null;
    }

    /**
     * Process an inner node (simplified version - delegates to main processNode logic).
     * @param node The inner node to process
     * @param input The input Mat (from input index 0)
     * @param connections The connections list (for dual-input nodes to get second input)
     * @param outputs The outputs map (for dual-input nodes to get second input)
     */
    private Mat processInnerNode(FXNode node, Mat input, List<FXConnection> connections, Map<Integer, Mat> outputs) {
        String type = node.nodeType;
        Mat output = new Mat();

        try {
            switch (type) {
                case "Grayscale":
                    Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    break;

                case "Invert":
                    Core.bitwise_not(input, output);
                    break;

                case "GaussianBlur":
                    Imgproc.GaussianBlur(input, output, new org.opencv.core.Size(15, 15), 0);
                    break;

                case "MedianBlur":
                    Imgproc.medianBlur(input, output, 15);
                    break;

                case "CannyEdge":
                    Mat gray = new Mat();
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.Canny(gray, output, 50, 150);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    gray.release();
                    break;

                case "Sobel":
                    Mat grayS = new Mat();
                    Imgproc.cvtColor(input, grayS, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.Sobel(grayS, output, org.opencv.core.CvType.CV_8U, 1, 1);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayS.release();
                    break;

                case "Laplacian":
                    Mat grayL = new Mat();
                    Imgproc.cvtColor(input, grayL, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.Laplacian(grayL, output, org.opencv.core.CvType.CV_8U);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayL.release();
                    break;

                case "Threshold":
                    Mat grayT = new Mat();
                    Imgproc.cvtColor(input, grayT, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.threshold(grayT, output, 127, 255, Imgproc.THRESH_BINARY);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayT.release();
                    break;

                case "Erode":
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.erode(input, output, kernel);
                    kernel.release();
                    break;

                case "Dilate":
                    Mat kernelD = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.dilate(input, output, kernelD);
                    kernelD.release();
                    break;

                case "BilateralFilter":
                    Imgproc.bilateralFilter(input, output, 9, 75, 75);
                    break;

                case "BoxBlur":
                    Imgproc.blur(input, output, new org.opencv.core.Size(15, 15));
                    break;

                case "AdaptiveThreshold":
                    Mat grayAT = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayAT, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayAT);
                    }
                    Mat threshAT = new Mat();
                    Imgproc.adaptiveThreshold(grayAT, threshAT, 255,
                        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);
                    Imgproc.cvtColor(threshAT, output, Imgproc.COLOR_GRAY2BGR);
                    grayAT.release();
                    threshAT.release();
                    break;

                case "Gain":
                    input.convertTo(output, -1, 1.5, 0);
                    break;

                case "CLAHE":
                    Mat grayCL = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayCL, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayCL);
                    }
                    org.opencv.imgproc.CLAHE clahe = Imgproc.createCLAHE(2.0, new org.opencv.core.Size(8, 8));
                    Mat claheDst = new Mat();
                    clahe.apply(grayCL, claheDst);
                    Imgproc.cvtColor(claheDst, output, Imgproc.COLOR_GRAY2BGR);
                    grayCL.release();
                    claheDst.release();
                    break;

                case "Scharr":
                    Mat graySch = new Mat();
                    Imgproc.cvtColor(input, graySch, Imgproc.COLOR_BGR2GRAY);
                    Mat scharrX = new Mat();
                    Mat scharrY = new Mat();
                    Imgproc.Scharr(graySch, scharrX, org.opencv.core.CvType.CV_16S, 1, 0);
                    Imgproc.Scharr(graySch, scharrY, org.opencv.core.CvType.CV_16S, 0, 1);
                    Core.convertScaleAbs(scharrX, scharrX);
                    Core.convertScaleAbs(scharrY, scharrY);
                    Core.addWeighted(scharrX, 0.5, scharrY, 0.5, 0, output);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    graySch.release();
                    scharrX.release();
                    scharrY.release();
                    break;

                case "MorphOpen":
                    Mat kernelOpen = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_OPEN, kernelOpen);
                    kernelOpen.release();
                    break;

                case "MorphClose":
                    Mat kernelClose = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, kernelClose);
                    kernelClose.release();
                    break;

                case "MorphologyEx":
                    Mat kernelMorph = Imgproc.getStructuringElement(Imgproc.MORPH_RECT,
                        new org.opencv.core.Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_GRADIENT, kernelMorph);
                    kernelMorph.release();
                    break;

                case "BitwiseNot":
                    Core.bitwise_not(input, output);
                    break;

                case "ColorInRange":
                    Mat hsv = new Mat();
                    Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
                    Mat mask = new Mat();
                    Core.inRange(hsv, new Scalar(100, 50, 50), new Scalar(130, 255, 255), mask);
                    Imgproc.cvtColor(mask, output, Imgproc.COLOR_GRAY2BGR);
                    hsv.release();
                    mask.release();
                    break;

                case "MeanShift":
                    Imgproc.pyrMeanShiftFiltering(input, output, 21, 51);
                    break;

                case "Clone":
                case "Monitor":
                    input.copyTo(output);
                    break;

                case "Contours":
                    Mat grayCont = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayCont, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayCont);
                    }
                    Mat binary = new Mat();
                    Imgproc.threshold(grayCont, binary, 127, 255, Imgproc.THRESH_BINARY);
                    java.util.List<org.opencv.core.MatOfPoint> contours = new java.util.ArrayList<>();
                    Mat hierarchy = new Mat();
                    Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
                    input.copyTo(output);
                    Imgproc.drawContours(output, contours, -1, new Scalar(0, 255, 0), 2);
                    grayCont.release();
                    binary.release();
                    hierarchy.release();
                    break;

                case "HoughCircles":
                    Mat grayHC = new Mat();
                    Imgproc.cvtColor(input, grayHC, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.GaussianBlur(grayHC, grayHC, new org.opencv.core.Size(9, 9), 2);
                    Mat circles = new Mat();
                    Imgproc.HoughCircles(grayHC, circles, Imgproc.HOUGH_GRADIENT, 1, 50, 100, 30, 10, 100);
                    input.copyTo(output);
                    for (int i = 0; i < circles.cols(); i++) {
                        double[] c = circles.get(0, i);
                        org.opencv.core.Point center = new org.opencv.core.Point(c[0], c[1]);
                        int radius = (int) Math.round(c[2]);
                        Imgproc.circle(output, center, radius, new Scalar(0, 255, 0), 2);
                        Imgproc.circle(output, center, 3, new Scalar(0, 0, 255), -1);
                    }
                    grayHC.release();
                    circles.release();
                    break;

                case "HoughLines":
                    Mat grayHL = new Mat();
                    Imgproc.cvtColor(input, grayHL, Imgproc.COLOR_BGR2GRAY);
                    Mat edges = new Mat();
                    Imgproc.Canny(grayHL, edges, 50, 150);
                    Mat lines = new Mat();
                    Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180, 50, 50, 10);
                    input.copyTo(output);
                    for (int i = 0; i < lines.rows(); i++) {
                        double[] l = lines.get(i, 0);
                        Imgproc.line(output, new org.opencv.core.Point(l[0], l[1]),
                            new org.opencv.core.Point(l[2], l[3]), new Scalar(0, 0, 255), 2);
                    }
                    grayHL.release();
                    edges.release();
                    lines.release();
                    break;

                case "HarrisCorners":
                    Mat grayHarris = new Mat();
                    Imgproc.cvtColor(input, grayHarris, Imgproc.COLOR_BGR2GRAY);
                    Mat dst = new Mat();
                    Imgproc.cornerHarris(grayHarris, dst, 2, 3, 0.04);
                    Mat dstNorm = new Mat();
                    Core.normalize(dst, dstNorm, 0, 255, Core.NORM_MINMAX);
                    input.copyTo(output);
                    for (int j = 0; j < dstNorm.rows(); j++) {
                        for (int ii = 0; ii < dstNorm.cols(); ii++) {
                            if (dstNorm.get(j, ii)[0] > 200) {
                                Imgproc.circle(output, new org.opencv.core.Point(ii, j), 5, new Scalar(0, 0, 255), 2);
                            }
                        }
                    }
                    grayHarris.release();
                    dst.release();
                    dstNorm.release();
                    break;

                case "ShiTomasi":
                    Mat grayST = new Mat();
                    Imgproc.cvtColor(input, grayST, Imgproc.COLOR_BGR2GRAY);
                    org.opencv.core.MatOfPoint corners = new org.opencv.core.MatOfPoint();
                    Imgproc.goodFeaturesToTrack(grayST, corners, 100, 0.01, 10);
                    input.copyTo(output);
                    org.opencv.core.Point[] cornerArray = corners.toArray();
                    for (org.opencv.core.Point pt : cornerArray) {
                        Imgproc.circle(output, pt, 5, new Scalar(0, 255, 0), -1);
                    }
                    grayST.release();
                    break;

                case "Histogram":
                    int histHeight = input.rows();
                    int histWidth = input.cols();
                    Mat hist = new Mat();
                    java.util.List<Mat> bgr = new java.util.ArrayList<>();
                    Core.split(input, bgr);
                    input.copyTo(output);
                    Scalar[] colors = {new Scalar(255, 0, 0), new Scalar(0, 255, 0), new Scalar(0, 0, 255)};
                    for (int ch = 0; ch < Math.min(3, bgr.size()); ch++) {
                        Imgproc.calcHist(java.util.Collections.singletonList(bgr.get(ch)),
                            new org.opencv.core.MatOfInt(0), new Mat(), hist,
                            new org.opencv.core.MatOfInt(256),
                            new org.opencv.core.MatOfFloat(0, 256));
                        Core.normalize(hist, hist, 0, histHeight * 0.4, Core.NORM_MINMAX);
                        int binW = Math.max(1, histWidth / 256);
                        for (int ii = 1; ii < 256; ii++) {
                            Imgproc.line(output,
                                new org.opencv.core.Point(binW * (ii - 1), histHeight - hist.get(ii - 1, 0)[0]),
                                new org.opencv.core.Point(binW * ii, histHeight - hist.get(ii, 0)[0]),
                                colors[ch], 2);
                        }
                    }
                    for (Mat m : bgr) m.release();
                    hist.release();
                    break;

                case "Container":
                    // Nested container - recursively execute
                    Mat nestedResult = executeContainer(node, input);
                    if (nestedResult != null) {
                        nestedResult.copyTo(output);
                        nestedResult.release();
                    } else {
                        input.copyTo(output);
                    }
                    break;

                // ===== BitPlanes nodes =====
                case "BitPlanesGrayscale":
                    Mat grayBP = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayBP, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayBP);
                    }
                    Mat resultBP = Mat.zeros(grayBP.rows(), grayBP.cols(), org.opencv.core.CvType.CV_32F);
                    byte[] grayData = new byte[grayBP.rows() * grayBP.cols()];
                    grayBP.get(0, 0, grayData);
                    float[] resultData = new float[grayBP.rows() * grayBP.cols()];
                    for (int j = 0; j < grayData.length; j++) {
                        resultData[j] = (grayData[j] & 0xFF);
                    }
                    resultBP.put(0, 0, resultData);
                    resultBP.convertTo(output, org.opencv.core.CvType.CV_8U);
                    Imgproc.cvtColor(output, output, Imgproc.COLOR_GRAY2BGR);
                    grayBP.release();
                    resultBP.release();
                    break;

                case "BitPlanesColor":
                    output = applyColorBitPlanes(input);
                    break;

                // ===== Content/Drawing nodes =====
                case "Rectangle":
                    input.copyTo(output);
                    Imgproc.rectangle(output, new org.opencv.core.Point(50, 50),
                        new org.opencv.core.Point(200, 150), new Scalar(0, 255, 0), 2);
                    break;

                case "Circle":
                    input.copyTo(output);
                    Imgproc.circle(output, new org.opencv.core.Point(100, 100), 50,
                        new Scalar(0, 255, 0), 2);
                    break;

                case "Ellipse":
                    input.copyTo(output);
                    Imgproc.ellipse(output, new org.opencv.core.Point(100, 100),
                        new org.opencv.core.Size(100, 50), 0, 0, 360, new Scalar(0, 255, 0), 2);
                    break;

                case "Line":
                    input.copyTo(output);
                    Imgproc.line(output, new org.opencv.core.Point(50, 50),
                        new org.opencv.core.Point(200, 150), new Scalar(0, 255, 0), 2);
                    break;

                case "Arrow":
                    input.copyTo(output);
                    Imgproc.arrowedLine(output, new org.opencv.core.Point(50, 50),
                        new org.opencv.core.Point(200, 150), new Scalar(0, 255, 0), 2);
                    break;

                case "Text":
                    input.copyTo(output);
                    Imgproc.putText(output, "Hello", new org.opencv.core.Point(50, 100),
                        Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, new Scalar(0, 255, 0), 2);
                    break;

                // ===== Filter nodes =====
                case "Filter2D": {
                    // Get kernel size and values from properties
                    int kSize = 3;
                    int[] kValues = null;
                    Object ksObj = node.properties.get("kernelSize");
                    if (ksObj instanceof Number) {
                        kSize = ((Number) ksObj).intValue();
                    }
                    if (kSize < 3) kSize = 3;
                    if (kSize > 9) kSize = 9;
                    if (kSize % 2 == 0) kSize++;

                    Object kvObj = node.properties.get("kernelValues");
                    if (kvObj instanceof int[]) {
                        kValues = (int[]) kvObj;
                    } else if (kvObj instanceof java.util.List) {
                        java.util.List<?> list = (java.util.List<?>) kvObj;
                        kValues = new int[list.size()];
                        for (int ki = 0; ki < list.size(); ki++) {
                            Object v = list.get(ki);
                            kValues[ki] = v instanceof Number ? ((Number) v).intValue() : 0;
                        }
                    }

                    // Create kernel mat
                    Mat kernel2D = new Mat(kSize, kSize, org.opencv.core.CvType.CV_32F);
                    if (kValues != null && kValues.length == kSize * kSize) {
                        for (int ki = 0; ki < kSize; ki++) {
                            for (int kj = 0; kj < kSize; kj++) {
                                kernel2D.put(ki, kj, (float) kValues[ki * kSize + kj]);
                            }
                        }
                    } else {
                        // Default identity kernel
                        for (int ki = 0; ki < kSize; ki++) {
                            for (int kj = 0; kj < kSize; kj++) {
                                kernel2D.put(ki, kj, (ki == kSize/2 && kj == kSize/2) ? 1.0f : 0.0f);
                            }
                        }
                    }
                    Imgproc.filter2D(input, output, -1, kernel2D);
                    kernel2D.release();
                    break;
                }

                // ===== Transform nodes =====
                case "Crop":
                    int cropW = Math.min(100, input.cols());
                    int cropH = Math.min(100, input.rows());
                    org.opencv.core.Rect roi = new org.opencv.core.Rect(0, 0, cropW, cropH);
                    Mat submat = new Mat(input, roi);
                    submat.copyTo(output);
                    submat.release();
                    break;

                case "WarpAffine":
                    double cxW = input.cols() / 2.0;
                    double cyW = input.rows() / 2.0;
                    Mat M = Imgproc.getRotationMatrix2D(new org.opencv.core.Point(cxW, cyW), 0, 1.0);
                    Imgproc.warpAffine(input, output, M, new org.opencv.core.Size(input.cols(), input.rows()));
                    M.release();
                    break;

                // ===== Detection nodes =====
                case "BlobDetector":
                    Mat grayBlob = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayBlob, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayBlob);
                    }
                    org.opencv.features2d.SimpleBlobDetector detector = org.opencv.features2d.SimpleBlobDetector.create();
                    org.opencv.core.MatOfKeyPoint keypoints = new org.opencv.core.MatOfKeyPoint();
                    detector.detect(grayBlob, keypoints);
                    input.copyTo(output);
                    org.opencv.features2d.Features2d.drawKeypoints(output, keypoints, output,
                        new Scalar(255, 0, 0), org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    grayBlob.release();
                    break;

                case "ConnectedComponents":
                    Mat grayCC = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayCC, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayCC);
                    }
                    Mat binaryCC = new Mat();
                    Imgproc.threshold(grayCC, binaryCC, 127, 255, Imgproc.THRESH_BINARY);
                    Mat labelsCC = new Mat();
                    Mat statsCC = new Mat();
                    Mat centroidsCC = new Mat();
                    int numLabelsCC = Imgproc.connectedComponentsWithStats(binaryCC, labelsCC, statsCC, centroidsCC, 8, org.opencv.core.CvType.CV_32S);
                    java.util.Random randCC = new java.util.Random(42);
                    output = new Mat(input.rows(), input.cols(), org.opencv.core.CvType.CV_8UC3);
                    for (int row = 0; row < labelsCC.rows(); row++) {
                        for (int col = 0; col < labelsCC.cols(); col++) {
                            int label = (int) labelsCC.get(row, col)[0];
                            if (label == 0) {
                                output.put(row, col, 0, 0, 0);
                            } else {
                                randCC.setSeed(label * 42);
                                output.put(row, col, randCC.nextInt(256), randCC.nextInt(256), randCC.nextInt(256));
                            }
                        }
                    }
                    grayCC.release();
                    binaryCC.release();
                    labelsCC.release();
                    statsCC.release();
                    centroidsCC.release();
                    break;

                case "SIFTFeatures":
                    Mat graySIFT = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, graySIFT, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(graySIFT);
                    }
                    org.opencv.features2d.SIFT sift = org.opencv.features2d.SIFT.create(500);
                    org.opencv.core.MatOfKeyPoint siftKp = new org.opencv.core.MatOfKeyPoint();
                    sift.detect(graySIFT, siftKp);
                    input.copyTo(output);
                    org.opencv.features2d.Features2d.drawKeypoints(output, siftKp, output,
                        new Scalar(0, 255, 0), org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    graySIFT.release();
                    break;

                case "ORBFeatures":
                    Mat grayORB = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, grayORB, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        input.copyTo(grayORB);
                    }
                    org.opencv.features2d.ORB orb = org.opencv.features2d.ORB.create(500);
                    org.opencv.core.MatOfKeyPoint orbKp = new org.opencv.core.MatOfKeyPoint();
                    orb.detect(grayORB, orbKp);
                    input.copyTo(output);
                    org.opencv.features2d.Features2d.drawKeypoints(output, orbKp, output,
                        new Scalar(0, 255, 0), org.opencv.features2d.Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    grayORB.release();
                    break;

                case "MatchTemplate":
                    input.copyTo(output);
                    break;

                // ===== Dual Input nodes (require second input) =====
                case "AddClamp":
                case "SubtractClamp":
                case "AddWeighted":
                case "BitwiseAnd":
                case "BitwiseOr":
                case "BitwiseXor":
                    output = processInnerDualInputNode(node, type, input, connections, outputs);
                    break;

                // ===== FFT Filter nodes =====
                case "FFTLowPass":
                    output = applyFFTLowPass(input, 100, 0);
                    break;

                case "FFTHighPass":
                    output = applyFFTHighPass(input, 30, 0);
                    break;

                case "FFTLowPass4":
                case "FFTHighPass4":
                    if (type.equals("FFTLowPass4")) {
                        output = applyFFTLowPass(input, 100, 0);
                    } else {
                        output = applyFFTHighPass(input, 30, 0);
                    }
                    break;

                default:
                    // Pass through for unimplemented nodes
                    input.copyTo(output);
                    break;
            }
        } catch (Exception e) {
            System.err.println("Error processing inner node " + type + ": " + e.getMessage());
            input.copyTo(output);
        }

        // Note: output counters are now incremented in getInnerNodeInput when data actually
        // flows through a connection, so we can track per-output-port statistics for
        // multi-output nodes like Clone.

        return output;
    }

    // ===== FFT Filter Helper Methods =====

    private static final double BUTTERWORTH_ORDER_MAX = 10.0;
    private static final double BUTTERWORTH_ORDER_MIN = 0.5;
    private static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    private static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    // ===== Dual Input Processing =====

    /**
     * Process dual-input nodes (AddClamp, SubtractClamp, AddWeighted, BitwiseAnd/Or/Xor).
     * Gets the second input from input index 1 and combines with the first input.
     * Used for main pipeline execution.
     */
    private Mat processDualInputNode(FXNode node, String type, Mat input1, Map<Integer, Mat> nodeOutputs) {
        // Get second input (input index 1)
        Mat input2 = getInputForNodeByIndex(node, nodeOutputs, 1);
        return applyDualInputOperation(type, input1, input2);
    }

    /**
     * Process dual-input nodes inside containers.
     * Gets the second input from inner connections and combines with the first input.
     */
    private Mat processInnerDualInputNode(FXNode node, String type, Mat input1,
                                          List<FXConnection> connections, Map<Integer, Mat> outputs) {
        // Get second input (input index 1) from inner connections
        Mat input2 = getInnerNodeInputByIndex(node, connections, outputs, 1);
        return applyDualInputOperation(type, input1, input2);
    }

    /**
     * Apply the dual-input operation to two inputs.
     * Common logic shared between main and container execution.
     */
    private Mat applyDualInputOperation(String type, Mat input1, Mat input2) {

        // If no second input, return first input as-is
        if (input2 == null) {
            Mat output = new Mat();
            input1.copyTo(output);
            return output;
        }

        try {
            Mat output = new Mat();

            // Resize input2 to match input1 if sizes differ
            Mat resized2 = input2;
            if (input1.width() != input2.width() || input1.height() != input2.height()) {
                resized2 = new Mat();
                Imgproc.resize(input2, resized2, new org.opencv.core.Size(input1.width(), input1.height()));
            }

            // Convert to same type if needed
            Mat converted2 = resized2;
            if (input1.type() != resized2.type()) {
                converted2 = new Mat();
                resized2.convertTo(converted2, input1.type());
            }

            // Apply the operation based on type
            switch (type) {
                case "AddClamp":
                    Core.add(input1, converted2, output);
                    break;
                case "SubtractClamp":
                    Core.subtract(input1, converted2, output);
                    break;
                case "AddWeighted":
                    // Default weights: alpha=0.5, beta=0.5, gamma=0
                    Core.addWeighted(input1, 0.5, converted2, 0.5, 0, output);
                    break;
                case "BitwiseAnd":
                    Core.bitwise_and(input1, converted2, output);
                    break;
                case "BitwiseOr":
                    Core.bitwise_or(input1, converted2, output);
                    break;
                case "BitwiseXor":
                    Core.bitwise_xor(input1, converted2, output);
                    break;
                default:
                    input1.copyTo(output);
                    break;
            }

            // Clean up temporary mats
            if (resized2 != input2) resized2.release();
            if (converted2 != resized2) converted2.release();
            input2.release();

            return output;
        } catch (Exception e) {
            System.err.println("Error in dual-input processing for " + type + ": " + e.getMessage());
            Mat output = new Mat();
            input1.copyTo(output);
            if (input2 != null) input2.release();
            return output;
        }
    }

    // ===== FFT Filter Processing =====

    private static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    private static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    /**
     * Apply FFT low-pass filter to an image.
     * @param input The input image
     * @param radius The cutoff radius (0-200)
     * @param smoothness The filter smoothness (0-100, 0=hard edge, 100=very smooth)
     * @return Filtered image
     */
    private Mat applyFFTLowPass(Mat input, int radius, int smoothness) {
        if (input == null || input.empty()) {
            return input;
        }

        // Split into BGR channels
        List<Mat> channels = new ArrayList<>();
        Core.split(input, channels);

        try {
            List<Mat> filteredChannels = new ArrayList<>();
            try {
                for (Mat channel : channels) {
                    Mat filtered = applyFFTLowPassToChannel(channel, radius, smoothness);
                    filteredChannels.add(filtered);
                }

                Mat output = new Mat();
                Core.merge(filteredChannels, output);
                return output;
            } finally {
                for (Mat m : filteredChannels) {
                    m.release();
                }
            }
        } finally {
            for (Mat m : channels) {
                m.release();
            }
        }
    }

    /**
     * Apply FFT high-pass filter to an image.
     * @param input The input image
     * @param radius The cutoff radius (0-200)
     * @param smoothness The filter smoothness (0-100)
     * @return Filtered image
     */
    private Mat applyFFTHighPass(Mat input, int radius, int smoothness) {
        if (input == null || input.empty()) {
            return input;
        }

        List<Mat> channels = new ArrayList<>();
        Core.split(input, channels);

        try {
            List<Mat> filteredChannels = new ArrayList<>();
            try {
                for (Mat channel : channels) {
                    Mat filtered = applyFFTHighPassToChannel(channel, radius, smoothness);
                    filteredChannels.add(filtered);
                }

                Mat output = new Mat();
                Core.merge(filteredChannels, output);
                return output;
            } finally {
                for (Mat m : filteredChannels) {
                    m.release();
                }
            }
        } finally {
            for (Mat m : channels) {
                m.release();
            }
        }
    }

    private int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    private int getOptimalFFTSize(int n) {
        int pow2 = nextPowerOf2(n);
        int optimal = Core.getOptimalDFTSize(n);
        if (pow2 <= optimal * 1.25) {
            return pow2;
        }
        return optimal % 2 == 0 ? optimal : optimal + 1;
    }

    private Mat applyFFTLowPassToChannel(Mat channel, int radius, int smoothness) {
        int origRows = channel.rows();
        int origCols = channel.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        Mat padded = new Mat();
        Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
            Core.BORDER_CONSTANT, Scalar.all(0));

        Mat floatChannel = new Mat();
        padded.convertTo(floatChannel, org.opencv.core.CvType.CV_32F);
        padded.release();

        Mat complexI = new Mat();
        List<Mat> planes = new ArrayList<>();
        planes.add(floatChannel);
        planes.add(Mat.zeros(floatChannel.size(), org.opencv.core.CvType.CV_32F));
        Core.merge(planes, complexI);
        planes.get(1).release();
        floatChannel.release();

        Core.dft(complexI, complexI);
        fftShift(complexI);

        Mat mask = createLowPassMask(optRows, optCols, radius, smoothness);

        List<Mat> dftPlanes = new ArrayList<>();
        Core.split(complexI, dftPlanes);
        complexI.release();

        Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
        Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));
        mask.release();

        Mat maskedDft = new Mat();
        Core.merge(dftPlanes, maskedDft);
        for (Mat p : dftPlanes) p.release();

        fftShift(maskedDft);
        Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

        List<Mat> idftPlanes = new ArrayList<>();
        Core.split(maskedDft, idftPlanes);
        maskedDft.release();

        Mat magnitude = idftPlanes.get(0);
        idftPlanes.get(1).release();

        Mat cropped = new Mat(magnitude, new org.opencv.core.Rect(0, 0, origCols, origRows));
        Mat result = cropped.clone();
        magnitude.release();

        Core.min(result, new Scalar(255), result);
        Core.max(result, new Scalar(0), result);

        Mat output = new Mat();
        result.convertTo(output, org.opencv.core.CvType.CV_8U);
        result.release();

        return output;
    }

    private Mat applyFFTHighPassToChannel(Mat channel, int radius, int smoothness) {
        int origRows = channel.rows();
        int origCols = channel.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        Mat padded = new Mat();
        Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
            Core.BORDER_CONSTANT, Scalar.all(0));

        Mat floatChannel = new Mat();
        padded.convertTo(floatChannel, org.opencv.core.CvType.CV_32F);
        padded.release();

        Mat complexI = new Mat();
        List<Mat> planes = new ArrayList<>();
        planes.add(floatChannel);
        planes.add(Mat.zeros(floatChannel.size(), org.opencv.core.CvType.CV_32F));
        Core.merge(planes, complexI);
        planes.get(1).release();
        floatChannel.release();

        Core.dft(complexI, complexI);
        fftShift(complexI);

        Mat mask = createHighPassMask(optRows, optCols, radius, smoothness);

        List<Mat> dftPlanes = new ArrayList<>();
        Core.split(complexI, dftPlanes);
        complexI.release();

        Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
        Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));
        mask.release();

        Mat maskedDft = new Mat();
        Core.merge(dftPlanes, maskedDft);
        for (Mat p : dftPlanes) p.release();

        fftShift(maskedDft);
        Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

        List<Mat> idftPlanes = new ArrayList<>();
        Core.split(maskedDft, idftPlanes);
        maskedDft.release();

        Mat magnitude = idftPlanes.get(0);
        idftPlanes.get(1).release();

        Mat cropped = new Mat(magnitude, new org.opencv.core.Rect(0, 0, origCols, origRows));
        Mat result = cropped.clone();
        magnitude.release();

        Core.min(result, new Scalar(255), result);
        Core.max(result, new Scalar(0), result);

        Mat output = new Mat();
        result.convertTo(output, org.opencv.core.CvType.CV_8U);
        result.release();

        return output;
    }

    private void fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        Mat q0 = new Mat(input, new org.opencv.core.Rect(0, 0, cx, cy));
        Mat q1 = new Mat(input, new org.opencv.core.Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(input, new org.opencv.core.Rect(0, cy, cx, cy));
        Mat q3 = new Mat(input, new org.opencv.core.Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        tmp.release();
    }

    private Mat createLowPassMask(int rows, int cols, int radius, int smoothness) {
        Mat mask = new Mat(rows, cols, org.opencv.core.CvType.CV_32F);

        if (radius == 0) {
            mask.setTo(new Scalar(0.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            mask.setTo(new Scalar(0.0));
            Imgproc.circle(mask, new org.opencv.core.Point(ccol, crow), radius, new Scalar(1.0), -1);
        } else {
            float[] maskData = new float[rows * cols];
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) {
                order = BUTTERWORTH_ORDER_MIN;
            }
            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;
            double twoN = 2 * order;

            for (int y = 0; y < rows; y++) {
                int dy = y - crow;
                int dy2 = dy * dy;
                for (int x = 0; x < cols; x++) {
                    int dx = x - ccol;
                    double distance = Math.sqrt(dx * dx + dy2);
                    double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
                    float value = (float) (1.0 / (1.0 + Math.pow(ratio, twoN)));
                    maskData[y * cols + x] = value;
                }
            }
            mask.put(0, 0, maskData);
        }

        return mask;
    }

    private Mat createHighPassMask(int rows, int cols, int radius, int smoothness) {
        Mat mask = new Mat(rows, cols, org.opencv.core.CvType.CV_32F);

        if (radius == 0) {
            mask.setTo(new Scalar(1.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            // Hard circle mask - inverted from low-pass
            mask.setTo(new Scalar(1.0));
            Imgproc.circle(mask, new org.opencv.core.Point(ccol, crow), radius, new Scalar(0.0), -1);
        } else {
            // Butterworth high-pass filter (1 - lowpass)
            float[] maskData = new float[rows * cols];
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) {
                order = BUTTERWORTH_ORDER_MIN;
            }
            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;
            double twoN = 2 * order;

            for (int y = 0; y < rows; y++) {
                int dy = y - crow;
                int dy2 = dy * dy;
                for (int x = 0; x < cols; x++) {
                    int dx = x - ccol;
                    double distance = Math.sqrt(dx * dx + dy2);
                    double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
                    float lowPassValue = (float) (1.0 / (1.0 + Math.pow(ratio, twoN)));
                    maskData[y * cols + x] = 1.0f - lowPassValue;
                }
            }
            mask.put(0, 0, maskData);
        }

        return mask;
    }

    // ===== FFT4 Multi-Output Processing =====

    /**
     * Process FFT High-Pass with 4 outputs:
     * [0] = filtered image (high-pass)
     * [1] = absolute difference (blocked low frequencies)
     * [2] = FFT spectrum visualization
     * [3] = filter curve visualization
     */
    private Mat[] processFFTHighPass4(Mat input, int radius, int smoothness) {
        if (input == null || input.empty()) {
            return new Mat[] { input != null ? input.clone() : null, null, null, null };
        }

        int origRows = input.rows();
        int origCols = input.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Create the HIGH-PASS filter mask at optimal size
        Mat mask = createHighPassMask(optRows, optCols, radius, smoothness);

        // Create radial filter visualization (for overlay on spectrum)
        Mat radialVis = createRadialFilterVisualization(origCols, origRows, radius, smoothness, true);

        // Create line graph filter visualization (for output 4)
        Mat filterVis = createFilterCurveVisualization(origCols, origRows, radius, smoothness, true);

        Mat filtered;
        Mat spectrum;

        if (input.channels() > 1) {
            // Process each BGR channel separately
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            List<Mat> filteredChannels = new ArrayList<>();
            Mat spectrumAccum = null;

            for (int c = 0; c < channels.size(); c++) {
                Mat channel = channels.get(c);

                // Pad to optimal size
                Mat padded = new Mat();
                Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
                    Core.BORDER_CONSTANT, Scalar.all(0));

                // Convert to float
                Mat floatChannel = new Mat();
                padded.convertTo(floatChannel, org.opencv.core.CvType.CV_32F);
                padded.release();

                // Create complex mat and compute DFT
                Mat complexI = new Mat();
                List<Mat> planes = new ArrayList<>();
                planes.add(floatChannel);
                planes.add(Mat.zeros(floatChannel.size(), org.opencv.core.CvType.CV_32F));
                Core.merge(planes, complexI);
                planes.get(1).release();
                floatChannel.release();

                Core.dft(complexI, complexI);
                fftShift(complexI);

                // For spectrum visualization, use the first channel
                if (c == 0) {
                    spectrumAccum = createSpectrumVisualization(complexI, origRows, origCols);
                }

                // Apply mask
                List<Mat> dftPlanes = new ArrayList<>();
                Core.split(complexI, dftPlanes);
                complexI.release();

                Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
                Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

                Mat maskedDft = new Mat();
                Core.merge(dftPlanes, maskedDft);
                for (Mat p : dftPlanes) p.release();

                // Inverse shift and DFT
                fftShift(maskedDft);
                Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

                // Get real part
                List<Mat> idftPlanes = new ArrayList<>();
                Core.split(maskedDft, idftPlanes);
                maskedDft.release();

                Mat magnitude = idftPlanes.get(0);
                idftPlanes.get(1).release();

                // Crop to original size
                Mat cropped = new Mat(magnitude, new org.opencv.core.Rect(0, 0, origCols, origRows));
                Mat result = cropped.clone();
                magnitude.release();

                // Clip and convert to 8-bit
                Core.min(result, new Scalar(255), result);
                Core.max(result, new Scalar(0), result);
                Mat filteredChannel = new Mat();
                result.convertTo(filteredChannel, org.opencv.core.CvType.CV_8U);
                result.release();

                filteredChannels.add(filteredChannel);
            }

            // Merge filtered channels back to BGR
            filtered = new Mat();
            Core.merge(filteredChannels, filtered);

            spectrum = spectrumAccum;

            // Release channel Mats
            for (Mat ch : channels) ch.release();
            for (Mat ch : filteredChannels) ch.release();
        } else {
            // Single channel (grayscale) processing
            Mat padded = new Mat();
            Core.copyMakeBorder(input, padded, 0, optRows - origRows, 0, optCols - origCols,
                Core.BORDER_CONSTANT, Scalar.all(0));

            Mat floatInput = new Mat();
            padded.convertTo(floatInput, org.opencv.core.CvType.CV_32F);
            padded.release();

            Mat complexI = new Mat();
            List<Mat> planes = new ArrayList<>();
            planes.add(floatInput);
            planes.add(Mat.zeros(floatInput.size(), org.opencv.core.CvType.CV_32F));
            Core.merge(planes, complexI);
            planes.get(1).release();
            floatInput.release();

            Core.dft(complexI, complexI);
            fftShift(complexI);

            spectrum = createSpectrumVisualization(complexI, origRows, origCols);

            // Apply mask
            List<Mat> dftPlanes = new ArrayList<>();
            Core.split(complexI, dftPlanes);
            complexI.release();

            Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
            Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

            Mat maskedDft = new Mat();
            Core.merge(dftPlanes, maskedDft);
            for (Mat p : dftPlanes) p.release();

            fftShift(maskedDft);
            Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

            List<Mat> idftPlanes = new ArrayList<>();
            Core.split(maskedDft, idftPlanes);
            maskedDft.release();

            Mat magnitude = idftPlanes.get(0);
            idftPlanes.get(1).release();

            Mat cropped = new Mat(magnitude, new org.opencv.core.Rect(0, 0, origCols, origRows));
            Mat result = cropped.clone();
            magnitude.release();

            Core.min(result, new Scalar(255), result);
            Core.max(result, new Scalar(0), result);
            filtered = new Mat();
            result.convertTo(filtered, org.opencv.core.CvType.CV_8U);
            result.release();

            // Convert grayscale to BGR for consistency
            Mat filteredBGR = new Mat();
            Imgproc.cvtColor(filtered, filteredBGR, Imgproc.COLOR_GRAY2BGR);
            filtered.release();
            filtered = filteredBGR;
        }

        // Calculate absolute difference (shows blocked low frequencies)
        Mat difference = new Mat();
        Core.absdiff(input, filtered, difference);

        // Overlay filter circle on spectrum (output 3 shows spectrum with filter radius)
        // For high-pass: show blocked area (center) with red tint
        if (spectrum != null && !spectrum.empty()) {
            int centerX = spectrum.cols() / 2;
            int centerY = spectrum.rows() / 2;

            if (radius > 0) {
                // Create a single-channel mask for the blocked region
                Mat circleMask = Mat.zeros(spectrum.rows(), spectrum.cols(), org.opencv.core.CvType.CV_8U);
                Imgproc.circle(circleMask, new org.opencv.core.Point(centerX, centerY),
                    radius, new Scalar(255), -1);  // White filled circle

                // Split spectrum into channels
                List<Mat> channels = new ArrayList<>();
                Core.split(spectrum, channels);

                // Create inverted mask (for keeping original values outside)
                Mat inverseMask = new Mat();
                Core.bitwise_not(circleMask, inverseMask);

                // For each channel, blend: outside circle = original, inside = tinted
                // Blue channel: darken inside circle (multiply by 0.3)
                Mat blueInside = new Mat();
                channels.get(0).copyTo(blueInside);
                Core.multiply(blueInside, new Scalar(0.3), blueInside);

                Mat blueResult = new Mat();
                channels.get(0).copyTo(blueResult, inverseMask);  // Keep original outside
                blueInside.copyTo(blueResult, circleMask);        // Use darkened inside
                channels.set(0, blueResult);
                blueInside.release();

                // Green channel: darken inside circle (multiply by 0.3)
                Mat greenInside = new Mat();
                channels.get(1).copyTo(greenInside);
                Core.multiply(greenInside, new Scalar(0.3), greenInside);

                Mat greenResult = new Mat();
                channels.get(1).copyTo(greenResult, inverseMask);
                greenInside.copyTo(greenResult, circleMask);
                channels.set(1, greenResult);
                greenInside.release();

                // Red channel: boost inside circle (add 100)
                Mat redInside = new Mat();
                channels.get(2).copyTo(redInside);
                Core.add(redInside, new Scalar(100), redInside);

                Mat redResult = new Mat();
                channels.get(2).copyTo(redResult, inverseMask);
                redInside.copyTo(redResult, circleMask);
                channels.set(2, redResult);
                redInside.release();

                // Merge channels back
                Core.merge(channels, spectrum);

                // Cleanup
                circleMask.release();
                inverseMask.release();
                for (Mat ch : channels) ch.release();

                // Draw bright green outline at radius (the filter boundary)
                Imgproc.circle(spectrum, new org.opencv.core.Point(centerX, centerY),
                    radius, new Scalar(0, 255, 0), 2);  // Green outline
            }
        }
        radialVis.release();

        mask.release();

        return new Mat[] { filtered, difference, spectrum, filterVis };
    }

    /**
     * Process FFT Low-Pass with 4 outputs.
     */
    private Mat[] processFFTLowPass4(Mat input, int radius, int smoothness) {
        if (input == null || input.empty()) {
            return new Mat[] { input != null ? input.clone() : null, null, null, null };
        }

        int origRows = input.rows();
        int origCols = input.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Create the LOW-PASS filter mask
        Mat mask = createLowPassMask(optRows, optCols, radius, smoothness);

        // Create radial filter visualization (for overlay on spectrum)
        Mat radialVis = createRadialFilterVisualization(origCols, origRows, radius, smoothness, false);

        // Create line graph filter visualization (for output 4)
        Mat filterVis = createFilterCurveVisualization(origCols, origRows, radius, smoothness, false);

        Mat filtered;
        Mat spectrum;

        if (input.channels() > 1) {
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            List<Mat> filteredChannels = new ArrayList<>();
            Mat spectrumAccum = null;

            for (int c = 0; c < channels.size(); c++) {
                Mat channel = channels.get(c);

                Mat padded = new Mat();
                Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
                    Core.BORDER_CONSTANT, Scalar.all(0));

                Mat floatChannel = new Mat();
                padded.convertTo(floatChannel, org.opencv.core.CvType.CV_32F);
                padded.release();

                Mat complexI = new Mat();
                List<Mat> planes = new ArrayList<>();
                planes.add(floatChannel);
                planes.add(Mat.zeros(floatChannel.size(), org.opencv.core.CvType.CV_32F));
                Core.merge(planes, complexI);
                planes.get(1).release();
                floatChannel.release();

                Core.dft(complexI, complexI);
                fftShift(complexI);

                if (c == 0) {
                    spectrumAccum = createSpectrumVisualization(complexI, origRows, origCols);
                }

                List<Mat> dftPlanes = new ArrayList<>();
                Core.split(complexI, dftPlanes);
                complexI.release();

                Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
                Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

                Mat maskedDft = new Mat();
                Core.merge(dftPlanes, maskedDft);
                for (Mat p : dftPlanes) p.release();

                fftShift(maskedDft);
                Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

                List<Mat> idftPlanes = new ArrayList<>();
                Core.split(maskedDft, idftPlanes);
                maskedDft.release();

                Mat magnitude = idftPlanes.get(0);
                idftPlanes.get(1).release();

                Mat cropped = new Mat(magnitude, new org.opencv.core.Rect(0, 0, origCols, origRows));
                Mat result = cropped.clone();
                magnitude.release();

                Core.min(result, new Scalar(255), result);
                Core.max(result, new Scalar(0), result);
                Mat filteredChannel = new Mat();
                result.convertTo(filteredChannel, org.opencv.core.CvType.CV_8U);
                result.release();

                filteredChannels.add(filteredChannel);
            }

            filtered = new Mat();
            Core.merge(filteredChannels, filtered);
            spectrum = spectrumAccum;

            for (Mat ch : channels) ch.release();
            for (Mat ch : filteredChannels) ch.release();
        } else {
            Mat padded = new Mat();
            Core.copyMakeBorder(input, padded, 0, optRows - origRows, 0, optCols - origCols,
                Core.BORDER_CONSTANT, Scalar.all(0));

            Mat floatInput = new Mat();
            padded.convertTo(floatInput, org.opencv.core.CvType.CV_32F);
            padded.release();

            Mat complexI = new Mat();
            List<Mat> planes = new ArrayList<>();
            planes.add(floatInput);
            planes.add(Mat.zeros(floatInput.size(), org.opencv.core.CvType.CV_32F));
            Core.merge(planes, complexI);
            planes.get(1).release();
            floatInput.release();

            Core.dft(complexI, complexI);
            fftShift(complexI);

            spectrum = createSpectrumVisualization(complexI, origRows, origCols);

            List<Mat> dftPlanes = new ArrayList<>();
            Core.split(complexI, dftPlanes);
            complexI.release();

            Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
            Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

            Mat maskedDft = new Mat();
            Core.merge(dftPlanes, maskedDft);
            for (Mat p : dftPlanes) p.release();

            fftShift(maskedDft);
            Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

            List<Mat> idftPlanes = new ArrayList<>();
            Core.split(maskedDft, idftPlanes);
            maskedDft.release();

            Mat magnitude = idftPlanes.get(0);
            idftPlanes.get(1).release();

            Mat cropped = new Mat(magnitude, new org.opencv.core.Rect(0, 0, origCols, origRows));
            Mat result = cropped.clone();
            magnitude.release();

            Core.min(result, new Scalar(255), result);
            Core.max(result, new Scalar(0), result);
            filtered = new Mat();
            result.convertTo(filtered, org.opencv.core.CvType.CV_8U);
            result.release();

            Mat filteredBGR = new Mat();
            Imgproc.cvtColor(filtered, filteredBGR, Imgproc.COLOR_GRAY2BGR);
            filtered.release();
            filtered = filteredBGR;
        }

        Mat difference = new Mat();
        Core.absdiff(input, filtered, difference);

        // Overlay filter circle on spectrum (output 3 shows spectrum with filter radius)
        // For low-pass: show blocked area (outside radius) with red tint
        if (spectrum != null && !spectrum.empty()) {
            int centerX = spectrum.cols() / 2;
            int centerY = spectrum.rows() / 2;

            if (radius > 0) {
                // Create a single-channel mask for the PASSED region (inside circle)
                Mat circleMask = Mat.zeros(spectrum.rows(), spectrum.cols(), org.opencv.core.CvType.CV_8U);
                Imgproc.circle(circleMask, new org.opencv.core.Point(centerX, centerY),
                    radius, new Scalar(255), -1);  // White filled circle = passed frequencies

                // Invert to get blocked region (outside circle)
                Mat blockedMask = new Mat();
                Core.bitwise_not(circleMask, blockedMask);

                // Split spectrum into channels
                List<Mat> channels = new ArrayList<>();
                Core.split(spectrum, channels);

                // Blue channel: darken outside circle (multiply by 0.3)
                Mat blueInside = new Mat();
                Mat blueOutside = new Mat();
                channels.get(0).copyTo(blueInside);
                channels.get(0).copyTo(blueOutside);
                Core.multiply(blueOutside, new Scalar(0.3), blueOutside);

                Mat blueResult = new Mat();
                blueInside.copyTo(blueResult, circleMask);  // Keep original inside
                blueOutside.copyTo(blueResult, blockedMask); // Use darkened outside
                channels.set(0, blueResult);
                blueInside.release();
                blueOutside.release();

                // Green channel: darken outside circle (multiply by 0.3)
                Mat greenInside = new Mat();
                Mat greenOutside = new Mat();
                channels.get(1).copyTo(greenInside);
                channels.get(1).copyTo(greenOutside);
                Core.multiply(greenOutside, new Scalar(0.3), greenOutside);

                Mat greenResult = new Mat();
                greenInside.copyTo(greenResult, circleMask);
                greenOutside.copyTo(greenResult, blockedMask);
                channels.set(1, greenResult);
                greenInside.release();
                greenOutside.release();

                // Red channel: boost outside circle (add 100)
                Mat redInside = new Mat();
                Mat redOutside = new Mat();
                channels.get(2).copyTo(redInside);
                channels.get(2).copyTo(redOutside);
                Core.add(redOutside, new Scalar(100), redOutside);

                Mat redResult = new Mat();
                redInside.copyTo(redResult, circleMask);
                redOutside.copyTo(redResult, blockedMask);
                channels.set(2, redResult);
                redInside.release();
                redOutside.release();

                // Merge channels back
                Core.merge(channels, spectrum);

                // Cleanup
                circleMask.release();
                blockedMask.release();
                for (Mat ch : channels) ch.release();

                // Draw bright green outline at radius (the filter boundary)
                Imgproc.circle(spectrum, new org.opencv.core.Point(centerX, centerY),
                    radius, new Scalar(0, 255, 0), 2);  // Green outline
            }
        }
        radialVis.release();

        mask.release();

        return new Mat[] { filtered, difference, spectrum, filterVis };
    }

    /**
     * Create FFT spectrum visualization from a shifted DFT.
     */
    private Mat createSpectrumVisualization(Mat dftShift, int origRows, int origCols) {
        System.out.println("[FXPipelineExecutor] createSpectrumVisualization: origRows=" + origRows + ", origCols=" + origCols);
        // Split into real and imaginary
        List<Mat> planes = new ArrayList<>();
        Core.split(dftShift, planes);

        // Calculate magnitude
        Mat magnitude = new Mat();
        Core.magnitude(planes.get(0), planes.get(1), magnitude);

        // Log scale for visualization
        Mat logMag = new Mat();
        Core.add(magnitude, new Scalar(1), logMag);
        Core.log(logMag, logMag);

        // Normalize to 0-255
        Core.normalize(logMag, logMag, 0, 255, Core.NORM_MINMAX);

        Mat spectrum8U = new Mat();
        logMag.convertTo(spectrum8U, org.opencv.core.CvType.CV_8U);

        // Crop to original size (center portion)
        int padRows = spectrum8U.rows();
        int padCols = spectrum8U.cols();
        int startRow = (padRows - origRows) / 2;
        int startCol = (padCols - origCols) / 2;
        Mat cropped = new Mat(spectrum8U, new org.opencv.core.Rect(startCol, startRow, origCols, origRows));
        Mat croppedClone = cropped.clone();

        // Convert to BGR for display
        Mat spectrumBGR = new Mat();
        Imgproc.cvtColor(croppedClone, spectrumBGR, Imgproc.COLOR_GRAY2BGR);

        // Cleanup
        for (Mat p : planes) p.release();
        magnitude.release();
        logMag.release();
        spectrum8U.release();
        croppedClone.release();

        return spectrumBGR;
    }

    /**
     * Create a radial disk visualization of the filter showing blocked frequencies in red.
     * Used for output 3 (spectrum with filter overlay).
     * The visualization matches the actual FFT mask - radius is in pixel units relative to the image center.
     * Creates pure red where blocked, pure black where passed (for clean addWeighted blending).
     */
    private Mat createRadialFilterVisualization(int width, int height, int radius, int smoothness, boolean highPass) {
        // Create a black background image
        Mat vis = new Mat(height, width, org.opencv.core.CvType.CV_8UC3, new Scalar(0, 0, 0));

        // Center of the image
        int centerX = width / 2;
        int centerY = height / 2;

        if (radius <= 0) {
            return vis; // No filter to visualize
        }

        if (smoothness == 0) {
            // Hard cutoff - use simple filled circle
            if (highPass) {
                // High-pass blocks center (inside radius) - draw red filled circle
                Imgproc.circle(vis, new org.opencv.core.Point(centerX, centerY),
                    radius, new Scalar(0, 0, 255), -1);  // BGR: pure red, filled
            } else {
                // Low-pass blocks outside radius - draw red ring from radius to edge
                // Fill entire image with red, then draw black circle in center
                vis.setTo(new Scalar(0, 0, 255));  // Fill with red
                Imgproc.circle(vis, new org.opencv.core.Point(centerX, centerY),
                    radius, new Scalar(0, 0, 0), -1);  // Black center, filled
            }
        } else {
            // Butterworth smooth transition - draw multiple concentric circles with varying alpha
            // For smoother effect, draw from outside in (or inside out)
            int maxRadius = (int) Math.sqrt(centerX * centerX + centerY * centerY) + 1;

            if (highPass) {
                // High-pass: blocked in center, fading out towards radius
                // Draw from center outward with decreasing intensity
                for (int r = radius * 2; r >= 0; r--) {
                    double filterValue = computeFilterValue(r, radius, smoothness, true);
                    double blockValue = 1.0 - filterValue;
                    if (blockValue > 0.01) {
                        int redIntensity = (int) (blockValue * 255);
                        Imgproc.circle(vis, new org.opencv.core.Point(centerX, centerY),
                            r, new Scalar(0, 0, redIntensity), -1);  // BGR: pure red
                    }
                }
            } else {
                // Low-pass: blocked outside radius, fading in towards radius
                // Draw from outside inward with decreasing intensity
                for (int r = maxRadius; r >= 0; r--) {
                    double filterValue = computeFilterValue(r, radius, smoothness, false);
                    double blockValue = 1.0 - filterValue;
                    if (blockValue > 0.01) {
                        int redIntensity = (int) (blockValue * 255);
                        Imgproc.circle(vis, new org.opencv.core.Point(centerX, centerY),
                            r, new Scalar(0, 0, redIntensity), -1);  // BGR: pure red
                    }
                }
            }
        }

        // Draw the filter radius boundary circle (bright cyan outline for visibility)
        if (radius > 0 && radius < Math.min(width, height) / 2) {
            Imgproc.circle(vis, new org.opencv.core.Point(centerX, centerY),
                radius, new Scalar(255, 255, 0), 2);  // Cyan outline
        }

        return vis;
    }

    /**
     * Compute filter value for a given distance.
     */
    private double computeFilterValue(double distance, int radius, int smoothness, boolean highPass) {
        if (radius == 0) {
            return highPass ? 1.0 : 0.0;
        }

        double lowPassValue;
        if (smoothness == 0) {
            // Hard cutoff
            lowPassValue = distance <= radius ? 1.0 : 0.0;
        } else {
            // Butterworth filter
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) order = BUTTERWORTH_ORDER_MIN;

            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;

            double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
            lowPassValue = 1.0 / (1.0 + Math.pow(ratio, 2 * order));
        }

        return highPass ? (1.0 - lowPassValue) : lowPassValue;
    }

    /**
     * Create a line graph visualization of the filter response curve.
     * X-axis: Distance from center (frequency)
     * Y-axis: Filter response (0=blocked, 1=passed)
     * Used for output 4 (filter curve graph).
     */
    private Mat createFilterCurveVisualization(int width, int height, int radius, int smoothness, boolean highPass) {
        System.out.println("[FXPipelineExecutor] createFilterCurveVisualization: width=" + width + ", height=" + height + ", radius=" + radius + ", highPass=" + highPass);
        // Create a black background image
        Mat vis = new Mat(height, width, org.opencv.core.CvType.CV_8UC3, new Scalar(0, 0, 0));

        // Graph margins
        int marginLeft = 50;
        int marginRight = 20;
        int marginTop = 30;
        int marginBottom = 40;

        int graphWidth = width - marginLeft - marginRight;
        int graphHeight = height - marginTop - marginBottom;

        if (graphWidth <= 0 || graphHeight <= 0) {
            return vis;
        }

        // Max distance for x-axis (2x the radius to show transition region)
        int maxDistance = Math.max(200, radius * 3);

        // Draw grid lines (gray) - thick lines for visibility when scaled down
        Scalar gridColor = new Scalar(60, 60, 60);
        // Horizontal grid lines at 0.0, 0.25, 0.5, 0.75, 1.0
        for (int i = 0; i <= 4; i++) {
            int yPos = marginTop + (int) (graphHeight * (1.0 - i / 4.0));
            Imgproc.line(vis,
                new org.opencv.core.Point(marginLeft, yPos),
                new org.opencv.core.Point(width - marginRight, yPos),
                gridColor, 6);
        }
        // Vertical grid lines
        for (int d = 0; d <= maxDistance; d += 50) {
            int xPos = marginLeft + (int) (graphWidth * d / (double) maxDistance);
            Imgproc.line(vis,
                new org.opencv.core.Point(xPos, marginTop),
                new org.opencv.core.Point(xPos, marginTop + graphHeight),
                gridColor, 6);
        }

        // Draw axes (white)
        Scalar axisColor = new Scalar(255, 255, 255);
        // Y-axis
        Imgproc.line(vis,
            new org.opencv.core.Point(marginLeft, marginTop),
            new org.opencv.core.Point(marginLeft, marginTop + graphHeight),
            axisColor, 12);
        // X-axis
        Imgproc.line(vis,
            new org.opencv.core.Point(marginLeft, marginTop + graphHeight),
            new org.opencv.core.Point(width - marginRight, marginTop + graphHeight),
            axisColor, 12);

        // Draw the filter curve (light blue) - extra thick for thumbnail visibility
        Scalar curveColor = new Scalar(255, 100, 100); // BGR - light blue
        org.opencv.core.Point prevPoint = null;
        for (int i = 0; i <= graphWidth; i++) {
            double distance = (i / (double) graphWidth) * maxDistance;
            double filterValue = computeFilterValue(distance, radius, smoothness, highPass);

            int xPos = marginLeft + i;
            int yPos = marginTop + (int) (graphHeight * (1.0 - filterValue));

            org.opencv.core.Point currentPoint = new org.opencv.core.Point(xPos, yPos);
            if (prevPoint != null) {
                Imgproc.line(vis, prevPoint, currentPoint, curveColor, 18);
            }
            prevPoint = currentPoint;
        }

        // Draw vertical line at cutoff radius (red) - thick for thumbnail visibility
        if (radius > 0 && radius <= maxDistance) {
            int radiusX = marginLeft + (int) (graphWidth * radius / (double) maxDistance);
            Scalar radiusColor = new Scalar(0, 0, 255); // BGR - red
            Imgproc.line(vis,
                new org.opencv.core.Point(radiusX, marginTop),
                new org.opencv.core.Point(radiusX, marginTop + graphHeight),
                radiusColor, 12);
        }

        // Draw labels
        Scalar textColor = new Scalar(200, 200, 200);
        double fontScale = 0.4;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        // Title
        String title = highPass ? "High-Pass Filter Response" : "Low-Pass Filter Response";
        Imgproc.putText(vis, title,
            new org.opencv.core.Point(marginLeft + 10, 20), fontFace, fontScale, textColor, 1);

        // Y-axis labels
        Imgproc.putText(vis, "1.0",
            new org.opencv.core.Point(5, marginTop + 5), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "0.5",
            new org.opencv.core.Point(5, marginTop + graphHeight / 2 + 5), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "0.0",
            new org.opencv.core.Point(5, marginTop + graphHeight + 5), fontFace, fontScale, textColor, 1);

        // X-axis labels
        Imgproc.putText(vis, "0",
            new org.opencv.core.Point(marginLeft - 5, height - 10), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, String.valueOf(maxDistance / 2),
            new org.opencv.core.Point(marginLeft + graphWidth / 2 - 10, height - 10), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, String.valueOf(maxDistance),
            new org.opencv.core.Point(width - marginRight - 20, height - 10), fontFace, fontScale, textColor, 1);

        // Radius label
        if (radius > 0) {
            Imgproc.putText(vis, "R=" + radius,
                new org.opencv.core.Point(width - marginRight - 40, marginTop + 15),
                fontFace, fontScale, new Scalar(0, 0, 255), 1);
        }

        // Smoothness label
        if (smoothness > 0) {
            Imgproc.putText(vis, "S=" + smoothness,
                new org.opencv.core.Point(width - marginRight - 40, marginTop + 30),
                fontFace, fontScale, new Scalar(100, 100, 255), 1);
        }

        return vis;
    }

    /**
     * Check if a node type is a multi-output node.
     */
    private boolean isMultiOutputNode(String nodeType) {
        return "FFTHighPass4".equals(nodeType) || "FFTLowPass4".equals(nodeType);
    }

    // ===== BitPlanes Color Helper Methods =====

    /**
     * Apply color bit plane decomposition with all bits enabled and gain 1.0.
     * This reconstructs the original image by default - configurable properties would
     * allow enabling/disabling specific bit planes per channel.
     */
    private Mat applyColorBitPlanes(Mat input) {
        if (input == null || input.empty()) {
            return input;
        }

        Mat color = null;
        boolean colorCreated = false;
        List<Mat> channels = new ArrayList<>();
        List<Mat> resultChannels = new ArrayList<>();

        try {
            // Ensure we have a color image
            if (input.channels() == 1) {
                color = new Mat();
                colorCreated = true;
                Imgproc.cvtColor(input, color, Imgproc.COLOR_GRAY2BGR);
            } else {
                color = input;
            }

            // Split into BGR channels
            Core.split(color, channels);

            // Process each channel (all 8 bits enabled, gain 1.0)
            for (int c = 0; c < channels.size(); c++) {
                Mat channel = channels.get(c);
                byte[] channelData = new byte[channel.rows() * channel.cols()];
                channel.get(0, 0, channelData);

                float[] resultData = new float[channelData.length];

                // Reconstruct from all 8 bit planes with gain 1.0
                for (int i = 0; i < 8; i++) {
                    int bitIndex = 7 - i;
                    for (int j = 0; j < channelData.length; j++) {
                        int pixelValue = channelData[j] & 0xFF;
                        int bit = (pixelValue >> bitIndex) & 1;
                        resultData[j] += bit * (1 << bitIndex);
                    }
                }

                // Clip to valid range [0, 255]
                for (int j = 0; j < resultData.length; j++) {
                    resultData[j] = Math.max(0, Math.min(255, resultData[j]));
                }

                // Convert to 8-bit
                Mat resultMat = new Mat(channel.rows(), channel.cols(), org.opencv.core.CvType.CV_32F);
                Mat result8u = new Mat();
                resultMat.put(0, 0, resultData);
                resultMat.convertTo(result8u, org.opencv.core.CvType.CV_8U);
                resultChannels.add(result8u);
                resultMat.release();
            }

            // Merge channels back
            Mat output = new Mat();
            Core.merge(resultChannels, output);
            return output;

        } finally {
            // Release intermediate Mats
            if (colorCreated && color != null) color.release();
            for (Mat ch : channels) {
                if (ch != null) ch.release();
            }
            for (Mat ch : resultChannels) {
                if (ch != null) ch.release();
            }
        }
    }
}
