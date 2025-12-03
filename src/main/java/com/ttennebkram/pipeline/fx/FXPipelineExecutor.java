package com.ttennebkram.pipeline.fx;

import com.ttennebkram.pipeline.fx.processors.FXMultiOutputProcessor;
import com.ttennebkram.pipeline.fx.processors.FXProcessorRegistry;
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

    private AtomicBoolean running = new AtomicBoolean(false);

    // Callback for updating node thumbnails
    private BiConsumer<FXNode, Mat> onNodeOutput;

    // FileSource state: cached images for source nodes
    private final Map<Integer, Mat> fileSourceCache = new HashMap<>();

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
        startPerNodeThreading();
        int processors = processorFactory.getProcessorCount();
        int totalThreads = processors + 1; // +1 for JavaFX
        System.out.println("[FXPipelineExecutor] Started: " + processors + " processors, " + totalThreads + " threads (1 for JavaFX)");
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
        // Mark top-level nodes as NOT embedded (they're the root diagram)
        for (FXNode node : executionOrder) {
            node.isEmbedded = false;  // Top-level/root diagram nodes are not embedded

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
     * Only loads from the external file if the container doesn't already have
     * substantial inner nodes (more than just boundary nodes). This preserves
     * user edits when running the pipeline from the container editor.
     */
    private void loadContainerFromExternalFile(FXNode containerNode) {
        if (containerNode.pipelineFilePath == null || containerNode.pipelineFilePath.isEmpty()) {
            return;
        }

        // If the container already has inner nodes beyond just boundary nodes,
        // don't reload from external file - the user may have made edits that
        // are only stored in the innerNodes list (saved to parent document).
        // Count non-boundary nodes to determine if user has made edits.
        if (containerNode.innerNodes != null && !containerNode.innerNodes.isEmpty()) {
            int nonBoundaryCount = 0;
            for (FXNode inner : containerNode.innerNodes) {
                if (!inner.isBoundaryNode) {
                    nonBoundaryCount++;
                }
            }
            // If there are non-boundary nodes, the container was already populated
            // (either from a previous load or from user edits in container editor)
            if (nonBoundaryCount > 0) {
                return;  // Don't reload from external file
            }
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
        // Inner nodes keep isEmbedded=true (the default), since they're inside a container
        // Also track source nodes inside the container that need feeder threads
        List<FXNode> innerSourceNodes = new ArrayList<>();
        for (FXNode innerNode : containerNode.innerNodes) {
            // isEmbedded defaults to true, so inner nodes are automatically marked as embedded

            ThreadedProcessor tp = processorFactory.createProcessor(innerNode);
            if (tp != null) {
                tp.setEnabled(innerNode.enabled);
            }

            // Track source nodes inside container for later feeder thread setup
            if (isSourceNode(innerNode)) {
                innerSourceNodes.add(innerNode);
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
        // Source nodes inside containers need special handling - they need output queues and feeder threads
        if (containerNode.innerConnections != null) {
            for (FXConnection conn : containerNode.innerConnections) {
                if (conn.source != null && conn.target != null) {
                    if (isSourceNode(conn.source)) {
                        // Source -> Processing inside container: create source output queue
                        BlockingQueue<Mat> queue = sourceOutputQueues.computeIfAbsent(
                            conn.source.id, k -> new LinkedBlockingQueue<>());
                        ThreadedProcessor targetProc = processorFactory.getProcessor(conn.target);
                        ThreadedProcessor sourceProc = processorFactory.getProcessor(conn.source);
                        if (targetProc != null) {
                            if (conn.targetInputIndex == 1) {
                                targetProc.setInputQueue2(queue);
                                if (sourceProc != null) {
                                    targetProc.setInputNode2(sourceProc);
                                }
                            } else {
                                targetProc.setInputQueue(queue);
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
        }

        // Start feeder threads for source nodes inside this container
        for (FXNode sourceNode : innerSourceNodes) {
            startSourceFeeder(sourceNode);
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
                    // Interrupted during frame processing/queue put - check if still running
                    if (!running.get()) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                    // Otherwise continue (triggered refresh)
                    Thread.interrupted(); // Clear interrupt flag
                    continue;
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
                        // Interrupted during sleep - this is a refresh trigger
                        // Clear interrupt flag and continue loop to emit new frame
                        Thread.interrupted();
                        if (!running.get()) {
                            break;
                        }
                        // Continue immediately to process with new parameters
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

        // Drain and release any remaining Mats in source output queues
        for (java.util.concurrent.BlockingQueue<Mat> queue : sourceOutputQueues.values()) {
            Mat mat;
            while ((mat = queue.poll()) != null) {
                mat.release();
            }
        }
        sourceOutputQueues.clear();

        // Release cached file source images
        for (Mat cached : fileSourceCache.values()) {
            if (cached != null && !cached.empty()) {
                cached.release();
            }
        }
        fileSourceCache.clear();
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
        return 0;
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
        triggerRefresh();
    }

    /**
     * Trigger a pipeline refresh by interrupting source feeder sleep.
     * Call this when parameters change on any node so the new settings
     * are applied immediately (especially useful for static image sources).
     */
    public void triggerRefresh() {
        // Interrupt all source feeder threads to wake them from sleep
        // This causes them to immediately emit a new frame with updated parameters
        for (Thread feederThread : sourceFeederThreads.values()) {
            if (feederThread != null && feederThread.isAlive()) {
                feederThread.interrupt();
            }
        }
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
}
