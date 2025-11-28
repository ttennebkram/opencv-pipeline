package com.ttennebkram.pipeline.fx;

import javafx.application.Platform;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;

/**
 * Executes a pipeline of connected nodes, passing frames from sources through processors.
 * This is a simplified JavaFX-native implementation that doesn't depend on SWT.
 */
public class FXPipelineExecutor {

    private final List<FXNode> nodes;
    private final List<FXConnection> connections;
    private final Map<Integer, FXWebcamSource> webcamSources;

    private Thread executionThread;
    private AtomicBoolean running = new AtomicBoolean(false);

    // Callback for updating node thumbnails
    private BiConsumer<FXNode, Mat> onNodeOutput;

    // Frame rate control
    private double targetFps = 10.0;

    public FXPipelineExecutor(List<FXNode> nodes, List<FXConnection> connections,
                              Map<Integer, FXWebcamSource> webcamSources) {
        this.nodes = nodes;
        this.connections = connections;
        this.webcamSources = webcamSources;
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
        executionThread = new Thread(this::executionLoop, "FXPipelineExecutor");
        executionThread.setDaemon(true);
        executionThread.start();
    }

    /**
     * Stop pipeline execution.
     */
    public void stop() {
        running.set(false);
        if (executionThread != null) {
            executionThread.interrupt();
            try {
                executionThread.join(1000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            executionThread = null;
        }
    }

    /**
     * Check if pipeline is running.
     */
    public boolean isRunning() {
        return running.get();
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
                            nodeOutputs.put(node.id, input.clone());
                        }
                        continue;
                    }

                    Mat output = processNode(node, nodeOutputs);
                    if (output != null) {
                        nodeOutputs.put(node.id, output);

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
     * Get input frame for a node from its connected source.
     */
    private Mat getInputForNode(FXNode node, Map<Integer, Mat> nodeOutputs) {
        for (FXConnection conn : connections) {
            if (conn.target == node && conn.source != null) {
                Mat sourceOutput = nodeOutputs.get(conn.source.id);
                if (sourceOutput != null) {
                    return sourceOutput.clone();
                }
            }
        }
        return null;
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
            // Load image from file path
            if (node.filePath != null && !node.filePath.isEmpty()) {
                Mat img = org.opencv.imgcodecs.Imgcodecs.imread(node.filePath);
                if (!img.empty()) {
                    node.outputCount1++;
                    return img;
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

                default:
                    // Pass through for unimplemented nodes
                    input.copyTo(output);
                    break;
            }
        } catch (Exception e) {
            System.err.println("Error processing " + type + ": " + e.getMessage());
            input.copyTo(output);
        }

        // Increment output counter
        node.outputCount1++;

        input.release();
        return output;
    }

    public void setTargetFps(double fps) {
        this.targetFps = fps;
    }
}
