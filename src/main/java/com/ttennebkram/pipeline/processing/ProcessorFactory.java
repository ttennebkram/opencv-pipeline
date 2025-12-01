package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.processors.FXDualInputProcessor;
import com.ttennebkram.pipeline.fx.processors.FXMultiOutputProcessor;
import com.ttennebkram.pipeline.fx.processors.FXProcessor;
import com.ttennebkram.pipeline.fx.processors.FXProcessorRegistry;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.function.Consumer;

/**
 * Factory for creating ImageProcessor instances for each node type.
 * Extracts pure OpenCV processing logic - no UI dependencies.
 */
public class ProcessorFactory {

    // Default resize dimensions
    public static final int DEFAULT_RESIZE_WIDTH = 640;
    public static final int DEFAULT_RESIZE_HEIGHT = 480;

    // Map from FXNode id to its ThreadedProcessor
    private final Map<Integer, ThreadedProcessor> processors = new HashMap<>();

    // Callback for node output
    private Consumer<NodeOutput> onNodeOutput;

    // Container internal queues: containerNodeId -> {toInput, fromOutput}
    private final Map<Integer, BlockingQueue<Mat>> containerToInputQueues = new HashMap<>();
    private final Map<Integer, BlockingQueue<Mat>> containerFromOutputQueues = new HashMap<>();

    public static class NodeOutput {
        public final FXNode node;
        public final Mat output;

        public NodeOutput(FXNode node, Mat output) {
            this.node = node;
            this.output = output;
        }
    }

    public void setOnNodeOutput(Consumer<NodeOutput> callback) {
        this.onNodeOutput = callback;
    }

    /**
     * Set the internal queues for a container node.
     * Called by FXPipelineExecutor when setting up container internals.
     * @param containerNode The container node
     * @param toInputQueue Queue from container thread to ContainerInput
     * @param fromOutputQueue Queue from ContainerOutput to container thread
     */
    public void setContainerQueues(FXNode containerNode, BlockingQueue<Mat> toInputQueue, BlockingQueue<Mat> fromOutputQueue) {
        containerToInputQueues.put(containerNode.id, toInputQueue);
        containerFromOutputQueues.put(containerNode.id, fromOutputQueue);

        // Wire both queues to the Container's ContainerProcessor
        ThreadedProcessor proc = processors.get(containerNode.id);
        if (proc instanceof ContainerProcessor) {
            ContainerProcessor containerProc = (ContainerProcessor) proc;
            containerProc.setContainerInputQueue(toInputQueue);
            containerProc.setContainerOutputQueue(fromOutputQueue);
        } else {
            System.err.println("[ProcessorFactory] WARNING: processor for " + containerNode.label + " (id=" + containerNode.id + ") is not a ContainerProcessor");
        }
    }

    /**
     * Create a ThreadedProcessor for the given FXNode.
     * For Container nodes, creates a ContainerProcessor subclass.
     */
    public ThreadedProcessor createProcessor(FXNode fxNode) {
        ThreadedProcessor tp;

        if ("Container".equals(fxNode.nodeType) || fxNode.isContainer) {
            // Container nodes use specialized ContainerProcessor
            tp = new ContainerProcessor(fxNode.label);
        } else if (isSourceNode(fxNode.nodeType)) {
            // Source nodes use SourceProcessor for backpressure signaling
            SourceProcessor sp = new SourceProcessor(fxNode.label);
            double fps = fxNode.fps > 0 ? fxNode.fps : 1.0;
            sp.setOriginalFps(fps);
            tp = sp;
        } else if (isDualInputNode(fxNode.nodeType)) {
            // Dual-input nodes use DualInputProcessor
            DualImageProcessor dualProcessor = createDualImageProcessor(fxNode);
            if (dualProcessor == null) {
                return null;
            }
            tp = new DualInputProcessor(fxNode.label, dualProcessor);
        } else if (isMultiOutputNode(fxNode.nodeType)) {
            // Multi-output nodes use MultiOutputProcessor
            // First try to create from registry (for FXMultiOutputProcessor implementations)
            FXMultiOutputProcessor fxMultiProc = FXProcessorRegistry.createMultiOutputProcessor(fxNode);
            if (fxMultiProc != null) {
                fxMultiProc.setFXNode(fxNode);
                int outputCount = fxMultiProc.getOutputCount();
                MultiOutputProcessor.MultiOutputImageProcessor multiProcessor = fxMultiProc::processMultiOutput;
                tp = new MultiOutputProcessor(fxNode.label, multiProcessor, outputCount);
            } else {
                // Fall back to legacy createMultiOutputProcessor
                MultiOutputProcessor.MultiOutputImageProcessor multiProcessor = createMultiOutputProcessor(fxNode);
                if (multiProcessor == null) {
                    return null;
                }
                tp = new MultiOutputProcessor(fxNode.label, multiProcessor, 4);
            }
        } else {
            ImageProcessor processor = createImageProcessor(fxNode);
            if (processor == null) {
                return null;
            }
            tp = new ThreadedProcessor(fxNode.label, processor);
        }

        tp.setEnabled(fxNode.enabled);
        tp.setFXNode(fxNode);  // Each processor updates its own FXNode directly

        // Set up callback to update FXNode thumbnail
        tp.setOnFrameCallback(mat -> {
            if (onNodeOutput != null && mat != null) {
                onNodeOutput.accept(new NodeOutput(fxNode, mat));
            }
        });

        processors.put(fxNode.id, tp);
        return tp;
    }

    /**
     * Check if a node type is a source node.
     */
    private boolean isSourceNode(String nodeType) {
        return "WebcamSource".equals(nodeType) ||
               "FileSource".equals(nodeType) ||
               "BlankSource".equals(nodeType);
    }

    /**
     * Check if a node type is a dual-input node.
     */
    private boolean isDualInputNode(String nodeType) {
        return "AddClamp".equals(nodeType) ||
               "SubtractClamp".equals(nodeType) ||
               "AddWeighted".equals(nodeType) ||
               "BitwiseAnd".equals(nodeType) ||
               "BitwiseOr".equals(nodeType) ||
               "BitwiseXor".equals(nodeType) ||
               "MatchTemplate".equals(nodeType);
    }

    /**
     * Check if a node type is a multi-output node.
     * Checks both the legacy hard-coded list and the FXProcessorRegistry.
     */
    private boolean isMultiOutputNode(String nodeType) {
        // Check registry first for FXMultiOutputProcessor implementations
        if (FXProcessorRegistry.isMultiOutput(nodeType)) {
            return true;
        }
        // Legacy check for hard-coded types (FFT4 nodes)
        return "FFTHighPass4".equals(nodeType) ||
               "FFTLowPass4".equals(nodeType);
    }

    /**
     * Get the processor for an FXNode.
     */
    public ThreadedProcessor getProcessor(FXNode fxNode) {
        return processors.get(fxNode.id);
    }

    /**
     * Wire a connection between two FXNodes.
     */
    public void wireConnection(FXNode source, FXNode target, int sourceOutputIndex, int targetInputIndex) {
        ThreadedProcessor sourceProc = processors.get(source.id);
        ThreadedProcessor targetProc = processors.get(target.id);

        if (sourceProc == null || targetProc == null) {
            System.err.println("[ProcessorFactory] WARNING: Cannot wire " + source.label + " -> " + target.label + ", processor is null!");
            return;
        }

        // Create queue for this connection
        BlockingQueue<Mat> queue = new java.util.concurrent.LinkedBlockingQueue<>();

        // Set source output queue (support up to 4 outputs for multi-output nodes)
        sourceProc.setOutputQueue(sourceOutputIndex, queue);

        // Set target input queue (support dual input)
        if (targetInputIndex == 1) {
            targetProc.setInputQueue2(queue);
            // Wire upstream reference for backpressure signaling
            targetProc.setInputNode2(sourceProc);
        } else {
            targetProc.setInputQueue(queue);
            // Wire upstream reference for backpressure signaling
            targetProc.setInputNode(sourceProc);
        }
    }

    /**
     * Start all processors.
     */
    public void startAll() {
        for (ThreadedProcessor tp : processors.values()) {
            tp.startProcessing();
        }
    }

    /**
     * Stop all processors.
     */
    public void stopAll() {
        for (ThreadedProcessor tp : processors.values()) {
            tp.stopProcessing();
        }
    }

    /**
     * Clear all processors.
     */
    public void clear() {
        stopAll();
        processors.clear();
    }

    /**
     * Get the number of processors (threads) currently created.
     */
    public int getProcessorCount() {
        return processors.size();
    }

    /**
     * Sync stats from processors back to FXNodes.
     * Updates input/output counters, thread priority, work units, and effective FPS.
     */
    public void syncStats(FXNode fxNode) {
        ThreadedProcessor tp = processors.get(fxNode.id);
        if (tp != null) {
            // For source nodes, outputCount1 is set by the feeder thread directly on the FXNode,
            // not by the SourceProcessor. Don't overwrite it here.
            boolean isSource = tp instanceof SourceProcessor;
            if (!isSource) {
                fxNode.inputCount = (int) tp.getInputReads1();
                fxNode.outputCount1 = (int) tp.getOutputWrites1();
            }
            // Sync backpressure stats for display (applies to all nodes including sources)
            fxNode.threadPriority = tp.getThreadPriority();
            fxNode.workUnitsCompleted = tp.getWorkUnitsCompleted();
            fxNode.effectiveFps = tp.getEffectiveFps();
        }
    }

    /**
     * Create an ImageProcessor for the given node type.
     * First checks the FXProcessorRegistry for modular processors,
     * then falls back to the legacy switch statement.
     */
    private ImageProcessor createImageProcessor(FXNode fxNode) {
        String type = fxNode.nodeType;

        // Try to use new modular processor from registry
        if (FXProcessorRegistry.hasProcessor(type) && !FXProcessorRegistry.isDualInput(type)) {
            FXProcessor processor = FXProcessorRegistry.createProcessor(fxNode);
            if (processor != null) {
                // Set the FXNode reference for live property updates
                processor.setFXNode(fxNode);
                return processor.createImageProcessor();
            }
        }

        // Legacy switch statement for processors not yet migrated
        switch (type) {
            // Basic processing
            case "Grayscale":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Imgproc.cvtColor(input, output, Imgproc.COLOR_BGR2GRAY);
                    Mat bgr = new Mat();
                    Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                    output.release();
                    return bgr;
                };

            case "Invert":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Core.bitwise_not(input, output);
                    return output;
                };

            case "Threshold":
                return input -> {
                    if (input == null || input.empty()) return input;
                    double threshold = 127;
                    double maxValue = 255;
                    if (fxNode.properties.containsKey("threshold")) {
                        threshold = ((Number) fxNode.properties.get("threshold")).doubleValue();
                    }
                    if (fxNode.properties.containsKey("maxValue")) {
                        maxValue = ((Number) fxNode.properties.get("maxValue")).doubleValue();
                    }
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat output = new Mat();
                    Imgproc.threshold(gray, output, threshold, maxValue, Imgproc.THRESH_BINARY);
                    gray.release();
                    Mat bgr = new Mat();
                    Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                    output.release();
                    return bgr;
                };

            case "GaussianBlur":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 15;
                    double sigmaX = 0;
                    if (fxNode.properties.containsKey("kernelSize")) {
                        ksize = ((Number) fxNode.properties.get("kernelSize")).intValue();
                    }
                    if (fxNode.properties.containsKey("sigmaX")) {
                        sigmaX = ((Number) fxNode.properties.get("sigmaX")).doubleValue();
                    }
                    if (ksize % 2 == 0) ksize++;  // Must be odd
                    Mat output = new Mat();
                    Imgproc.GaussianBlur(input, output, new Size(ksize, ksize), sigmaX);
                    return output;
                };

            case "MedianBlur":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 5;
                    if (fxNode.properties.containsKey("ksize")) {
                        ksize = ((Number) fxNode.properties.get("ksize")).intValue();
                    }
                    if (ksize % 2 == 0) ksize++;  // Must be odd
                    if (ksize < 1) ksize = 1;
                    Mat output = new Mat();
                    Imgproc.medianBlur(input, output, ksize);
                    return output;
                };

            case "BilateralFilter":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Imgproc.bilateralFilter(input, output, 9, 75, 75);
                    return output;
                };

            case "BoxBlur":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 5;
                    if (fxNode.properties.containsKey("ksize")) {
                        ksize = ((Number) fxNode.properties.get("ksize")).intValue();
                    }
                    if (ksize < 1) ksize = 1;
                    Mat output = new Mat();
                    Imgproc.blur(input, output, new Size(ksize, ksize));
                    return output;
                };

            case "CannyEdge":
                return input -> {
                    if (input == null || input.empty()) return input;
                    double threshold1 = 100;
                    double threshold2 = 200;
                    if (fxNode.properties.containsKey("threshold1")) {
                        threshold1 = ((Number) fxNode.properties.get("threshold1")).doubleValue();
                    }
                    if (fxNode.properties.containsKey("threshold2")) {
                        threshold2 = ((Number) fxNode.properties.get("threshold2")).doubleValue();
                    }
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat output = new Mat();
                    Imgproc.Canny(gray, output, threshold1, threshold2);
                    gray.release();
                    Mat bgr = new Mat();
                    Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                    output.release();
                    return bgr;
                };

            case "Sobel":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat gradX = new Mat();
                    Mat gradY = new Mat();
                    Imgproc.Sobel(gray, gradX, CvType.CV_16S, 1, 0);
                    Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, 1);
                    Core.convertScaleAbs(gradX, gradX);
                    Core.convertScaleAbs(gradY, gradY);
                    Mat output = new Mat();
                    Core.addWeighted(gradX, 0.5, gradY, 0.5, 0, output);
                    gray.release();
                    gradX.release();
                    gradY.release();
                    Mat bgr = new Mat();
                    Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                    output.release();
                    return bgr;
                };

            case "Laplacian":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat output = new Mat();
                    Imgproc.Laplacian(gray, output, CvType.CV_16S);
                    Core.convertScaleAbs(output, output);
                    gray.release();
                    Mat bgr = new Mat();
                    Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                    output.release();
                    return bgr;
                };

            case "Erode":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 5;
                    int iterations = 1;
                    if (fxNode.properties.containsKey("kernelSize")) {
                        ksize = ((Number) fxNode.properties.get("kernelSize")).intValue();
                    }
                    if (fxNode.properties.containsKey("iterations")) {
                        iterations = ((Number) fxNode.properties.get("iterations")).intValue();
                    }
                    if (ksize < 1) ksize = 1;
                    if (iterations < 1) iterations = 1;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
                    Imgproc.erode(input, output, kernel, new Point(-1, -1), iterations);
                    kernel.release();
                    return output;
                };

            case "Dilate":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 5;
                    int iterations = 1;
                    if (fxNode.properties.containsKey("kernelSize")) {
                        ksize = ((Number) fxNode.properties.get("kernelSize")).intValue();
                    }
                    if (fxNode.properties.containsKey("iterations")) {
                        iterations = ((Number) fxNode.properties.get("iterations")).intValue();
                    }
                    if (ksize < 1) ksize = 1;
                    if (iterations < 1) iterations = 1;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
                    Imgproc.dilate(input, output, kernel, new Point(-1, -1), iterations);
                    kernel.release();
                    return output;
                };

            case "MorphOpen":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 5;
                    if (fxNode.properties.containsKey("kernelSize")) {
                        ksize = ((Number) fxNode.properties.get("kernelSize")).intValue();
                    }
                    if (ksize < 1) ksize = 1;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_OPEN, kernel);
                    kernel.release();
                    return output;
                };

            case "MorphClose":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int ksize = 5;
                    if (fxNode.properties.containsKey("kernelSize")) {
                        ksize = ((Number) fxNode.properties.get("kernelSize")).intValue();
                    }
                    if (ksize < 1) ksize = 1;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(ksize, ksize));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, kernel);
                    kernel.release();
                    return output;
                };

            case "BitwiseNot":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Core.bitwise_not(input, output);
                    return output;
                };

            case "CLAHE":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat lab = new Mat();
                    Imgproc.cvtColor(input, lab, Imgproc.COLOR_BGR2Lab);
                    List<Mat> channels = new ArrayList<>();
                    Core.split(lab, channels);
                    CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8, 8));
                    clahe.apply(channels.get(0), channels.get(0));
                    Core.merge(channels, lab);
                    Mat output = new Mat();
                    Imgproc.cvtColor(lab, output, Imgproc.COLOR_Lab2BGR);
                    lab.release();
                    for (Mat ch : channels) ch.release();
                    return output;
                };

            case "Gain":
                return input -> {
                    if (input == null || input.empty()) return input;
                    double gain = 1.0;
                    if (fxNode.properties.containsKey("gain")) {
                        gain = ((Number) fxNode.properties.get("gain")).doubleValue();
                    }
                    Mat output = new Mat();
                    input.convertTo(output, -1, gain, 0);
                    return output;
                };

            case "Clone":
                return input -> {
                    if (input == null || input.empty()) return input;
                    return input.clone();
                };

            case "Monitor":
                return input -> {
                    if (input == null || input.empty()) return input;
                    return input.clone();
                };

            case "Contours":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat binary = new Mat();
                    Imgproc.threshold(gray, binary, 127, 255, Imgproc.THRESH_BINARY);
                    List<MatOfPoint> contours = new ArrayList<>();
                    Mat hierarchy = new Mat();
                    Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
                    Mat output = input.clone();
                    Imgproc.drawContours(output, contours, -1, new Scalar(0, 255, 0), 2);
                    gray.release();
                    binary.release();
                    hierarchy.release();
                    return output;
                };

            case "HoughCircles":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Imgproc.GaussianBlur(gray, gray, new Size(9, 9), 2);
                    Mat circles = new Mat();
                    Imgproc.HoughCircles(gray, circles, Imgproc.HOUGH_GRADIENT, 1, 20, 100, 50, 0, 0);
                    Mat output = input.clone();
                    for (int i = 0; i < circles.cols(); i++) {
                        double[] c = circles.get(0, i);
                        Point center = new Point(c[0], c[1]);
                        int radius = (int) Math.round(c[2]);
                        Imgproc.circle(output, center, radius, new Scalar(0, 255, 0), 2);
                        Imgproc.circle(output, center, 3, new Scalar(0, 0, 255), -1);
                    }
                    gray.release();
                    circles.release();
                    return output;
                };

            case "HarrisCorners":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat gray32f = new Mat();
                    gray.convertTo(gray32f, CvType.CV_32F);
                    Mat corners = new Mat();
                    Imgproc.cornerHarris(gray32f, corners, 2, 3, 0.04);
                    Mat output = input.clone();
                    double[] minMax = new double[2];
                    Core.minMaxLoc(corners).maxVal = minMax[1];
                    double thresh = minMax[1] * 0.01;
                    for (int y = 0; y < corners.rows(); y++) {
                        for (int x = 0; x < corners.cols(); x++) {
                            if (corners.get(y, x)[0] > thresh) {
                                Imgproc.circle(output, new Point(x, y), 5, new Scalar(0, 0, 255), 2);
                            }
                        }
                    }
                    gray.release();
                    gray32f.release();
                    corners.release();
                    return output;
                };

            case "ORBFeatures":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    ORB orb = ORB.create();
                    MatOfKeyPoint keypoints = new MatOfKeyPoint();
                    orb.detect(gray, keypoints);
                    Mat output = new Mat();
                    Features2d.drawKeypoints(input, keypoints, output, new Scalar(0, 255, 0), Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    gray.release();
                    return output;
                };

            case "SIFTFeatures":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    SIFT sift = SIFT.create();
                    MatOfKeyPoint keypoints = new MatOfKeyPoint();
                    sift.detect(gray, keypoints);
                    Mat output = new Mat();
                    Features2d.drawKeypoints(input, keypoints, output, new Scalar(0, 255, 0), Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                    gray.release();
                    return output;
                };

            case "Histogram":
                return input -> {
                    if (input == null || input.empty()) return input;
                    // Create histogram visualization
                    List<Mat> bgr = new ArrayList<>();
                    Core.split(input, bgr);
                    int histSize = 256;
                    float[] range = {0, 256};
                    MatOfFloat histRange = new MatOfFloat(range);
                    MatOfInt channels = new MatOfInt(0);
                    Mat histB = new Mat(), histG = new Mat(), histR = new Mat();
                    Imgproc.calcHist(java.util.Arrays.asList(bgr.get(0)), channels, new Mat(), histB, new MatOfInt(histSize), histRange);
                    if (bgr.size() > 1) {
                        Imgproc.calcHist(java.util.Arrays.asList(bgr.get(1)), channels, new Mat(), histG, new MatOfInt(histSize), histRange);
                    }
                    if (bgr.size() > 2) {
                        Imgproc.calcHist(java.util.Arrays.asList(bgr.get(2)), channels, new Mat(), histR, new MatOfInt(histSize), histRange);
                    }
                    int histW = 512, histH = 400;
                    Mat output = Mat.zeros(histH, histW, CvType.CV_8UC3);
                    Core.normalize(histB, histB, 0, histH, Core.NORM_MINMAX);
                    if (!histG.empty()) Core.normalize(histG, histG, 0, histH, Core.NORM_MINMAX);
                    if (!histR.empty()) Core.normalize(histR, histR, 0, histH, Core.NORM_MINMAX);
                    int binW = (int) Math.round((double) histW / histSize);
                    for (int i = 1; i < histSize; i++) {
                        Imgproc.line(output,
                            new Point(binW * (i - 1), histH - Math.round(histB.get(i - 1, 0)[0])),
                            new Point(binW * i, histH - Math.round(histB.get(i, 0)[0])),
                            new Scalar(255, 0, 0), 2);
                        if (!histG.empty()) {
                            Imgproc.line(output,
                                new Point(binW * (i - 1), histH - Math.round(histG.get(i - 1, 0)[0])),
                                new Point(binW * i, histH - Math.round(histG.get(i, 0)[0])),
                                new Scalar(0, 255, 0), 2);
                        }
                        if (!histR.empty()) {
                            Imgproc.line(output,
                                new Point(binW * (i - 1), histH - Math.round(histR.get(i - 1, 0)[0])),
                                new Point(binW * i, histH - Math.round(histR.get(i, 0)[0])),
                                new Scalar(0, 0, 255), 2);
                        }
                    }
                    for (Mat m : bgr) m.release();
                    histB.release();
                    if (!histG.empty()) histG.release();
                    if (!histR.empty()) histR.release();
                    return output;
                };

            case "FFTHighPass":
                return createFFTHighPassProcessor(fxNode);
            // Note: FFTHighPass4 and FFTLowPass4 are handled separately via createMultiOutputProcessor

            case "BitPlanesColor":
                return createBitPlanesColorProcessor(fxNode);

            case "FFTLowPass":
                return createFFTLowPassProcessor(fxNode);
            // Note: FFTLowPass4 is handled separately via createMultiOutputProcessor

            case "BitPlanesGrayscale":
                return createBitPlanesGrayscaleProcessor(fxNode);

            // Dual input nodes - these need special handling
            case "AddClamp":
            case "SubtractClamp":
            case "AddWeighted":
            case "BitwiseAnd":
            case "BitwiseOr":
            case "BitwiseXor":
            case "MatchTemplate":
                // For now, return passthrough - dual input needs separate handling
                return input -> input != null ? input.clone() : null;

            // Container nodes use ContainerProcessor subclass, not ImageProcessor
            // (handled in createProcessor, not here)

            // ContainerInput - passthrough processor (receives from container, sends to inner pipeline)
            case "ContainerInput":
                return input -> input != null ? input.clone() : null;

            // ContainerOutput - passthrough processor (receives from inner pipeline, sends to container)
            case "ContainerOutput":
                return input -> input != null ? input.clone() : null;

            // Transform
            case "Resize":
                return input -> {
                    if (input == null || input.empty()) return input;
                    int width = DEFAULT_RESIZE_WIDTH;
                    int height = DEFAULT_RESIZE_HEIGHT;
                    if (fxNode.properties.containsKey("width")) {
                        width = ((Number) fxNode.properties.get("width")).intValue();
                    }
                    if (fxNode.properties.containsKey("height")) {
                        height = ((Number) fxNode.properties.get("height")).intValue();
                    }
                    if (width <= 0) width = DEFAULT_RESIZE_WIDTH;
                    if (height <= 0) height = DEFAULT_RESIZE_HEIGHT;
                    Mat output = new Mat();
                    Imgproc.resize(input, output, new Size(width, height));
                    return output;
                };

            // Source nodes - handled separately
            case "WebcamSource":
            case "FileSource":
            case "BlankSource":
                return null;

            default:
                System.err.println("Unknown node type: " + type);
                return input -> input != null ? input.clone() : null;
        }
    }

    /**
     * Create a DualImageProcessor for dual-input node types.
     * First checks the FXProcessorRegistry for modular processors,
     * then falls back to the legacy switch statement.
     */
    private DualImageProcessor createDualImageProcessor(FXNode fxNode) {
        String type = fxNode.nodeType;

        // Try to use new modular processor from registry
        if (FXProcessorRegistry.hasProcessor(type) && FXProcessorRegistry.isDualInput(type)) {
            FXProcessor processor = FXProcessorRegistry.createProcessor(fxNode);
            if (processor instanceof FXDualInputProcessor) {
                return ((FXDualInputProcessor) processor).createDualImageProcessor();
            }
        }

        // Legacy switch statement for processors not yet migrated
        switch (type) {
            case "AddClamp":
                return (input1, input2) -> {
                    if (input1 == null && input2 == null) return null;
                    if (input1 == null) return input2.clone();
                    if (input2 == null) return input1.clone();

                    Mat output = new Mat();
                    Mat resized2 = ensureSameSize(input1, input2);
                    Mat converted2 = ensureSameType(input1, resized2);
                    Core.add(input1, converted2, output);
                    if (resized2 != input2) resized2.release();
                    if (converted2 != resized2) converted2.release();
                    return output;
                };

            case "SubtractClamp":
                return (input1, input2) -> {
                    if (input1 == null && input2 == null) return null;
                    if (input1 == null) return input2.clone();
                    if (input2 == null) return input1.clone();

                    Mat output = new Mat();
                    Mat resized2 = ensureSameSize(input1, input2);
                    Mat converted2 = ensureSameType(input1, resized2);
                    Core.subtract(input1, converted2, output);
                    if (resized2 != input2) resized2.release();
                    if (converted2 != resized2) converted2.release();
                    return output;
                };

            case "BitwiseAnd":
                return (input1, input2) -> {
                    if (input1 == null && input2 == null) return null;
                    if (input1 == null) return input2.clone();
                    if (input2 == null) return input1.clone();

                    Mat output = new Mat();
                    Mat resized2 = ensureSameSize(input1, input2);
                    Mat converted2 = ensureSameType(input1, resized2);
                    Core.bitwise_and(input1, converted2, output);
                    if (resized2 != input2) resized2.release();
                    if (converted2 != resized2) converted2.release();
                    return output;
                };

            case "BitwiseOr":
                return (input1, input2) -> {
                    if (input1 == null && input2 == null) return null;
                    if (input1 == null) return input2.clone();
                    if (input2 == null) return input1.clone();

                    Mat output = new Mat();
                    Mat resized2 = ensureSameSize(input1, input2);
                    Mat converted2 = ensureSameType(input1, resized2);
                    Core.bitwise_or(input1, converted2, output);
                    if (resized2 != input2) resized2.release();
                    if (converted2 != resized2) converted2.release();
                    return output;
                };

            case "BitwiseXor":
                return (input1, input2) -> {
                    if (input1 == null && input2 == null) return null;
                    if (input1 == null) return input2.clone();
                    if (input2 == null) return input1.clone();

                    Mat output = new Mat();
                    Mat resized2 = ensureSameSize(input1, input2);
                    Mat converted2 = ensureSameType(input1, resized2);
                    Core.bitwise_xor(input1, converted2, output);
                    if (resized2 != input2) resized2.release();
                    if (converted2 != resized2) converted2.release();
                    return output;
                };

            case "AddWeighted":
                return (input1, input2) -> {
                    if (input1 == null && input2 == null) return null;
                    if (input1 == null) return input2.clone();
                    if (input2 == null) return input1.clone();

                    // Get alpha from properties (default 0.5)
                    double alpha = 0.5;
                    if (fxNode.properties.containsKey("alpha")) {
                        alpha = ((Number) fxNode.properties.get("alpha")).doubleValue();
                    }
                    double beta = 1.0 - alpha;

                    Mat output = new Mat();
                    Mat resized2 = ensureSameSize(input1, input2);
                    Mat converted2 = ensureSameType(input1, resized2);
                    Core.addWeighted(input1, alpha, converted2, beta, 0, output);
                    if (resized2 != input2) resized2.release();
                    if (converted2 != resized2) converted2.release();
                    return output;
                };

            case "MatchTemplate":
                // Template matching is more complex - for now just passthrough
                return (input1, input2) -> input1 != null ? input1.clone() : null;

            default:
                System.err.println("Unknown dual-input node type: " + type);
                return (input1, input2) -> input1 != null ? input1.clone() : null;
        }
    }

    /**
     * Resize input2 to match input1 dimensions if needed.
     */
    private Mat ensureSameSize(Mat input1, Mat input2) {
        if (input1.width() == input2.width() && input1.height() == input2.height()) {
            return input2;
        }
        Mat resized = new Mat();
        Imgproc.resize(input2, resized, new Size(input1.width(), input1.height()));
        return resized;
    }

    /**
     * Convert input2 to match input1 type if needed.
     */
    private Mat ensureSameType(Mat input1, Mat input2) {
        if (input1.type() == input2.type()) {
            return input2;
        }
        Mat converted = new Mat();
        input2.convertTo(converted, input1.type());
        return converted;
    }

    // ====== Multi-Output Processor (FFT4) Implementation ======

    /**
     * Create a multi-output processor for FFT4 nodes.
     * Returns 4 outputs:
     * [0] = filtered image
     * [1] = difference (input - filtered) - shows blocked frequencies
     * [2] = spectrum visualization
     * [3] = filter curve visualization
     */
    private MultiOutputProcessor.MultiOutputImageProcessor createMultiOutputProcessor(FXNode fxNode) {
        String type = fxNode.nodeType;

        if ("FFTHighPass4".equals(type)) {
            return input -> processFFT4(input, fxNode, true);
        } else if ("FFTLowPass4".equals(type)) {
            return input -> processFFT4(input, fxNode, false);
        }

        return null;
    }

    /**
     * Process FFT with 4 outputs (filtered, difference, spectrum, filter curve).
     * @param isHighPass true for high-pass, false for low-pass
     */
    private Mat[] processFFT4(Mat input, FXNode fxNode, boolean isHighPass) {
        if (input == null || input.empty()) {
            return new Mat[] { input != null ? input.clone() : null, null, null, null };
        }

        // Read properties
        int radius = 30;
        int smoothness = 0;
        if (fxNode.properties.containsKey("radius")) {
            radius = ((Number) fxNode.properties.get("radius")).intValue();
        }
        if (fxNode.properties.containsKey("smoothness")) {
            smoothness = ((Number) fxNode.properties.get("smoothness")).intValue();
        }

        int origRows = input.rows();
        int origCols = input.cols();
        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Create filter mask
        Mat mask = isHighPass
            ? createFFTMask(optRows, optCols, radius, smoothness)  // High-pass
            : createFFTLowPassMask(optRows, optCols, radius, smoothness);  // Low-pass

        // Create filter curve visualization at original image size
        Mat filterVis = createFilterCurveVisualization(origCols, origRows, radius, smoothness, isHighPass);

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
                padded.convertTo(floatChannel, CvType.CV_32F);
                padded.release();

                // Compute DFT
                Mat dft = new Mat();
                Core.dft(floatChannel, dft, Core.DFT_COMPLEX_OUTPUT);
                floatChannel.release();

                // Shift zero frequency to center
                fftShift(dft);

                // For spectrum visualization, use the first channel
                if (c == 0) {
                    spectrumAccum = createSpectrumVisualization(dft, origRows, origCols, radius);
                }

                // Apply mask
                List<Mat> dftPlanes = new ArrayList<>();
                Core.split(dft, dftPlanes);
                dft.release();

                Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
                Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

                Mat maskedDft = new Mat();
                Core.merge(dftPlanes, maskedDft);
                for (Mat p : dftPlanes) p.release();

                // Inverse shift
                fftShift(maskedDft);

                // Inverse DFT
                Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

                // Get real part
                List<Mat> idftPlanes = new ArrayList<>();
                Core.split(maskedDft, idftPlanes);
                maskedDft.release();

                Mat magnitude = idftPlanes.get(0);
                idftPlanes.get(1).release();

                // Crop to original size
                Mat cropped = new Mat(magnitude, new Rect(0, 0, origCols, origRows));
                Mat result = cropped.clone();
                magnitude.release();

                // Clip and convert to 8-bit
                Core.min(result, new Scalar(255), result);
                Core.max(result, new Scalar(0), result);
                Mat filteredChannel = new Mat();
                result.convertTo(filteredChannel, CvType.CV_8U);
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
            // Grayscale processing
            Mat padded = new Mat();
            Core.copyMakeBorder(input, padded, 0, optRows - origRows, 0, optCols - origCols,
                Core.BORDER_CONSTANT, Scalar.all(0));

            Mat floatInput = new Mat();
            padded.convertTo(floatInput, CvType.CV_32F);
            padded.release();

            Mat dft = new Mat();
            Core.dft(floatInput, dft, Core.DFT_COMPLEX_OUTPUT);
            floatInput.release();

            fftShift(dft);

            spectrum = createSpectrumVisualization(dft, origRows, origCols, radius);

            List<Mat> dftPlanes = new ArrayList<>();
            Core.split(dft, dftPlanes);
            dft.release();

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

            Mat cropped = new Mat(magnitude, new Rect(0, 0, origCols, origRows));
            Mat result = cropped.clone();
            magnitude.release();

            Core.min(result, new Scalar(255), result);
            Core.max(result, new Scalar(0), result);
            filtered = new Mat();
            result.convertTo(filtered, CvType.CV_8U);
            result.release();
        }

        // Calculate absolute difference (shows blocked frequencies)
        Mat difference = new Mat();
        Core.absdiff(input, filtered, difference);

        mask.release();

        return new Mat[] { filtered, difference, spectrum, filterVis };
    }

    /**
     * Create spectrum visualization from DFT with red circle showing filter radius.
     */
    private Mat createSpectrumVisualization(Mat dftShift, int origRows, int origCols, int radius) {
        List<Mat> planes = new ArrayList<>();
        Core.split(dftShift, planes);

        Mat magnitude = new Mat();
        Core.magnitude(planes.get(0), planes.get(1), magnitude);

        // Log scale for visualization
        Mat logMag = new Mat();
        Core.add(magnitude, new Scalar(1), logMag);
        Core.log(logMag, logMag);

        // Normalize to 0-255
        Core.normalize(logMag, logMag, 0, 255, Core.NORM_MINMAX);

        Mat spectrum = new Mat();
        logMag.convertTo(spectrum, CvType.CV_8U);

        // Crop to original size (center portion)
        int padRows = spectrum.rows();
        int padCols = spectrum.cols();
        int startRow = (padRows - origRows) / 2;
        int startCol = (padCols - origCols) / 2;
        Mat cropped = new Mat(spectrum, new Rect(startCol, startRow, origCols, origRows));
        Mat croppedClone = cropped.clone();

        // Convert to BGR for display
        Mat spectrumBGR = new Mat();
        Imgproc.cvtColor(croppedClone, spectrumBGR, Imgproc.COLOR_GRAY2BGR);

        // Draw filled red circle with 50% transparency showing filter radius
        if (radius > 0) {
            int centerX = origCols / 2;
            int centerY = origRows / 2;
            // Create overlay with filled circle
            Mat overlay = spectrumBGR.clone();
            Imgproc.circle(overlay, new Point(centerX, centerY), radius, new Scalar(0, 0, 255), -1); // -1 = filled
            // Blend with 50% transparency
            Core.addWeighted(overlay, 0.5, spectrumBGR, 0.5, 0, spectrumBGR);
            overlay.release();
        }

        // Cleanup
        for (Mat p : planes) p.release();
        magnitude.release();
        logMag.release();
        spectrum.release();
        croppedClone.release();

        return spectrumBGR;
    }

    /**
     * Create filter curve visualization showing frequency response.
     */
    private Mat createFilterCurveVisualization(int width, int height, int radius, int smoothness, boolean isHighPass) {
        // Create black background
        Mat vis = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));

        int marginLeft = 50;
        int marginRight = 20;
        int marginTop = 30;
        int marginBottom = 40;

        int graphWidth = width - marginLeft - marginRight;
        int graphHeight = height - marginTop - marginBottom;

        if (graphWidth <= 0 || graphHeight <= 0) {
            return vis;
        }

        // Draw grid lines (gray)
        Scalar gridColor = new Scalar(60, 60, 60);
        for (int i = 0; i <= 4; i++) {
            int yPos = marginTop + (int) (graphHeight * (1.0 - i / 4.0));
            Imgproc.line(vis, new Point(marginLeft, yPos), new Point(width - marginRight, yPos), gridColor, 1);
        }
        int maxDistance = 200;
        for (int d = 0; d <= maxDistance; d += 50) {
            int xPos = marginLeft + (int) (graphWidth * d / (double) maxDistance);
            Imgproc.line(vis, new Point(xPos, marginTop), new Point(xPos, marginTop + graphHeight), gridColor, 1);
        }

        // Draw axes (white)
        Scalar axisColor = new Scalar(255, 255, 255);
        Imgproc.line(vis, new Point(marginLeft, marginTop), new Point(marginLeft, marginTop + graphHeight), axisColor, 2);
        Imgproc.line(vis, new Point(marginLeft, marginTop + graphHeight), new Point(width - marginRight, marginTop + graphHeight), axisColor, 2);

        // Draw the filter curve (blue for high-pass, red for low-pass)
        Scalar curveColor = isHighPass ? new Scalar(255, 100, 100) : new Scalar(100, 100, 255);
        Point prevPoint = null;
        for (int i = 0; i <= graphWidth; i++) {
            double distance = (i / (double) graphWidth) * maxDistance;
            double filterValue = computeFilterValue(distance, radius, smoothness, isHighPass);

            int xPos = marginLeft + i;
            int yPos = marginTop + (int) (graphHeight * (1.0 - filterValue));

            Point currentPoint = new Point(xPos, yPos);
            if (prevPoint != null) {
                Imgproc.line(vis, prevPoint, currentPoint, curveColor, 2);
            }
            prevPoint = currentPoint;
        }

        // Draw vertical line at radius (red/blue)
        if (radius > 0 && radius <= maxDistance) {
            int radiusX = marginLeft + (int) (graphWidth * radius / (double) maxDistance);
            Scalar radiusColor = isHighPass ? new Scalar(0, 0, 255) : new Scalar(255, 0, 0);
            Imgproc.line(vis, new Point(radiusX, marginTop), new Point(radiusX, marginTop + graphHeight), radiusColor, 1);
        }

        // Draw labels
        Scalar textColor = new Scalar(200, 200, 200);
        double fontScale = 0.4;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        String title = isHighPass ? "High-Pass Filter Response" : "Low-Pass Filter Response";
        Imgproc.putText(vis, title, new Point(marginLeft + 10, 20), fontFace, fontScale, textColor, 1);

        Imgproc.putText(vis, "1.0", new Point(5, marginTop + 5), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "0.5", new Point(5, marginTop + graphHeight / 2 + 5), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "0.0", new Point(5, marginTop + graphHeight + 5), fontFace, fontScale, textColor, 1);

        Imgproc.putText(vis, "0", new Point(marginLeft - 5, height - 10), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "100", new Point(marginLeft + graphWidth / 2 - 10, height - 10), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "200", new Point(width - marginRight - 15, height - 10), fontFace, fontScale, textColor, 1);

        if (radius > 0) {
            Imgproc.putText(vis, "R=" + radius, new Point(marginLeft + graphWidth - 40, marginTop + 15),
                fontFace, fontScale, new Scalar(0, 0, 255), 1);
        }

        return vis;
    }

    /**
     * Compute filter value for a given distance.
     */
    private double computeFilterValue(double distance, int radius, int smoothness, boolean isHighPass) {
        if (radius == 0) {
            return 1.0;
        }

        if (smoothness == 0) {
            // Hard cutoff
            if (isHighPass) {
                return distance <= radius ? 0.0 : 1.0;
            } else {
                return distance <= radius ? 1.0 : 0.0;
            }
        } else {
            // Butterworth filter
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) order = BUTTERWORTH_ORDER_MIN;

            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;

            if (isHighPass) {
                double ratio = effectiveCutoff / (distance + BUTTERWORTH_DIVISION_EPSILON);
                return 1.0 / (1.0 + Math.pow(ratio, 2 * order));
            } else {
                double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
                return 1.0 / (1.0 + Math.pow(ratio, 2 * order));
            }
        }
    }

    // ====== FFT High-Pass Filter Implementation ======

    // Butterworth filter constants
    private static final double BUTTERWORTH_ORDER_MAX = 10.0;
    private static final double BUTTERWORTH_ORDER_MIN = 0.5;
    private static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    private static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    private static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    private static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    private ImageProcessor createFFTHighPassProcessor(FXNode fxNode) {
        return input -> {
            if (input == null || input.empty()) return input;

            // Read properties from FXNode
            int radius = 0;
            int smoothness = 0;
            if (fxNode.properties.containsKey("radius")) {
                radius = ((Number) fxNode.properties.get("radius")).intValue();
            }
            if (fxNode.properties.containsKey("smoothness")) {
                smoothness = ((Number) fxNode.properties.get("smoothness")).intValue();
            }

            // Split into BGR channels
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            try {
                // Apply FFT filter to each channel
                List<Mat> filteredChannels = new ArrayList<>();
                try {
                    for (Mat channel : channels) {
                        Mat filtered = applyFFTToChannel(channel, radius, smoothness);
                        filteredChannels.add(filtered);
                    }

                    // Merge filtered channels
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
        };
    }

    private int getOptimalFFTSize(int n) {
        int pow2 = nextPowerOf2(n);
        int optimal = Core.getOptimalDFTSize(n);
        // Use power of 2 if it adds less than 25% overhead
        if (pow2 <= optimal * 1.25) {
            return pow2;
        }
        return optimal % 2 == 0 ? optimal : optimal + 1;
    }

    private int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    private Mat applyFFTToChannel(Mat channel, int radius, int smoothness) {
        int origRows = channel.rows();
        int origCols = channel.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Pad image to optimal size
        Mat padded = new Mat();
        Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
            Core.BORDER_CONSTANT, Scalar.all(0));

        // Convert to float
        Mat floatChannel = new Mat();
        padded.convertTo(floatChannel, CvType.CV_32F);
        padded.release();

        // Create complex image with zero imaginary part
        Mat complexI = new Mat();
        List<Mat> planes = new ArrayList<>();
        planes.add(floatChannel);
        planes.add(Mat.zeros(floatChannel.size(), CvType.CV_32F));
        Core.merge(planes, complexI);
        planes.get(1).release();
        floatChannel.release();

        // Compute DFT
        Core.dft(complexI, complexI);

        // Shift zero frequency to center
        fftShift(complexI);

        // Create and apply mask
        Mat mask = createFFTMask(optRows, optCols, radius, smoothness);

        // Split, multiply each plane by mask, merge back
        List<Mat> dftPlanes = new ArrayList<>();
        Core.split(complexI, dftPlanes);
        complexI.release();

        Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
        Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));
        mask.release();

        Mat maskedDft = new Mat();
        Core.merge(dftPlanes, maskedDft);
        for (Mat p : dftPlanes) p.release();

        // Inverse shift
        fftShift(maskedDft);

        // Inverse DFT
        Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

        // Get real part
        List<Mat> idftPlanes = new ArrayList<>();
        Core.split(maskedDft, idftPlanes);
        maskedDft.release();

        Mat magnitude = idftPlanes.get(0);
        idftPlanes.get(1).release();

        // Crop to original size
        Mat cropped = new Mat(magnitude, new Rect(0, 0, origCols, origRows));
        Mat result = cropped.clone();
        magnitude.release();

        // Clip to 0-255 and convert to 8-bit
        Core.min(result, new Scalar(255), result);
        Core.max(result, new Scalar(0), result);

        Mat output = new Mat();
        result.convertTo(output, CvType.CV_8U);
        result.release();

        return output;
    }

    private void fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        Mat q0 = new Mat(input, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(input, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(input, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(input, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        tmp.release();
    }

    private Mat createFFTMask(int rows, int cols, int radius, int smoothness) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (radius == 0) {
            mask.setTo(new Scalar(1.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            // Hard circle mask
            mask.setTo(new Scalar(1.0));
            Imgproc.circle(mask, new Point(ccol, crow), radius, new Scalar(0.0), -1);
        } else {
            // Butterworth filter
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
                    double ratio = effectiveCutoff / (distance + BUTTERWORTH_DIVISION_EPSILON);
                    float value = (float) (1.0 / (1.0 + Math.pow(ratio, twoN)));
                    maskData[y * cols + x] = value;
                }
            }
            mask.put(0, 0, maskData);
        }

        return mask;
    }

    // ====== Bit Planes Color Implementation ======

    private ImageProcessor createBitPlanesColorProcessor(FXNode fxNode) {
        return input -> {
            if (input == null || input.empty()) return input;

            // Read properties from FXNode
            boolean[][] bitEnabled = new boolean[3][8];
            double[][] bitGain = new double[3][8];

            // Initialize defaults
            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < 8; i++) {
                    bitEnabled[c][i] = true;
                    bitGain[c][i] = 1.0;
                }
            }

            // Load from properties if present
            String[] channelNames = {"red", "green", "blue"};
            for (int c = 0; c < 3; c++) {
                String enabledKey = channelNames[c] + "BitEnabled";
                String gainKey = channelNames[c] + "BitGain";

                if (fxNode.properties.containsKey(enabledKey)) {
                    boolean[] arr = (boolean[]) fxNode.properties.get(enabledKey);
                    for (int i = 0; i < Math.min(arr.length, 8); i++) {
                        bitEnabled[c][i] = arr[i];
                    }
                }
                if (fxNode.properties.containsKey(gainKey)) {
                    double[] arr = (double[]) fxNode.properties.get(gainKey);
                    for (int i = 0; i < Math.min(arr.length, 8); i++) {
                        bitGain[c][i] = arr[i];
                    }
                }
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

                // Process each channel (BGR order in OpenCV)
                // channels: 0=Blue, 1=Green, 2=Red
                int[] channelMap = {2, 1, 0}; // Red, Green, Blue -> BGR indices

                // Initialize result channels list
                for (int c = 0; c < 3; c++) {
                    resultChannels.add(null);
                }

                for (int colorIdx = 0; colorIdx < 3; colorIdx++) {
                    int bgrIdx = channelMap[colorIdx];
                    Mat channel = channels.get(bgrIdx);

                    // Get channel data
                    byte[] channelData = new byte[channel.rows() * channel.cols()];
                    channel.get(0, 0, channelData);

                    float[] resultData = new float[channelData.length];

                    // Process each bit plane
                    for (int i = 0; i < 8; i++) {
                        if (!bitEnabled[colorIdx][i]) {
                            continue;
                        }

                        // Extract bit plane (bit 7-i, since i=0 is MSB)
                        int bitIndex = 7 - i;

                        for (int j = 0; j < channelData.length; j++) {
                            int pixelValue = channelData[j] & 0xFF;
                            int bit = (pixelValue >> bitIndex) & 1;
                            // Scale to original bit weight and apply gain
                            resultData[j] += bit * (1 << bitIndex) * (float) bitGain[colorIdx][i];
                        }
                    }

                    // Clip to valid range [0, 255]
                    for (int j = 0; j < resultData.length; j++) {
                        resultData[j] = Math.max(0, Math.min(255, resultData[j]));
                    }

                    // Convert to 8-bit
                    Mat resultMat = new Mat(channel.rows(), channel.cols(), CvType.CV_32F);
                    Mat result8u = new Mat();
                    try {
                        resultMat.put(0, 0, resultData);
                        resultMat.convertTo(result8u, CvType.CV_8U);
                        resultChannels.set(bgrIdx, result8u);
                    } finally {
                        resultMat.release();
                    }
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
        };
    }

    // ====== FFT Low-Pass Filter Implementation ======

    private ImageProcessor createFFTLowPassProcessor(FXNode fxNode) {
        return input -> {
            if (input == null || input.empty()) return input;

            // Read properties from FXNode
            int radius = 100;  // Default is higher for low-pass
            int smoothness = 0;
            if (fxNode.properties.containsKey("radius")) {
                radius = ((Number) fxNode.properties.get("radius")).intValue();
            }
            if (fxNode.properties.containsKey("smoothness")) {
                smoothness = ((Number) fxNode.properties.get("smoothness")).intValue();
            }

            // Split into BGR channels
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            try {
                // Apply FFT filter to each channel
                List<Mat> filteredChannels = new ArrayList<>();
                try {
                    for (Mat channel : channels) {
                        Mat filtered = applyFFTLowPassToChannel(channel, radius, smoothness);
                        filteredChannels.add(filtered);
                    }

                    // Merge filtered channels
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
        };
    }

    private Mat applyFFTLowPassToChannel(Mat channel, int radius, int smoothness) {
        int origRows = channel.rows();
        int origCols = channel.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Pad image to optimal size
        Mat padded = new Mat();
        Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
            Core.BORDER_CONSTANT, Scalar.all(0));

        // Convert to float
        Mat floatChannel = new Mat();
        padded.convertTo(floatChannel, CvType.CV_32F);
        padded.release();

        // Create complex image with zero imaginary part
        Mat complexI = new Mat();
        List<Mat> planes = new ArrayList<>();
        planes.add(floatChannel);
        planes.add(Mat.zeros(floatChannel.size(), CvType.CV_32F));
        Core.merge(planes, complexI);
        planes.get(1).release();
        floatChannel.release();

        // Compute DFT
        Core.dft(complexI, complexI);

        // Shift zero frequency to center
        fftShift(complexI);

        // Create and apply LOW-PASS mask (inverted from high-pass)
        Mat mask = createFFTLowPassMask(optRows, optCols, radius, smoothness);

        // Split, multiply each plane by mask, merge back
        List<Mat> dftPlanes = new ArrayList<>();
        Core.split(complexI, dftPlanes);
        complexI.release();

        Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
        Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));
        mask.release();

        Mat maskedDft = new Mat();
        Core.merge(dftPlanes, maskedDft);
        for (Mat p : dftPlanes) p.release();

        // Inverse shift
        fftShift(maskedDft);

        // Inverse DFT
        Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

        // Get real part
        List<Mat> idftPlanes = new ArrayList<>();
        Core.split(maskedDft, idftPlanes);
        maskedDft.release();

        Mat magnitude = idftPlanes.get(0);
        idftPlanes.get(1).release();

        // Crop to original size
        Mat cropped = new Mat(magnitude, new Rect(0, 0, origCols, origRows));
        Mat result = cropped.clone();
        magnitude.release();

        // Clip to 0-255 and convert to 8-bit
        Core.min(result, new Scalar(255), result);
        Core.max(result, new Scalar(0), result);

        Mat output = new Mat();
        result.convertTo(output, CvType.CV_8U);
        result.release();

        return output;
    }

    private Mat createFFTLowPassMask(int rows, int cols, int radius, int smoothness) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (radius == 0) {
            // No filtering - block everything
            mask.setTo(new Scalar(0.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            // Hard circle mask - filled circle passes, outside blocks
            mask.setTo(new Scalar(0.0));
            Imgproc.circle(mask, new Point(ccol, crow), radius, new Scalar(1.0), -1);
        } else {
            // Butterworth low-pass filter
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
                    // Low-pass: ratio is distance/cutoff (opposite of high-pass)
                    double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
                    float value = (float) (1.0 / (1.0 + Math.pow(ratio, twoN)));
                    maskData[y * cols + x] = value;
                }
            }
            mask.put(0, 0, maskData);
        }

        return mask;
    }

    // ====== Bit Planes Grayscale Implementation ======

    private ImageProcessor createBitPlanesGrayscaleProcessor(FXNode fxNode) {
        return input -> {
            if (input == null || input.empty()) return input;

            // Read properties from FXNode
            boolean[] bitEnabled = new boolean[8];
            double[] bitGain = new double[8];

            // Initialize defaults (all enabled, gain 1.0)
            for (int i = 0; i < 8; i++) {
                bitEnabled[i] = true;
                bitGain[i] = 1.0;
            }

            // Load from properties if present
            if (fxNode.properties.containsKey("bitEnabled")) {
                boolean[] arr = (boolean[]) fxNode.properties.get("bitEnabled");
                for (int i = 0; i < Math.min(arr.length, 8); i++) {
                    bitEnabled[i] = arr[i];
                }
            }
            if (fxNode.properties.containsKey("bitGain")) {
                double[] arr = (double[]) fxNode.properties.get("bitGain");
                for (int i = 0; i < Math.min(arr.length, 8); i++) {
                    bitGain[i] = arr[i];
                }
            }

            Mat gray = null;

            try {
                // Convert to grayscale if needed
                gray = new Mat();
                if (input.channels() == 3) {
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                } else {
                    gray = input.clone();
                }

                // Get grayscale data
                byte[] grayData = new byte[gray.rows() * gray.cols()];
                gray.get(0, 0, grayData);

                float[] resultData = new float[grayData.length];

                // Process each bit plane
                for (int i = 0; i < 8; i++) {
                    if (!bitEnabled[i]) {
                        continue;
                    }

                    // Extract bit plane (bit 7-i, since i=0 is MSB)
                    int bitIndex = 7 - i;

                    for (int j = 0; j < grayData.length; j++) {
                        int pixelValue = grayData[j] & 0xFF;
                        int bit = (pixelValue >> bitIndex) & 1;
                        // Scale to original bit weight and apply gain
                        resultData[j] += bit * (1 << bitIndex) * (float) bitGain[i];
                    }
                }

                // Clip to valid range [0, 255]
                for (int j = 0; j < resultData.length; j++) {
                    resultData[j] = Math.max(0, Math.min(255, resultData[j]));
                }

                // Convert to 8-bit grayscale
                Mat resultMat = new Mat(gray.rows(), gray.cols(), CvType.CV_32F);
                resultMat.put(0, 0, resultData);

                Mat result8u = new Mat();
                resultMat.convertTo(result8u, CvType.CV_8U);
                resultMat.release();

                // Convert back to BGR for display
                Mat output = new Mat();
                Imgproc.cvtColor(result8u, output, Imgproc.COLOR_GRAY2BGR);
                result8u.release();

                return output;
            } finally {
                if (gray != null) gray.release();
            }
        };
    }
}
