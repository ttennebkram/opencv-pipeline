package com.ttennebkram.pipeline.processing;

import com.ttennebkram.pipeline.fx.FXNode;
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
                // Debug: trace which FXNode is being updated
                if (fxNode.nodeType.contains("Invert") ||
                    "ContainerInput".equals(fxNode.nodeType) ||
                    "ContainerOutput".equals(fxNode.nodeType)) {
                    System.out.println("[ProcessorFactory] Callback fired for " + fxNode.label +
                                       " (id=" + fxNode.id + ", type=" + fxNode.nodeType +
                                       ", hashCode=" + System.identityHashCode(fxNode) + ")");
                }
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

        System.out.println("[ProcessorFactory] wireConnection: " + source.label + " (id=" + source.id + ") -> " + target.label + " (id=" + target.id + ")");
        System.out.println("[ProcessorFactory]   sourceProc=" + (sourceProc != null ? sourceProc.getName() : "NULL") +
                           ", targetProc=" + (targetProc != null ? targetProc.getName() : "NULL") +
                           ", sourceOutputIndex=" + sourceOutputIndex + ", targetInputIndex=" + targetInputIndex);

        if (sourceProc == null || targetProc == null) {
            System.err.println("[ProcessorFactory]   WARNING: Cannot wire, processor is null!");
            return;
        }

        // Create queue for this connection
        BlockingQueue<Mat> queue = new java.util.concurrent.LinkedBlockingQueue<>();

        // Wire output of source to input of target
        System.out.println("[ProcessorFactory]   Wiring: setting " + sourceProc.getName() + ".outputQueue" +
                           (sourceOutputIndex == 1 ? "2" : "") + " and " +
                           targetProc.getName() + ".inputQueue" + (targetInputIndex == 1 ? "2" : ""));

        // Set source output queue (support dual output)
        if (sourceOutputIndex == 1) {
            sourceProc.setOutputQueue2(queue);
        } else {
            sourceProc.setOutputQueue(queue);
        }

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
        System.out.println("[ProcessorFactory]   After wiring: " + targetProc.getName() + ".inputQueue=" +
                           (targetProc.getInputQueue() != null ? "set" : "NULL"));
    }

    /**
     * Start all processors.
     */
    public void startAll() {
        System.out.println("[ProcessorFactory] startAll: starting " + processors.size() + " processors");
        for (ThreadedProcessor tp : processors.values()) {
            System.out.println("[ProcessorFactory]   Starting " + tp.getName() + " (" + tp.getClass().getSimpleName() + ")");
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
     * Sync stats from processors back to FXNodes.
     * Updates input/output counters, thread priority, work units, and effective FPS.
     */
    public void syncStats(FXNode fxNode) {
        ThreadedProcessor tp = processors.get(fxNode.id);
        if (tp != null) {
            fxNode.inputCount = (int) tp.getInputReads1();
            fxNode.outputCount1 = (int) tp.getOutputWrites1();
            // Sync backpressure stats for display
            fxNode.threadPriority = tp.getThreadPriority();
            fxNode.workUnitsCompleted = tp.getWorkUnitsCompleted();
            fxNode.effectiveFps = tp.getEffectiveFps();
        }
    }

    /**
     * Create an ImageProcessor for the given node type.
     */
    private ImageProcessor createImageProcessor(FXNode fxNode) {
        String type = fxNode.nodeType;

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
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat output = new Mat();
                    Imgproc.threshold(gray, output, 127, 255, Imgproc.THRESH_BINARY);
                    gray.release();
                    Mat bgr = new Mat();
                    Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
                    output.release();
                    return bgr;
                };

            case "GaussianBlur":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Imgproc.GaussianBlur(input, output, new Size(15, 15), 0);
                    return output;
                };

            case "MedianBlur":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Imgproc.medianBlur(input, output, 5);
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
                    Mat output = new Mat();
                    Imgproc.blur(input, output, new Size(5, 5));
                    return output;
                };

            case "CannyEdge":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat gray = new Mat();
                    if (input.channels() == 3) {
                        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                    } else {
                        gray = input.clone();
                    }
                    Mat output = new Mat();
                    Imgproc.Canny(gray, output, 100, 200);
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
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
                    Imgproc.erode(input, output, kernel);
                    kernel.release();
                    return output;
                };

            case "Dilate":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
                    Imgproc.dilate(input, output, kernel);
                    kernel.release();
                    return output;
                };

            case "MorphOpen":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
                    Imgproc.morphologyEx(input, output, Imgproc.MORPH_OPEN, kernel);
                    kernel.release();
                    return output;
                };

            case "MorphClose":
                return input -> {
                    if (input == null || input.empty()) return input;
                    Mat output = new Mat();
                    Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(5, 5));
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

            case "BitPlanesColor":
                return createBitPlanesColorProcessor(fxNode);

            case "FFTLowPass":
                return createFFTLowPassProcessor(fxNode);

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
     */
    private DualImageProcessor createDualImageProcessor(FXNode fxNode) {
        String type = fxNode.nodeType;

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
