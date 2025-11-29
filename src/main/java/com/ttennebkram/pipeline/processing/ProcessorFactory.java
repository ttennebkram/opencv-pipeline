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
                    Mat output = new Mat();
                    input.convertTo(output, -1, 1.5, 20);
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
}
