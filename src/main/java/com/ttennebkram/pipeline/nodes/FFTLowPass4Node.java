package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;

/**
 * FFT Low-Pass Filter node with 4 outputs:
 * 1. Filtered image
 * 2. Absolute difference (input - filtered)
 * 3. FFT spectrum visualization (same resolution as input)
 * 4. Filter cutoff curve visualization
 */
@NodeInfo(name = "FFTLowPass4", category = "Filter", aliases = {"FFT Low-Pass 4-Output"})
public class FFTLowPass4Node extends MultiOutputNode {
    private int radius = 100;
    private int smoothness = 0;

    // Butterworth filter constants
    private static final double BUTTERWORTH_ORDER_MAX = 10.0;
    private static final double BUTTERWORTH_ORDER_MIN = 0.5;
    private static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    private static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    private static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    private static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    public FFTLowPass4Node(Display display, Shell shell, int x, int y) {
        super(display, shell, "FFT LowPass 4", x, y);
        initMultiOutputQueues(4); // Fixed 4 outputs
        setOutputLabels("Filtered", "Difference", "Spectrum", "Filter Curve");
    }

    // Custom colors for low-pass filter (light blue)
    @Override
    protected org.eclipse.swt.graphics.Color getBackgroundColor() {
        return new org.eclipse.swt.graphics.Color(220, 240, 255);
    }

    @Override
    protected org.eclipse.swt.graphics.Color getBorderColor() {
        return new org.eclipse.swt.graphics.Color(70, 130, 180);
    }

    // Getters/setters for serialization
    public int getRadius() { return radius; }
    public void setRadius(int v) { radius = v; }
    public int getSmoothness() { return smoothness; }
    public void setSmoothness(int v) { smoothness = v; }

    @Override
    public Mat process(Mat input) {
        // Not used - we override startProcessing instead
        return input;
    }

    @Override
    public void startProcessing() {
        if (running.get()) {
            return;
        }

        running.set(true);
        workUnitsCompleted = 0;

        processingThread = new Thread(() -> {
            while (running.get()) {
                try {
                    if (inputQueue == null) {
                        Thread.sleep(100);
                        continue;
                    }

                    Mat input = inputQueue.take();
                    incrementInputReads1();
                    if (input == null || input.empty()) {
                        continue;
                    }

                    // Process and get all 4 outputs
                    Mat[] outputs = processFFT(input);
                    Mat filtered = outputs[0];
                    Mat difference = outputs[1];
                    Mat spectrum = outputs[2];
                    Mat filterVis = outputs[3];

                    incrementWorkUnits();

                    // Send to output 1: filtered image
                    BlockingQueue<Mat> queue0 = getMultiOutputQueue(0);
                    if (queue0 != null && filtered != null) {
                        queue0.put(filtered.clone());
                    }

                    // Send to output 2: difference
                    BlockingQueue<Mat> queue1 = getMultiOutputQueue(1);
                    if (queue1 != null && difference != null) {
                        queue1.put(difference.clone());
                    }

                    // Send to output 3: spectrum
                    BlockingQueue<Mat> queue2 = getMultiOutputQueue(2);
                    if (queue2 != null && spectrum != null) {
                        queue2.put(spectrum.clone());
                    }

                    // Send to output 4: filter visualization
                    BlockingQueue<Mat> queue3 = getMultiOutputQueue(3);
                    if (queue3 != null && filterVis != null) {
                        queue3.put(filterVis.clone());
                    }

                    // Set primary output for thumbnail
                    if (filtered != null) {
                        setOutputMat(filtered.clone());
                        notifyFrame(filtered.clone());
                    }

                    // Release all
                    if (filtered != null) filtered.release();
                    if (difference != null) difference.release();
                    if (spectrum != null) spectrum.release();
                    if (filterVis != null) filterVis.release();

                    checkBackpressure();
                    input.release();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }, "FFTLowPass4-" + name + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Get next power of 2 >= n
     */
    private int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    /**
     * Get optimal FFT size - prefer power of 2 if overhead is small,
     * otherwise use OpenCV's optimal size (products of 2, 3, 5)
     */
    private int getOptimalFFTSize(int n) {
        int pow2 = nextPowerOf2(n);
        int optimal = Core.getOptimalDFTSize(n);

        // Use power of 2 if it adds less than 25% overhead
        if (pow2 <= optimal * 1.25) {
            return pow2;
        }
        // Otherwise use OpenCV's optimal (but ensure even for fftShift)
        return optimal % 2 == 0 ? optimal : optimal + 1;
    }

    /**
     * Process input and return 4 outputs:
     * [0] = filtered image
     * [1] = absolute difference
     * [2] = FFT spectrum
     * [3] = filter visualization
     */
    private Mat[] processFFT(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return new Mat[] { input.clone(), null, null, null };
        }

        int origRows = input.rows();
        int origCols = input.cols();

        // Get optimal size for FFT (prefers power-of-2 when efficient)
        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Create the filter mask at optimal size
        Mat mask = createMask(optRows, optCols);

        // Create filter curve visualization (line graph) at original size
        Mat filterVis = createFilterCurveVisualization(origCols, origRows);

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
                    spectrumAccum = createSpectrumVisualization(dft, origRows, origCols);
                }

                // Apply mask using Core.multiply (optimized)
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
            // Single channel (grayscale) processing
            // Pad to optimal size
            Mat padded = new Mat();
            Core.copyMakeBorder(input, padded, 0, optRows - origRows, 0, optCols - origCols,
                Core.BORDER_CONSTANT, Scalar.all(0));

            Mat floatInput = new Mat();
            padded.convertTo(floatInput, CvType.CV_32F);
            padded.release();

            Mat dft = new Mat();
            Core.dft(floatInput, dft, Core.DFT_COMPLEX_OUTPUT);
            floatInput.release();

            // Shift zero frequency to center
            fftShift(dft);

            spectrum = createSpectrumVisualization(dft, origRows, origCols);

            // Apply mask using Core.multiply (optimized)
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
            filtered = new Mat();
            result.convertTo(filtered, CvType.CV_8U);
            result.release();
        }

        // Calculate absolute difference
        Mat difference = new Mat();
        Core.absdiff(input, filtered, difference);

        mask.release();

        return new Mat[] { filtered, difference, spectrum, filterVis };
    }

    private Mat createSpectrumVisualization(Mat dftShift, int origRows, int origCols) {
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

        // Cleanup
        for (Mat p : planes) p.release();
        magnitude.release();
        logMag.release();
        spectrum.release();
        croppedClone.release();

        return spectrumBGR;
    }

    /**
     * Create a line graph visualization of the filter curve.
     * X-axis: Distance from center (pixels)
     * Y-axis: Filter response (0=blocked, 1=passed)
     */
    private Mat createFilterCurveVisualization(int width, int height) {
        // Create a black background image
        Mat vis = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));

        // Graph margins
        int marginLeft = 50;
        int marginRight = 20;
        int marginTop = 30;
        int marginBottom = 40;

        int graphWidth = width - marginLeft - marginRight;
        int graphHeight = height - marginTop - marginBottom;

        // Draw grid lines (gray)
        Scalar gridColor = new Scalar(60, 60, 60);
        // Horizontal grid lines at 0.0, 0.25, 0.5, 0.75, 1.0
        for (int i = 0; i <= 4; i++) {
            int yPos = marginTop + (int) (graphHeight * (1.0 - i / 4.0));
            org.opencv.imgproc.Imgproc.line(vis,
                new org.opencv.core.Point(marginLeft, yPos),
                new org.opencv.core.Point(width - marginRight, yPos),
                gridColor, 1);
        }
        // Vertical grid lines every 50 pixels of distance
        int maxDistance = 200;
        for (int d = 0; d <= maxDistance; d += 50) {
            int xPos = marginLeft + (int) (graphWidth * d / (double) maxDistance);
            org.opencv.imgproc.Imgproc.line(vis,
                new org.opencv.core.Point(xPos, marginTop),
                new org.opencv.core.Point(xPos, marginTop + graphHeight),
                gridColor, 1);
        }

        // Draw axes (white)
        Scalar axisColor = new Scalar(255, 255, 255);
        // Y-axis
        org.opencv.imgproc.Imgproc.line(vis,
            new org.opencv.core.Point(marginLeft, marginTop),
            new org.opencv.core.Point(marginLeft, marginTop + graphHeight),
            axisColor, 2);
        // X-axis
        org.opencv.imgproc.Imgproc.line(vis,
            new org.opencv.core.Point(marginLeft, marginTop + graphHeight),
            new org.opencv.core.Point(width - marginRight, marginTop + graphHeight),
            axisColor, 2);

        // Draw the filter curve (blue)
        Scalar curveColor = new Scalar(255, 100, 100); // BGR - light blue
        org.opencv.core.Point prevPoint = null;
        for (int i = 0; i <= graphWidth; i++) {
            double distance = (i / (double) graphWidth) * maxDistance;
            double filterValue = computeFilterValue(distance);

            int xPos = marginLeft + i;
            int yPos = marginTop + (int) (graphHeight * (1.0 - filterValue));

            org.opencv.core.Point currentPoint = new org.opencv.core.Point(xPos, yPos);
            if (prevPoint != null) {
                org.opencv.imgproc.Imgproc.line(vis, prevPoint, currentPoint, curveColor, 2);
            }
            prevPoint = currentPoint;
        }

        // Draw vertical line at radius (red dashed - we'll use solid for simplicity)
        if (radius > 0 && radius <= maxDistance) {
            int radiusX = marginLeft + (int) (graphWidth * radius / (double) maxDistance);
            Scalar radiusColor = new Scalar(0, 0, 255); // BGR - red
            org.opencv.imgproc.Imgproc.line(vis,
                new org.opencv.core.Point(radiusX, marginTop),
                new org.opencv.core.Point(radiusX, marginTop + graphHeight),
                radiusColor, 1);
        }

        // Draw labels (using putText)
        Scalar textColor = new Scalar(200, 200, 200);
        double fontScale = 0.4;
        int fontFace = org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;

        // Title
        org.opencv.imgproc.Imgproc.putText(vis, "Filter Response Curve",
            new org.opencv.core.Point(marginLeft + 10, 20), fontFace, fontScale, textColor, 1);

        // Y-axis labels
        org.opencv.imgproc.Imgproc.putText(vis, "1.0",
            new org.opencv.core.Point(5, marginTop + 5), fontFace, fontScale, textColor, 1);
        org.opencv.imgproc.Imgproc.putText(vis, "0.5",
            new org.opencv.core.Point(5, marginTop + graphHeight / 2 + 5), fontFace, fontScale, textColor, 1);
        org.opencv.imgproc.Imgproc.putText(vis, "0.0",
            new org.opencv.core.Point(5, marginTop + graphHeight + 5), fontFace, fontScale, textColor, 1);

        // X-axis labels
        org.opencv.imgproc.Imgproc.putText(vis, "0",
            new org.opencv.core.Point(marginLeft - 5, height - 10), fontFace, fontScale, textColor, 1);
        org.opencv.imgproc.Imgproc.putText(vis, "100",
            new org.opencv.core.Point(marginLeft + graphWidth / 2 - 10, height - 10), fontFace, fontScale, textColor, 1);
        org.opencv.imgproc.Imgproc.putText(vis, "200",
            new org.opencv.core.Point(width - marginRight - 15, height - 10), fontFace, fontScale, textColor, 1);

        // Radius label
        if (radius > 0) {
            org.opencv.imgproc.Imgproc.putText(vis, "R=" + radius,
                new org.opencv.core.Point(marginLeft + graphWidth + 5 - 40, marginTop + 15),
                fontFace, fontScale, new Scalar(0, 0, 255), 1);
        }

        return vis;
    }

    /**
     * Compute filter value for a given distance (for LOW-PASS: 1 near center, 0 at edges)
     */
    private double computeFilterValue(double distance) {
        if (radius == 0) {
            return 0.0; // No filtering when radius is 0
        }

        if (smoothness == 0) {
            // Hard cutoff
            return distance <= radius ? 1.0 : 0.0;
        } else {
            // Butterworth low-pass filter
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) order = BUTTERWORTH_ORDER_MIN;

            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;

            double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
            double value = 1.0 / (1.0 + Math.pow(ratio, 2 * order));
            return Math.max(0.0, Math.min(1.0, value));
        }
    }

    private void fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        // Optimal size ensures even dimensions
        Mat q0 = new Mat(input, new Rect(0, 0, cx, cy));      // Top-Left
        Mat q1 = new Mat(input, new Rect(cx, 0, cx, cy));     // Top-Right
        Mat q2 = new Mat(input, new Rect(0, cy, cx, cy));     // Bottom-Left
        Mat q3 = new Mat(input, new Rect(cx, cy, cx, cy));    // Bottom-Right

        // Swap quadrants (Top-Left with Bottom-Right)
        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        // Swap quadrants (Top-Right with Bottom-Left)
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        tmp.release();
    }

    private Mat createMask(int rows, int cols) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (radius == 0) {
            // No filtering - block everything
            mask.setTo(new Scalar(0.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            // Hard circle mask using OpenCV - much faster
            mask.setTo(new Scalar(0.0));
            Imgproc.circle(mask, new org.opencv.core.Point(ccol, crow), radius, new Scalar(1.0), -1);
        } else {
            // Butterworth filter - optimized loop
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

    @Override
    public String getDescription() {
        return "FFT Low-Pass Filter (4 outputs) [Power-of-2]\n1:Filtered 2:Diff 3:Spectrum 4:Filter";
    }

    @Override
    public String getDisplayName() {
        return "FFT LowPass 4";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("FFT Low-Pass Filter Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Node name field
        Text nameText = addNameField(dialog, 3);

        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 3;
        sep.setLayoutData(sepGd);

        // Radius
        new Label(dialog, SWT.NONE).setText("Radius:");
        Scale radiusScale = new Scale(dialog, SWT.HORIZONTAL);
        radiusScale.setMinimum(0);
        radiusScale.setMaximum(200);
        radiusScale.setSelection(radius);
        radiusScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label radiusLabel = new Label(dialog, SWT.NONE);
        radiusLabel.setText(String.valueOf(radius));
        radiusScale.addListener(SWT.Selection, e -> radiusLabel.setText(String.valueOf(radiusScale.getSelection())));

        // Smoothness
        new Label(dialog, SWT.NONE).setText("Smoothness:");
        Scale smoothnessScale = new Scale(dialog, SWT.HORIZONTAL);
        smoothnessScale.setMinimum(0);
        smoothnessScale.setMaximum(100);
        smoothnessScale.setSelection(smoothness);
        smoothnessScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label smoothnessLabel = new Label(dialog, SWT.NONE);
        smoothnessLabel.setText(String.valueOf(smoothness));
        smoothnessScale.addListener(SWT.Selection, e -> smoothnessLabel.setText(String.valueOf(smoothnessScale.getSelection())));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            saveNameField(nameText);
            radius = radiusScale.getSelection();
            smoothness = smoothnessScale.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("radius", radius);
        json.addProperty("smoothness", smoothness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("radius")) radius = json.get("radius").getAsInt();
        if (json.has("smoothness")) smoothness = json.get("smoothness").getAsInt();
    }
}
