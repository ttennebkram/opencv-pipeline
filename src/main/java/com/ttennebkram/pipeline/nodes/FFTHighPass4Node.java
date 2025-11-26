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
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;

/**
 * FFT High-Pass Filter node with 4 outputs:
 * 1. Filtered image (high frequencies preserved)
 * 2. Absolute difference (input - filtered) = blocked low frequencies
 * 3. FFT spectrum visualization (same resolution as input)
 * 4. Filter cutoff curve visualization
 */
@NodeInfo(name = "FFTHighPass4", category = "Filter", aliases = {"FFT High-Pass 4-Output"})
public class FFTHighPass4Node extends MultiOutputNode {
    private int radius = 30;
    private int smoothness = 0;

    // Butterworth filter constants
    private static final double BUTTERWORTH_ORDER_MAX = 10.0;
    private static final double BUTTERWORTH_ORDER_MIN = 0.5;
    private static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    private static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    private static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    private static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    public FFTHighPass4Node(Display display, Shell shell, int x, int y) {
        super(display, shell, "FFT HighPass 4", x, y);
        initMultiOutputQueues(4); // Fixed 4 outputs
        setOutputLabels("Filtered", "Difference", "Spectrum", "Filter Curve");
    }

    // Custom colors for high-pass filter (light orange)
    @Override
    protected org.eclipse.swt.graphics.Color getBackgroundColor() {
        return new org.eclipse.swt.graphics.Color(255, 235, 220);
    }

    @Override
    protected org.eclipse.swt.graphics.Color getBorderColor() {
        return new org.eclipse.swt.graphics.Color(200, 120, 70);
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
        }, "FFTHighPass4-" + name + "-Thread");
        processingThread.setPriority(threadPriority);
        processingThread.start();
    }

    /**
     * Process input and return 4 outputs:
     * [0] = filtered image (high-pass)
     * [1] = absolute difference (blocked low frequencies)
     * [2] = FFT spectrum
     * [3] = filter visualization
     */
    private Mat[] processFFT(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return new Mat[] { input.clone(), null, null, null };
        }

        int rows = input.rows();
        int cols = input.cols();

        // Create the HIGH-PASS filter mask (shared across all channels)
        Mat mask = createHighPassMask(rows, cols);

        // Create filter curve visualization (line graph)
        Mat filterVis = createFilterCurveVisualization(cols, rows);

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

                // Convert to float
                Mat floatChannel = new Mat();
                channel.convertTo(floatChannel, CvType.CV_32F);

                // Compute DFT
                Mat dft = new Mat();
                Core.dft(floatChannel, dft, Core.DFT_COMPLEX_OUTPUT);

                // Shift zero frequency to center
                Mat dftShift = fftShift(dft.clone());

                // For spectrum visualization, use the first channel (or accumulate)
                if (c == 0) {
                    spectrumAccum = createSpectrumVisualization(dftShift, rows, cols);
                }

                // Apply mask to DFT
                Mat maskedDft = applyMask(dftShift, mask);

                // Inverse shift
                Mat dftIshift = ifftShift(maskedDft);

                // Inverse DFT
                Mat idft = new Mat();
                Core.idft(dftIshift, idft, Core.DFT_SCALE);

                // Get magnitude (filtered result)
                List<Mat> idftPlanes = new ArrayList<>();
                Core.split(idft, idftPlanes);
                Mat magnitude = new Mat();
                Core.magnitude(idftPlanes.get(0), idftPlanes.get(1), magnitude);

                // Clip and convert to 8-bit
                Core.min(magnitude, new Scalar(255), magnitude);
                Core.max(magnitude, new Scalar(0), magnitude);
                Mat filteredChannel = new Mat();
                magnitude.convertTo(filteredChannel, CvType.CV_8U);

                filteredChannels.add(filteredChannel);

                // Release intermediate Mats
                floatChannel.release();
                dft.release();
                dftShift.release();
                maskedDft.release();
                dftIshift.release();
                idft.release();
                magnitude.release();
                for (Mat plane : idftPlanes) plane.release();
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
            Mat floatInput = new Mat();
            input.convertTo(floatInput, CvType.CV_32F);

            Mat dft = new Mat();
            Core.dft(floatInput, dft, Core.DFT_COMPLEX_OUTPUT);

            Mat dftShift = fftShift(dft.clone());

            spectrum = createSpectrumVisualization(dftShift, rows, cols);

            Mat maskedDft = applyMask(dftShift, mask);
            Mat dftIshift = ifftShift(maskedDft);

            Mat idft = new Mat();
            Core.idft(dftIshift, idft, Core.DFT_SCALE);

            List<Mat> idftPlanes = new ArrayList<>();
            Core.split(idft, idftPlanes);
            Mat magnitude = new Mat();
            Core.magnitude(idftPlanes.get(0), idftPlanes.get(1), magnitude);

            Core.min(magnitude, new Scalar(255), magnitude);
            Core.max(magnitude, new Scalar(0), magnitude);
            filtered = new Mat();
            magnitude.convertTo(filtered, CvType.CV_8U);

            // Release intermediate Mats
            floatInput.release();
            dft.release();
            dftShift.release();
            maskedDft.release();
            dftIshift.release();
            idft.release();
            magnitude.release();
            for (Mat plane : idftPlanes) plane.release();
        }

        // Calculate absolute difference (shows blocked low frequencies)
        Mat difference = new Mat();
        Core.absdiff(input, filtered, difference);

        mask.release();

        return new Mat[] { filtered, difference, spectrum, filterVis };
    }

    private Mat createSpectrumVisualization(Mat dftShift, int rows, int cols) {
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

        // Convert to BGR for display
        Mat spectrumBGR = new Mat();
        org.opencv.imgproc.Imgproc.cvtColor(spectrum, spectrumBGR, org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR);

        // Cleanup
        for (Mat p : planes) p.release();
        magnitude.release();
        logMag.release();
        spectrum.release();

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
     * Compute filter value for a given distance (for HIGH-PASS: 0 near center, 1 at edges)
     */
    private double computeFilterValue(double distance) {
        if (radius == 0) {
            return 1.0; // Pass everything when radius is 0
        }

        if (smoothness == 0) {
            // Hard cutoff: block inside radius, pass outside
            return distance <= radius ? 0.0 : 1.0;
        } else {
            // Butterworth high-pass filter
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) order = BUTTERWORTH_ORDER_MIN;

            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;

            // High-pass: 1 / (1 + (cutoff/distance)^2n)
            double ratio = effectiveCutoff / (distance + BUTTERWORTH_DIVISION_EPSILON);
            double value = 1.0 / (1.0 + Math.pow(ratio, 2 * order));
            return Math.max(0.0, Math.min(1.0, value));
        }
    }

    private Mat applyMask(Mat dftShift, Mat mask) {
        List<Mat> planes = new ArrayList<>();
        Core.split(dftShift, planes);

        int rows = mask.rows();
        int cols = mask.cols();
        float[] maskData = new float[rows * cols];
        mask.get(0, 0, maskData);

        for (Mat plane : planes) {
            float[] planeData = new float[rows * cols];
            plane.get(0, 0, planeData);
            for (int i = 0; i < planeData.length; i++) {
                planeData[i] *= maskData[i];
            }
            plane.put(0, 0, planeData);
        }

        Mat result = new Mat();
        Core.merge(planes, result);
        return result;
    }

    private Mat fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        Mat q0 = input.submat(0, cy, 0, cx);
        Mat q1 = input.submat(0, cy, cx, input.cols());
        Mat q2 = input.submat(cy, input.rows(), 0, cx);
        Mat q3 = input.submat(cy, input.rows(), cx, input.cols());

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        tmp.release();
        return input;
    }

    private Mat ifftShift(Mat input) {
        return fftShift(input);
    }

    /**
     * Create HIGH-PASS mask - blocks low frequencies (center), passes high frequencies (edges).
     * This is the inverse of low-pass: 0 at center, 1 at edges.
     */
    private Mat createHighPassMask(int rows, int cols) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (radius == 0) {
            // radius=0 means pass everything
            mask.setTo(new Scalar(1.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        float[] maskData = new float[rows * cols];

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                double distance = Math.sqrt((x - ccol) * (x - ccol) + (y - crow) * (y - crow));

                float value;
                if (smoothness == 0) {
                    // Hard cutoff: block inside radius, pass outside
                    value = distance <= radius ? 0.0f : 1.0f;
                } else {
                    // Butterworth high-pass filter
                    double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
                    if (order < BUTTERWORTH_ORDER_MIN) order = BUTTERWORTH_ORDER_MIN;

                    double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
                    double effectiveCutoff = radius * shiftFactor;

                    // High-pass: 1 / (1 + (cutoff/distance)^2n)
                    double ratio = effectiveCutoff / (distance + BUTTERWORTH_DIVISION_EPSILON);
                    value = (float) (1.0 / (1.0 + Math.pow(ratio, 2 * order)));
                    value = Math.max(0.0f, Math.min(1.0f, value));
                }

                maskData[y * cols + x] = value;
            }
        }

        mask.put(0, 0, maskData);
        return mask;
    }

    @Override
    public String getDescription() {
        return "FFT High-Pass Filter (4 outputs)\n1:Filtered 2:Diff 3:Spectrum 4:Filter";
    }

    @Override
    public String getDisplayName() {
        return "FFT HighPass 4";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("FFT High-Pass Filter Properties");
        dialog.setLayout(new GridLayout(3, false));

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
        GridData radiusLabelGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        radiusLabelGd.widthHint = 30;
        radiusLabel.setLayoutData(radiusLabelGd);
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
        GridData smoothnessLabelGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
        smoothnessLabelGd.widthHint = 30;
        smoothnessLabel.setLayoutData(smoothnessLabelGd);
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
        json.addProperty("radius", radius);
        json.addProperty("smoothness", smoothness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("radius")) radius = json.get("radius").getAsInt();
        if (json.has("smoothness")) smoothness = json.get("smoothness").getAsInt();
    }
}
