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

/**
 * FFT High-Pass Filter node.
 */
@NodeInfo(name = "FFTHighPass", category = "Filter", aliases = {"FFT High-Pass Filter"})
public class FFTHighPassFilterNode extends ProcessingNode {
    private int radius = 0;
    private int smoothness = 0;

    // Butterworth filter constants
    private static final double BUTTERWORTH_ORDER_MAX = 10.0;
    private static final double BUTTERWORTH_ORDER_MIN = 0.5;
    private static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    private static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    private static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    private static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    public FFTHighPassFilterNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "FFT High-Pass Filter", x, y);
    }

    // Getters/setters for serialization
    public int getRadius() { return radius; }
    public void setRadius(int v) { radius = v; }
    public int getSmoothness() { return smoothness; }
    public void setSmoothness(int v) { smoothness = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Split into BGR channels
        List<Mat> channels = new ArrayList<>();
        Core.split(input, channels);

        try {
            // Apply FFT filter to each channel
            List<Mat> filteredChannels = new ArrayList<>();
            try {
                for (Mat channel : channels) {
                    Mat filtered = applyFFTToChannel(channel);
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

    private Mat applyFFTToChannel(Mat channel) {
        int origRows = channel.rows();
        int origCols = channel.cols();

        // Get optimal size for FFT (prefers power-of-2 when efficient)
        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Pad image to power-of-2 size
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

        // Create and apply mask using OpenCV multiply
        Mat mask = createMask(optRows, optCols);

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

        // Create quadrants (power-of-2 ensures even dimensions)
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
            // No filtering - pass everything
            mask.setTo(new Scalar(1.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            // Hard circle mask using OpenCV - much faster
            mask.setTo(new Scalar(1.0));
            Imgproc.circle(mask, new org.opencv.core.Point(ccol, crow), radius, new Scalar(0.0), -1);
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
                    double ratio = effectiveCutoff / (distance + BUTTERWORTH_DIVISION_EPSILON);
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
        return "FFT High-Pass Filter\nCore.dft() / Core.idft() [Power-of-2]";
    }

    @Override
    public String getDisplayName() {
        return "FFT High-Pass";
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

        // Node name field
        Text nameText = addNameField(dialog, 3);

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Separator
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
        // Position dialog near cursor
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
