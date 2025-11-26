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

        // Apply FFT filter to each channel
        List<Mat> filteredChannels = new ArrayList<>();
        for (Mat channel : channels) {
            Mat filtered = applyFFTToChannel(channel);
            filteredChannels.add(filtered);
        }

        // Merge filtered channels
        Mat output = new Mat();
        Core.merge(filteredChannels, output);

        return output;
    }

    private Mat applyFFTToChannel(Mat channel) {
        int rows = channel.rows();
        int cols = channel.cols();

        // Convert to float
        Mat floatChannel = new Mat();
        channel.convertTo(floatChannel, CvType.CV_32F);

        // Compute DFT
        Mat dft = new Mat();
        Core.dft(floatChannel, dft, Core.DFT_COMPLEX_OUTPUT);

        // Shift zero frequency to center
        Mat dftShift = fftShift(dft);

        // Create mask
        Mat mask = createMask(rows, cols);

        // Apply mask
        List<Mat> planes = new ArrayList<>();
        Core.split(dftShift, planes);

        // Extract single channel from mask (it's the same for real and imaginary)
        float[] maskData = new float[rows * cols];
        mask.get(0, 0, maskData);

        // Apply mask to both real and imaginary parts
        for (Mat plane : planes) {
            float[] planeData = new float[rows * cols];
            plane.get(0, 0, planeData);
            for (int i = 0; i < planeData.length; i++) {
                planeData[i] *= maskData[i];
            }
            plane.put(0, 0, planeData);
        }

        Mat maskedDft = new Mat();
        Core.merge(planes, maskedDft);

        // Inverse shift
        Mat dftIshift = ifftShift(maskedDft);

        // Inverse DFT
        Mat idft = new Mat();
        Core.idft(dftIshift, idft, Core.DFT_SCALE);

        // Get magnitude
        List<Mat> idftPlanes = new ArrayList<>();
        Core.split(idft, idftPlanes);
        Mat magnitude = new Mat();
        Core.magnitude(idftPlanes.get(0), idftPlanes.get(1), magnitude);

        // Clip to 0-255 and convert to 8-bit
        Core.min(magnitude, new Scalar(255), magnitude);
        Core.max(magnitude, new Scalar(0), magnitude);

        Mat result = new Mat();
        magnitude.convertTo(result, CvType.CV_8U);

        return result;
    }

    private Mat fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        // Create quadrants
        Mat q0 = input.submat(0, cy, 0, cx);      // Top-Left
        Mat q1 = input.submat(0, cy, cx, input.cols());  // Top-Right
        Mat q2 = input.submat(cy, input.rows(), 0, cx);  // Bottom-Left
        Mat q3 = input.submat(cy, input.rows(), cx, input.cols()); // Bottom-Right

        // Swap quadrants (Top-Left with Bottom-Right)
        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        // Swap quadrants (Top-Right with Bottom-Left)
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        return input;
    }

    private Mat ifftShift(Mat input) {
        // ifftShift is the same as fftShift
        return fftShift(input);
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

        float[] maskData = new float[rows * cols];

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                double distance = Math.sqrt((x - ccol) * (x - ccol) + (y - crow) * (y - crow));

                float value;
                if (smoothness == 0) {
                    // Hard circle mask (high-pass)
                    value = distance <= radius ? 0.0f : 1.0f;
                } else {
                    // Butterworth highpass filter
                    double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
                    if (order < BUTTERWORTH_ORDER_MIN) {
                        order = BUTTERWORTH_ORDER_MIN;
                    }

                    double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
                    double effectiveCutoff = radius * shiftFactor;

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
        return "FFT High-Pass Filter\nnp.fft.fft2() / np.fft.ifft2()";
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
        json.addProperty("radius", radius);
        json.addProperty("smoothness", smoothness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("radius")) radius = json.get("radius").getAsInt();
        if (json.has("smoothness")) smoothness = json.get("smoothness").getAsInt();
    }
}
