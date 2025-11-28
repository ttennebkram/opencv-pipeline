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
 * FFT Low-Pass Filter node.
 */
@NodeInfo(name = "FFTLowPass", category = "Filter", aliases = {"FFT Low-Pass Filter"})
public class FFTLowPassFilterNode extends ProcessingNode {
    private int radius = 100;
    private int smoothness = 0;

    // Butterworth filter constants
    private static final double BUTTERWORTH_ORDER_MAX = 10.0;
    private static final double BUTTERWORTH_ORDER_MIN = 0.5;
    private static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    private static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    private static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    private static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    public FFTLowPassFilterNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "FFT Low-Pass Filter", x, y);
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

    private Mat applyFFTToChannel(Mat channel) {
        int rows = channel.rows();
        int cols = channel.cols();

        // Convert to float
        Mat floatChannel = new Mat();
        channel.convertTo(floatChannel, CvType.CV_32F);

        try {
            // Compute DFT
            Mat dft = new Mat();
            Core.dft(floatChannel, dft, Core.DFT_COMPLEX_OUTPUT);

            try {
                // Shift zero frequency to center (modifies dft in place)
                fftShift(dft);

                // Create mask
                Mat mask = createMask(rows, cols);

                try {
                    // Apply mask
                    List<Mat> planes = new ArrayList<>();
                    Core.split(dft, planes);

                    try {
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

                        try {
                            // Inverse shift (modifies maskedDft in place)
                            ifftShift(maskedDft);

                            // Inverse DFT
                            Mat idft = new Mat();
                            Core.idft(maskedDft, idft, Core.DFT_SCALE);

                            try {
                                // Get magnitude
                                List<Mat> idftPlanes = new ArrayList<>();
                                Core.split(idft, idftPlanes);

                                try {
                                    Mat magnitude = new Mat();
                                    Core.magnitude(idftPlanes.get(0), idftPlanes.get(1), magnitude);

                                    try {
                                        // Clip to 0-255 and convert to 8-bit
                                        Core.min(magnitude, new Scalar(255), magnitude);
                                        Core.max(magnitude, new Scalar(0), magnitude);

                                        Mat result = new Mat();
                                        magnitude.convertTo(result, CvType.CV_8U);

                                        return result;
                                    } finally {
                                        magnitude.release();
                                    }
                                } finally {
                                    for (Mat m : idftPlanes) {
                                        m.release();
                                    }
                                }
                            } finally {
                                idft.release();
                            }
                        } finally {
                            maskedDft.release();
                        }
                    } finally {
                        for (Mat m : planes) {
                            m.release();
                        }
                    }
                } finally {
                    mask.release();
                }
            } finally {
                dft.release();
            }
        } finally {
            floatChannel.release();
        }
    }

    private void fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        // Create quadrants
        Mat q0 = input.submat(0, cy, 0, cx);      // Top-Left
        Mat q1 = input.submat(0, cy, cx, input.cols());  // Top-Right
        Mat q2 = input.submat(cy, input.rows(), 0, cx);  // Bottom-Left
        Mat q3 = input.submat(cy, input.rows(), cx, input.cols()); // Bottom-Right

        try {
            // Swap quadrants (Top-Left with Bottom-Right)
            Mat tmp = new Mat();
            try {
                q0.copyTo(tmp);
                q3.copyTo(q0);
                tmp.copyTo(q3);

                // Swap quadrants (Top-Right with Bottom-Left)
                q1.copyTo(tmp);
                q2.copyTo(q1);
                tmp.copyTo(q2);
            } finally {
                tmp.release();
            }
        } finally {
            q0.release();
            q1.release();
            q2.release();
            q3.release();
        }
    }

    private void ifftShift(Mat input) {
        // ifftShift is the same as fftShift
        fftShift(input);
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

        float[] maskData = new float[rows * cols];

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < cols; x++) {
                double distance = Math.sqrt((x - ccol) * (x - ccol) + (y - crow) * (y - crow));

                float value;
                if (smoothness == 0) {
                    // Hard circle mask (low-pass)
                    value = distance <= radius ? 1.0f : 0.0f;
                } else {
                    // Butterworth lowpass filter
                    double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
                    if (order < BUTTERWORTH_ORDER_MIN) {
                        order = BUTTERWORTH_ORDER_MIN;
                    }

                    double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
                    double effectiveCutoff = radius * shiftFactor;

                    // Inverted ratio for low-pass (distance / cutoff instead of cutoff / distance)
                    double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
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
        return "FFT Low-Pass Filter\nnp.fft.fft2() / np.fft.ifft2()";
    }

    @Override
    public String getDisplayName() {
        return "FFT Low-Pass";
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

