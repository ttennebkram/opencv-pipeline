package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Bit Plane Decomposition node for grayscale images.
 */
public class BitPlanesGrayscaleNode extends ProcessingNode {
    // Slider constants for logarithmic gain control
    private static final int GAIN_SLIDER_MIN = 0;
    private static final int GAIN_SLIDER_MAX = 200;
    private static final int GAIN_SLIDER_CENTER = 100;
    private static final double GAIN_SLIDER_SCALE = 100.0;
    private static final int GAIN_SLIDER_WIDTH = 244;

    // 8 bit planes (index 0 = bit 7 MSB, index 7 = bit 0 LSB)
    private boolean[] bitEnabled = {true, true, true, true, true, true, true, true};
    private double[] bitGain = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    public BitPlanesGrayscaleNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Bit Planes Grayscale", x, y);
    }

    // Getters/setters for serialization
    public boolean getBitEnabled(int index) { return bitEnabled[index]; }
    public void setBitEnabled(int index, boolean v) { bitEnabled[index] = v; }
    public double getBitGain(int index) { return bitGain[index]; }
    public void setBitGain(int index, double v) { bitGain[index] = v; }

    // For JSON serialization
    public boolean[] getAllBitEnabled() { return bitEnabled.clone(); }
    public void setAllBitEnabled(boolean[] v) {
        for (int i = 0; i < 8 && i < v.length; i++) bitEnabled[i] = v[i];
    }
    public double[] getAllBitGain() { return bitGain.clone(); }
    public void setAllBitGain(double[] v) {
        for (int i = 0; i < 8 && i < v.length; i++) bitGain[i] = v[i];
    }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Convert to grayscale if needed
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Initialize result as float for accumulation
        Mat result = Mat.zeros(gray.rows(), gray.cols(), CvType.CV_32F);

        // Process each bit plane
        for (int i = 0; i < 8; i++) {
            if (!bitEnabled[i]) {
                continue;
            }

            // Extract bit plane (bit 7-i, since i=0 is MSB)
            int bitIndex = 7 - i;

            // Create mask for this bit
            Mat bitPlane = new Mat();
            gray.convertTo(bitPlane, CvType.CV_32F);

            // Extract the bit: (gray >> bitIndex) & 1
            // In OpenCV Java, we need to do this differently
            Mat temp = new Mat();
            gray.convertTo(temp, CvType.CV_32F);

            // Divide by 2^bitIndex to shift right
            Core.multiply(temp, new org.opencv.core.Scalar(1.0 / (1 << bitIndex)), temp);

            // Floor to integer
            Mat floored = new Mat();
            temp.convertTo(floored, CvType.CV_32S);
            floored.convertTo(temp, CvType.CV_32F);

            // AND with 1 (modulo 2)
            Mat ones = new Mat(temp.rows(), temp.cols(), CvType.CV_32F, new org.opencv.core.Scalar(2.0));
            Core.subtract(temp, ones, floored);
            Core.absdiff(temp, floored, temp);

            // Actually let's do this more directly
            // Extract bit plane by shifting and masking
            byte[] grayData = new byte[gray.rows() * gray.cols()];
            gray.get(0, 0, grayData);

            float[] bitData = new float[gray.rows() * gray.cols()];
            for (int j = 0; j < grayData.length; j++) {
                int pixelValue = grayData[j] & 0xFF;
                int bit = (pixelValue >> bitIndex) & 1;
                // Scale to original bit weight and apply gain
                bitData[j] = bit * (1 << bitIndex) * (float) bitGain[i];
            }

            Mat bitValue = new Mat(gray.rows(), gray.cols(), CvType.CV_32F);
            bitValue.put(0, 0, bitData);

            // Accumulate
            Core.add(result, bitValue, result);
        }

        // Clip to valid range [0, 255]
        Core.min(result, new org.opencv.core.Scalar(255), result);
        Core.max(result, new org.opencv.core.Scalar(0), result);

        // Convert to 8-bit
        Mat result8u = new Mat();
        result.convertTo(result8u, CvType.CV_8U);

        // Convert back to BGR for display
        Mat output = new Mat();
        Imgproc.cvtColor(result8u, output, Imgproc.COLOR_GRAY2BGR);

        return output;
    }

    @Override
    public String getDescription() {
        return "Bit Planes Grayscale: Select and adjust bit planes\nBit plane decomposition with gain";
    }

    @Override
    public String getDisplayName() {
        return "Bit Planes Grayscale";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Bit Planes Grayscale Properties");
        dialog.setLayout(new GridLayout(4, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 4;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 4;
        sep.setLayoutData(sepGd);

        // Header row
        new Label(dialog, SWT.NONE).setText("Bit");
        new Label(dialog, SWT.NONE).setText("On");
        new Label(dialog, SWT.NONE).setText("Gain (0.1x - 10x)");
        new Label(dialog, SWT.NONE).setText("");

        // Arrays to hold controls
        Button[] checkButtons = new Button[8];
        Scale[] gainScales = new Scale[8];
        Label[] gainLabels = new Label[8];

        // Create 8 rows for bit planes
        for (int i = 0; i < 8; i++) {
            int bitNum = 7 - i;

            // Bit label
            new Label(dialog, SWT.NONE).setText(String.valueOf(bitNum));

            // Enable checkbox
            checkButtons[i] = new Button(dialog, SWT.CHECK);
            checkButtons[i].setSelection(bitEnabled[i]);

            // Gain slider (logarithmic: 0-200 maps to 0.1x-10x)
            gainScales[i] = new Scale(dialog, SWT.HORIZONTAL);
            gainScales[i].setMinimum(GAIN_SLIDER_MIN);
            gainScales[i].setMaximum(GAIN_SLIDER_MAX);
            // Convert gain to slider: slider = log10(gain) * SCALE + CENTER
            int sliderVal = (int)(Math.log10(bitGain[i]) * GAIN_SLIDER_SCALE + GAIN_SLIDER_CENTER);
            gainScales[i].setSelection(sliderVal);
            gainScales[i].setLayoutData(new GridData(GAIN_SLIDER_WIDTH, SWT.DEFAULT));

            // Gain label
            gainLabels[i] = new Label(dialog, SWT.NONE);
            gainLabels[i].setText(String.format("%.2fx", bitGain[i]));
            GridData lblGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
            lblGd.widthHint = 50;
            gainLabels[i].setLayoutData(lblGd);

            // Update label when scale changes
            final int idx = i;
            gainScales[i].addListener(SWT.Selection, e -> {
                // Convert slider to gain: gain = 10^((slider - CENTER) / SCALE)
                double val = Math.pow(10, (gainScales[idx].getSelection() - GAIN_SLIDER_CENTER) / GAIN_SLIDER_SCALE);
                gainLabels[idx].setText(String.format("%.2fx", val));
            });
        }

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 4;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            for (int i = 0; i < 8; i++) {
                bitEnabled[i] = checkButtons[i].getSelection();
                // Convert slider to gain: gain = 10^((slider - CENTER) / SCALE)
                bitGain[i] = Math.pow(10, (gainScales[i].getSelection() - GAIN_SLIDER_CENTER) / GAIN_SLIDER_SCALE);
            }
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
}
