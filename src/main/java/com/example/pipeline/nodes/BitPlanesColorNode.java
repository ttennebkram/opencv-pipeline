package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.custom.CTabFolder;
import org.eclipse.swt.custom.CTabItem;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Color Bit Plane Decomposition node for RGB images.
 */
public class BitPlanesColorNode extends ProcessingNode {
    // Slider constants for logarithmic gain control
    private static final int GAIN_SLIDER_MIN = 0;
    private static final int GAIN_SLIDER_MAX = 200;
    private static final int GAIN_SLIDER_CENTER = 100;
    private static final double GAIN_SLIDER_SCALE = 100.0;
    private static final int GAIN_SLIDER_WIDTH = 244;

    // 3 channels × 8 bit planes (index 0 = bit 7 MSB, index 7 = bit 0 LSB)
    // Order: Red, Green, Blue
    private boolean[][] bitEnabled = new boolean[3][8];
    private double[][] bitGain = new double[3][8];

    private static final String[] CHANNEL_NAMES = {"Red", "Green", "Blue"};

    public BitPlanesColorNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Bit Planes Color", x, y);
        // Initialize all to enabled with gain 1.0
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 8; i++) {
                bitEnabled[c][i] = true;
                bitGain[c][i] = 1.0;
            }
        }
    }

    // Getters/setters for serialization
    public boolean getBitEnabled(int channel, int index) { return bitEnabled[channel][index]; }
    public void setBitEnabled(int channel, int index, boolean v) { bitEnabled[channel][index] = v; }
    public double getBitGain(int channel, int index) { return bitGain[channel][index]; }
    public void setBitGain(int channel, int index, double v) { bitGain[channel][index] = v; }

    // For JSON serialization - flatten arrays
    public boolean[] getRedBitEnabled() { return bitEnabled[0].clone(); }
    public void setRedBitEnabled(boolean[] v) { for (int i = 0; i < 8 && i < v.length; i++) bitEnabled[0][i] = v[i]; }
    public double[] getRedBitGain() { return bitGain[0].clone(); }
    public void setRedBitGain(double[] v) { for (int i = 0; i < 8 && i < v.length; i++) bitGain[0][i] = v[i]; }

    public boolean[] getGreenBitEnabled() { return bitEnabled[1].clone(); }
    public void setGreenBitEnabled(boolean[] v) { for (int i = 0; i < 8 && i < v.length; i++) bitEnabled[1][i] = v[i]; }
    public double[] getGreenBitGain() { return bitGain[1].clone(); }
    public void setGreenBitGain(double[] v) { for (int i = 0; i < 8 && i < v.length; i++) bitGain[1][i] = v[i]; }

    public boolean[] getBlueBitEnabled() { return bitEnabled[2].clone(); }
    public void setBlueBitEnabled(boolean[] v) { for (int i = 0; i < 8 && i < v.length; i++) bitEnabled[2][i] = v[i]; }
    public double[] getBlueBitGain() { return bitGain[2].clone(); }
    public void setBlueBitGain(double[] v) { for (int i = 0; i < 8 && i < v.length; i++) bitGain[2][i] = v[i]; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Ensure we have a color image
        Mat color = input;
        if (input.channels() == 1) {
            color = new Mat();
            Imgproc.cvtColor(input, color, Imgproc.COLOR_GRAY2BGR);
        }

        // Split into BGR channels
        List<Mat> channels = new ArrayList<>();
        Core.split(color, channels);

        // Process each channel (BGR order in OpenCV)
        // channels: 0=Blue, 1=Green, 2=Red
        int[] channelMap = {2, 1, 0}; // Red, Green, Blue -> BGR indices

        List<Mat> resultChannels = new ArrayList<>();
        for (int c = 0; c < 3; c++) {
            resultChannels.add(new Mat());
        }

        for (int colorIdx = 0; colorIdx < 3; colorIdx++) {
            int bgrIdx = channelMap[colorIdx];
            Mat channel = channels.get(bgrIdx);

            // Initialize result as float for accumulation
            Mat result = Mat.zeros(channel.rows(), channel.cols(), CvType.CV_32F);

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
            resultMat.put(0, 0, resultData);

            Mat result8u = new Mat();
            resultMat.convertTo(result8u, CvType.CV_8U);

            resultChannels.set(bgrIdx, result8u);
        }

        // Merge channels back
        Mat output = new Mat();
        Core.merge(resultChannels, output);

        return output;
    }

    @Override
    public String getDescription() {
        return "Bit Planes Color: Select and adjust RGB bit planes\nBit plane decomposition with gain (RGB)";
    }

    @Override
    public String getDisplayName() {
        return "Bit Planes Color";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL | SWT.RESIZE);
        dialog.setText("Bit Planes Color Properties");
        dialog.setLayout(new GridLayout(1, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        sigLabel.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        sep.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Tab folder for Red, Green, Blue
        CTabFolder tabFolder = new CTabFolder(dialog, SWT.BORDER);
        tabFolder.setLayoutData(new GridData(SWT.FILL, SWT.FILL, true, true));

        // Arrays to hold all controls
        Button[][] checkButtons = new Button[3][8];
        Scale[][] gainScales = new Scale[3][8];
        Label[][] gainLabels = new Label[3][8];

        for (int c = 0; c < 3; c++) {
            CTabItem tabItem = new CTabItem(tabFolder, SWT.NONE);
            tabItem.setText(CHANNEL_NAMES[c]);

            Composite tabContent = new Composite(tabFolder, SWT.NONE);
            tabContent.setLayout(new GridLayout(4, false));
            tabItem.setControl(tabContent);

            // Header row
            new Label(tabContent, SWT.NONE).setText("Bit");
            new Label(tabContent, SWT.NONE).setText("On");
            new Label(tabContent, SWT.NONE).setText("Gain (0.1x - 10x)");
            new Label(tabContent, SWT.NONE).setText("");

            // Create 8 rows for bit planes
            for (int i = 0; i < 8; i++) {
                int bitNum = 7 - i;

                // Bit label
                new Label(tabContent, SWT.NONE).setText(String.valueOf(bitNum));

                // Enable checkbox
                checkButtons[c][i] = new Button(tabContent, SWT.CHECK);
                checkButtons[c][i].setSelection(bitEnabled[c][i]);

                // Gain slider (logarithmic: 0-200 maps to 0.1x-10x)
                gainScales[c][i] = new Scale(tabContent, SWT.HORIZONTAL);
                gainScales[c][i].setMinimum(GAIN_SLIDER_MIN);
                gainScales[c][i].setMaximum(GAIN_SLIDER_MAX);
                // Convert gain to slider: slider = log10(gain) * SCALE + CENTER
                int sliderVal = (int)(Math.log10(bitGain[c][i]) * GAIN_SLIDER_SCALE + GAIN_SLIDER_CENTER);
                gainScales[c][i].setSelection(sliderVal);
                gainScales[c][i].setLayoutData(new GridData(GAIN_SLIDER_WIDTH, SWT.DEFAULT));

                // Gain label
                gainLabels[c][i] = new Label(tabContent, SWT.NONE);
                gainLabels[c][i].setText(String.format("%.2fx", bitGain[c][i]));
                GridData lblGd = new GridData(SWT.LEFT, SWT.CENTER, false, false);
                lblGd.widthHint = 50;
                gainLabels[c][i].setLayoutData(lblGd);

                // Update label when scale changes
                final int channel = c;
                final int idx = i;
                gainScales[c][i].addListener(SWT.Selection, e -> {
                    // Convert slider to gain: gain = 10^((slider - CENTER) / SCALE)
                    double val = Math.pow(10, (gainScales[channel][idx].getSelection() - GAIN_SLIDER_CENTER) / GAIN_SLIDER_SCALE);
                    gainLabels[channel][idx].setText(String.format("%.2fx", val));
                });
            }
        }

        tabFolder.setSelection(0);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        buttonComp.setLayoutData(new GridData(SWT.RIGHT, SWT.CENTER, true, false));

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            for (int c = 0; c < 3; c++) {
                for (int i = 0; i < 8; i++) {
                    bitEnabled[c][i] = checkButtons[c][i].getSelection();
                    // Convert slider to gain: gain = 10^((slider - CENTER) / SCALE)
                    bitGain[c][i] = Math.pow(10, (gainScales[c][i].getSelection() - GAIN_SLIDER_CENTER) / GAIN_SLIDER_SCALE);
                }
            }
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.setSize(400, 450);
        // Position dialog near cursor
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }
}
