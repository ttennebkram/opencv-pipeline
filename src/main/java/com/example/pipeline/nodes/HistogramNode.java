package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Histogram visualization node - replaces image with histogram plot.
 * Output dimensions match input image dimensions.
 */
public class HistogramNode extends ProcessingNode {
    private static final String[] MODES = {"Color (BGR)", "Grayscale", "Per Channel"};
    private static final int AXIS_HEIGHT = 30;
    private int modeIndex = 0;
    private boolean fillBars = false;
    private int lineThickness = 4;

    public HistogramNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Histogram", x, y);
    }

    // Getters/setters for serialization
    public int getModeIndex() { return modeIndex; }
    public void setModeIndex(int v) { modeIndex = v; }
    public boolean getFillBars() { return fillBars; }
    public void setFillBars(boolean v) { fillBars = v; }
    public int getLineThickness() { return lineThickness; }
    public void setLineThickness(int v) { lineThickness = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        int width = input.cols();
        int height = input.rows();
        int histSize = 256;

        // Create output image matching input dimensions
        Mat output = new Mat(height, width, CvType.CV_8UC3, new Scalar(255, 255, 255));

        if (modeIndex == 1) {
            // Grayscale mode
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            drawHistogramOnly(gray, output, new Scalar(0, 0, 0));
            drawAxis(output);
            gray.release();
        } else if (modeIndex == 0) {
            // Color combined mode - overlay all three channels
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            // Draw Blue, Green, Red with respective colors (lines only, no axis yet)
            drawHistogramOnly(channels.get(0), output, new Scalar(255, 0, 0));
            drawHistogramOnly(channels.get(1), output, new Scalar(0, 255, 0));
            drawHistogramOnly(channels.get(2), output, new Scalar(0, 0, 255));

            // Draw axis once at the end
            drawAxis(output);

            for (Mat ch : channels) ch.release();
        } else {
            // Per Channel mode - side by side
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            int channelWidth = width / 3;

            // Draw each channel in its own section
            for (int i = 0; i < 3; i++) {
                Mat roi = output.submat(0, height, i * channelWidth, (i + 1) * channelWidth);
                Mat roiCopy = roi.clone();

                Scalar color;
                switch (i) {
                    case 0: color = new Scalar(255, 0, 0); break;   // Blue
                    case 1: color = new Scalar(0, 255, 0); break;   // Green
                    default: color = new Scalar(0, 0, 255); break;  // Red
                }

                drawHistogramInRegion(channels.get(i), roiCopy, color);
                roiCopy.copyTo(roi);
                roiCopy.release();
            }

            for (Mat ch : channels) ch.release();
        }

        return output;
    }

    private void drawHistogramOnly(Mat channel, Mat output, Scalar color) {
        int histSize = 256;
        MatOfInt histSizeList = new MatOfInt(histSize);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Mat hist = new Mat();

        Imgproc.calcHist(
            java.util.Arrays.asList(channel),
            new MatOfInt(0),
            new Mat(),
            hist,
            histSizeList,
            ranges
        );

        // Normalize histogram to fit output height (leave space for axis labels)
        Core.normalize(hist, hist, 0, (output.rows() - AXIS_HEIGHT) * 0.9, Core.NORM_MINMAX);

        // Draw histogram
        int width = output.cols();
        int height = output.rows();
        int binWidth = Math.max(1, width / histSize);

        for (int i = 0; i < histSize - 1; i++) {
            int x1 = (i * width) / histSize;
            int x2 = ((i + 1) * width) / histSize;
            int y1 = height - AXIS_HEIGHT - (int) hist.get(i, 0)[0];
            int y2 = height - AXIS_HEIGHT - (int) hist.get(i + 1, 0)[0];

            y1 = Math.max(0, Math.min(height - AXIS_HEIGHT - 1, y1));
            y2 = Math.max(0, Math.min(height - AXIS_HEIGHT - 1, y2));

            if (fillBars) {
                // Draw filled rectangle
                org.opencv.core.Point p1 = new org.opencv.core.Point(x1, y1);
                org.opencv.core.Point p2 = new org.opencv.core.Point(x2, height - AXIS_HEIGHT);
                Imgproc.rectangle(output, p1, p2, color, -1);
            } else {
                // Draw line
                org.opencv.core.Point p1 = new org.opencv.core.Point(x1, y1);
                org.opencv.core.Point p2 = new org.opencv.core.Point(x2, y2);
                Imgproc.line(output, p1, p2, color, lineThickness);
            }
        }

        hist.release();
        histSizeList.release();
        ranges.release();
    }

    private void drawAxis(Mat output) {
        int width = output.cols();
        int height = output.rows();

        // Calculate font size proportional to image width (minimum 0.5, scales up with width)
        double fontScale = Math.max(0.5, width / 800.0);
        int thickness = Math.max(1, (int)(width / 400.0));

        // Fill the axis area with light gray to make it visible
        Imgproc.rectangle(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height),
            new Scalar(240, 240, 240), -1);

        // Draw x-axis line
        Imgproc.line(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height - AXIS_HEIGHT),
            new Scalar(0, 0, 0), 2);

        // Draw tick marks and labels at regular intervals
        int numTicks = 8; // Will give us 0, 32, 64, 96, 128, 160, 192, 224, 255
        for (int i = 0; i <= numTicks; i++) {
            int value = (i * 255) / numTicks;
            int x = (value * width) / 256;

            // Draw tick mark
            Imgproc.line(output,
                new org.opencv.core.Point(x, height - AXIS_HEIGHT),
                new org.opencv.core.Point(x, height - AXIS_HEIGHT + 8),
                new Scalar(0, 0, 0), 2);

            // Draw label - font size scales with image size
            String label = String.valueOf(value);
            int textWidth = (int)(label.length() * 10 * fontScale);
            int textX = x - textWidth / 2;  // Center the text
            int textY = height - 8;
            Imgproc.putText(output, label,
                new org.opencv.core.Point(textX, textY),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                new Scalar(0, 0, 0), thickness);
        }
    }

    private void drawHistogramInRegion(Mat channel, Mat output, Scalar color) {
        int histSize = 256;
        MatOfInt histSizeList = new MatOfInt(histSize);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Mat hist = new Mat();

        Imgproc.calcHist(
            java.util.Arrays.asList(channel),
            new MatOfInt(0),
            new Mat(),
            hist,
            histSizeList,
            ranges
        );

        // Normalize histogram to fit output height (leave space for axis labels)
        Core.normalize(hist, hist, 0, (output.rows() - AXIS_HEIGHT) * 0.9, Core.NORM_MINMAX);

        // Draw histogram
        int width = output.cols();
        int height = output.rows();

        for (int i = 0; i < histSize - 1; i++) {
            int x1 = (i * width) / histSize;
            int x2 = ((i + 1) * width) / histSize;
            int y1 = height - AXIS_HEIGHT - (int) hist.get(i, 0)[0];
            int y2 = height - AXIS_HEIGHT - (int) hist.get(i + 1, 0)[0];

            y1 = Math.max(0, Math.min(height - AXIS_HEIGHT - 1, y1));
            y2 = Math.max(0, Math.min(height - AXIS_HEIGHT - 1, y2));

            if (fillBars) {
                org.opencv.core.Point p1 = new org.opencv.core.Point(x1, y1);
                org.opencv.core.Point p2 = new org.opencv.core.Point(x2, height - AXIS_HEIGHT);
                Imgproc.rectangle(output, p1, p2, color, -1);
            } else {
                org.opencv.core.Point p1 = new org.opencv.core.Point(x1, y1);
                org.opencv.core.Point p2 = new org.opencv.core.Point(x2, y2);
                Imgproc.line(output, p1, p2, color, lineThickness);
            }
        }

        // Draw x-axis
        Imgproc.line(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height - AXIS_HEIGHT),
            new Scalar(0, 0, 0), 2);

        // Calculate font size proportional to image width
        double fontScale = Math.max(0.5, width / 800.0);
        int thickness = Math.max(1, (int)(width / 400.0));

        // Fill the axis area with light gray to make it visible
        Imgproc.rectangle(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height),
            new Scalar(240, 240, 240), -1);

        // Draw x-axis
        Imgproc.line(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height - AXIS_HEIGHT),
            new Scalar(0, 0, 0), 2);

        // Draw tick marks and labels - fewer ticks for smaller regions
        int numTicks = 4; // 0, 64, 128, 192, 255
        for (int i = 0; i <= numTicks; i++) {
            int value = (i * 255) / numTicks;
            int x = (value * width) / 256;

            // Draw tick mark
            Imgproc.line(output,
                new org.opencv.core.Point(x, height - AXIS_HEIGHT),
                new org.opencv.core.Point(x, height - AXIS_HEIGHT + 8),
                new Scalar(0, 0, 0), 2);

            // Draw label - font size scales with image size
            String label = String.valueOf(value);
            int textWidth = (int)(label.length() * 10 * fontScale);
            int textX = x - textWidth / 2;
            int textY = height - 8;
            Imgproc.putText(output, label,
                new org.opencv.core.Point(textX, textY),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                new Scalar(0, 0, 0), thickness);
        }

        hist.release();
        histSizeList.release();
        ranges.release();
    }

    @Override
    public String getDescription() {
        return "Histogram Visualization\ncv2.calcHist()";
    }

    @Override
    public String getDisplayName() {
        return "Histogram";
    }

    @Override
    public String getCategory() {
        return "Visualization";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Histogram Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Mode selection
        new Label(dialog, SWT.NONE).setText("Display Mode:");
        Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modeCombo.setItems(MODES);
        modeCombo.select(modeIndex);
        GridData modeGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        modeCombo.setLayoutData(modeGd);

        // Fill bars checkbox
        new Label(dialog, SWT.NONE).setText("Fill Bars:");
        Button fillCheck = new Button(dialog, SWT.CHECK);
        fillCheck.setSelection(fillBars);

        // Line thickness
        new Label(dialog, SWT.NONE).setText("Line Thickness:");
        Scale thicknessScale = new Scale(dialog, SWT.HORIZONTAL);
        thicknessScale.setMinimum(1);
        thicknessScale.setMaximum(10);
        // Clamp slider position to valid range, but keep actual value
        int thicknessSliderPos = Math.min(Math.max(lineThickness, 1), 10);
        thicknessScale.setSelection(thicknessSliderPos);
        thicknessScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label thicknessLabel = new Label(dialog, SWT.NONE);
        thicknessLabel.setText(String.valueOf(lineThickness)); // Show real value
        thicknessScale.addListener(SWT.Selection, e ->
            thicknessLabel.setText(String.valueOf(thicknessScale.getSelection())));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            modeIndex = modeCombo.getSelectionIndex();
            fillBars = fillCheck.getSelection();
            lineThickness = thicknessScale.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        org.eclipse.swt.graphics.Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }
}
