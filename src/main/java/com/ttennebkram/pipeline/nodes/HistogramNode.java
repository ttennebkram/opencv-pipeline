package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Histogram visualization node - replaces image with histogram plot.
 * Output dimensions match input image dimensions.
 * Supports optional second input for mask image.
 */
@NodeInfo(name = "Histogram", category = "Visualization", aliases = {})
public class HistogramNode extends DualInputNode {
    private static final String[] MODES = {"Color (BGR)", "Grayscale", "Per Channel"};
    private static final String[] BACKGROUND_MODES = {"White", "Black", "Background Image"};
    private static final int AXIS_HEIGHT = 30;
    private int modeIndex = 0;
    private int backgroundMode = 0;  // 0=White, 1=Black, 2=Background Image
    private boolean fillBars = false;
    private int lineThickness = 4;

    public HistogramNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Histogram", x, y);
    }

    // Dual input support
    public org.eclipse.swt.graphics.Point getInputPoint2() {
        return new org.eclipse.swt.graphics.Point(x, y + height * 3 / 4);
    }

    @Override
    public org.eclipse.swt.graphics.Point getInputPoint() {
        return new org.eclipse.swt.graphics.Point(x, y + height / 4);
    }

    // Getters/setters for serialization
    public int getModeIndex() { return modeIndex; }
    public void setModeIndex(int v) { modeIndex = v; }
    public int getBackgroundMode() { return backgroundMode; }
    public void setBackgroundMode(int v) { backgroundMode = v; }
    public boolean getFillBars() { return fillBars; }
    public void setFillBars(boolean v) { fillBars = v; }
    public int getLineThickness() { return lineThickness; }
    public void setLineThickness(int v) { lineThickness = v; }

    @Override
    public Mat process(Mat input) {
        return processDual(input, null);
    }

    @Override
    public Mat processDual(Mat input, Mat mask) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Convert mask to grayscale if needed (OpenCV requires CV_8UC1 for masks)
        // Also resize mask to match input image dimensions
        Mat processedMask = null;
        if (mask != null && !mask.empty()) {
            Mat grayMask;
            if (mask.channels() == 3) {
                grayMask = new Mat();
                Imgproc.cvtColor(mask, grayMask, Imgproc.COLOR_BGR2GRAY);
            } else {
                grayMask = mask;
            }

            // Resize mask to match input dimensions if needed
            if (grayMask.width() != input.cols() || grayMask.height() != input.rows()) {
                processedMask = new Mat();
                Imgproc.resize(grayMask, processedMask, new Size(input.cols(), input.rows()));
                if (grayMask != mask) {
                    grayMask.release();
                }
            } else {
                processedMask = grayMask;
            }
        }

        int width = input.cols();
        int height = input.rows();
        int histSize = 256;

        // Create output image matching input dimensions with selected background
        Mat output;
        if (backgroundMode == 1) {
            // Black background
            output = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));
        } else if (backgroundMode == 2) {
            // Use input image as background
            output = input.clone();
        } else {
            // White background (default)
            output = new Mat(height, width, CvType.CV_8UC3, new Scalar(255, 255, 255));
        }

        if (modeIndex == 1) {
            // Grayscale mode
            Mat gray = new Mat();
            if (input.channels() == 3) {
                Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            } else {
                gray = input.clone();
            }

            drawHistogramOnly(gray, output, new Scalar(0, 0, 0), processedMask);
            drawAxis(output, backgroundMode);
            gray.release();
        } else if (modeIndex == 0) {
            // Color combined mode - overlay all three channels
            List<Mat> channels = new ArrayList<>();
            Core.split(input, channels);

            // Draw Blue, Green, Red with respective colors (lines only, no axis yet)
            drawHistogramOnly(channels.get(0), output, new Scalar(255, 0, 0), processedMask);
            drawHistogramOnly(channels.get(1), output, new Scalar(0, 255, 0), processedMask);
            drawHistogramOnly(channels.get(2), output, new Scalar(0, 0, 255), processedMask);

            // Draw axis once at the end
            drawAxis(output, backgroundMode);

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

                drawHistogramInRegion(channels.get(i), roiCopy, color, backgroundMode, processedMask);
                roiCopy.copyTo(roi);
                roiCopy.release();
            }

            for (Mat ch : channels) ch.release();
        }

        // Clean up converted mask if we created it
        if (processedMask != null && processedMask != mask) {
            processedMask.release();
        }

        return output;
    }

    private void drawHistogramOnly(Mat channel, Mat output, Scalar color, Mat mask) {
        int histSize = 256;
        MatOfInt histSizeList = new MatOfInt(histSize);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Mat hist = new Mat();

        // Use mask if provided, otherwise use empty Mat (no mask)
        Mat maskToUse = (mask != null && !mask.empty()) ? mask : new Mat();

        Imgproc.calcHist(
            java.util.Arrays.asList(channel),
            new MatOfInt(0),
            maskToUse,
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

    private void drawAxis(Mat output, int bgMode) {
        int width = output.cols();
        int height = output.rows();

        // Calculate font size proportional to image width (minimum 0.5, scales up with width)
        double fontScale = Math.max(0.5, width / 800.0);
        int thickness = Math.max(1, (int)(width / 400.0));

        // Choose colors based on background mode
        Scalar axisColor, textColor;
        if (bgMode == 0) {
            // White background - use black text
            axisColor = new Scalar(0, 0, 0);
            textColor = new Scalar(0, 0, 0);
        } else {
            // Black or image background - use white text
            axisColor = new Scalar(255, 255, 255);
            textColor = new Scalar(255, 255, 255);
        }

        // Draw x-axis line
        Imgproc.line(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height - AXIS_HEIGHT),
            axisColor, 2);

        // Draw tick marks and labels at regular intervals
        int numTicks = 8; // Will give us 0, 32, 64, 96, 128, 160, 192, 224, 255
        for (int i = 0; i <= numTicks; i++) {
            int value = (i * 255) / numTicks;
            int x = (value * width) / 256;

            // Draw tick mark
            Imgproc.line(output,
                new org.opencv.core.Point(x, height - AXIS_HEIGHT),
                new org.opencv.core.Point(x, height - AXIS_HEIGHT + 8),
                axisColor, 2);

            // Draw label - font size scales with image size
            String label = String.valueOf(value);
            int textWidth = (int)(label.length() * 10 * fontScale);
            int textX = x - textWidth / 2;  // Center the text
            int textY = height - 8;
            Imgproc.putText(output, label,
                new org.opencv.core.Point(textX, textY),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                textColor, thickness);
        }
    }

    private void drawHistogramInRegion(Mat channel, Mat output, Scalar color, int bgMode, Mat mask) {
        int histSize = 256;
        MatOfInt histSizeList = new MatOfInt(histSize);
        MatOfFloat ranges = new MatOfFloat(0f, 256f);
        Mat hist = new Mat();

        // Use mask if provided, otherwise use empty Mat (no mask)
        Mat maskToUse = (mask != null && !mask.empty()) ? mask : new Mat();

        Imgproc.calcHist(
            java.util.Arrays.asList(channel),
            new MatOfInt(0),
            maskToUse,
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

        // Choose colors based on background mode
        Scalar axisColor, textColor;
        if (bgMode == 0) {
            // White background - use black text
            axisColor = new Scalar(0, 0, 0);
            textColor = new Scalar(0, 0, 0);
        } else {
            // Black or image background - use white text
            axisColor = new Scalar(255, 255, 255);
            textColor = new Scalar(255, 255, 255);
        }

        // Draw x-axis
        Imgproc.line(output,
            new org.opencv.core.Point(0, height - AXIS_HEIGHT),
            new org.opencv.core.Point(width, height - AXIS_HEIGHT),
            axisColor, 2);

        // Draw tick marks and labels - fewer ticks for smaller regions
        int numTicks = 4; // 0, 64, 128, 192, 255
        for (int i = 0; i <= numTicks; i++) {
            int value = (i * 255) / numTicks;
            int x = (value * width) / 256;

            // Draw tick mark
            Imgproc.line(output,
                new org.opencv.core.Point(x, height - AXIS_HEIGHT),
                new org.opencv.core.Point(x, height - AXIS_HEIGHT + 8),
                axisColor, 2);

            // Draw label - font size scales with image size
            String label = String.valueOf(value);
            int textWidth = (int)(label.length() * 10 * fontScale);
            int textX = x - textWidth / 2;
            int textY = height - 8;
            Imgproc.putText(output, label,
                new org.opencv.core.Point(textX, textY),
                Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                textColor, thickness);
        }

        hist.release();
        histSizeList.release();
        ranges.release();
    }

    @Override
    public void paint(org.eclipse.swt.graphics.GC gc) {
        // Use default ProcessingNode painting
        super.paint(gc);

        // But override connection points to show dual inputs
        drawDualInputConnectionPoints(gc);
    }

    protected void drawDualInputConnectionPoints(org.eclipse.swt.graphics.GC gc) {
        int radius = 6;

        // Draw first input point (top left)
        org.eclipse.swt.graphics.Point input1 = getInputPoint();
        gc.setBackground(new org.eclipse.swt.graphics.Color(200, 255, 200));
        gc.fillOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);
        gc.setForeground(new org.eclipse.swt.graphics.Color(50, 150, 50));
        gc.setLineWidth(2);
        gc.drawOval(input1.x - radius, input1.y - radius, radius * 2, radius * 2);

        // Draw second input point (bottom left) - for mask
        org.eclipse.swt.graphics.Point input2 = getInputPoint2();
        gc.setBackground(new org.eclipse.swt.graphics.Color(200, 255, 200));
        gc.fillOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);
        gc.setForeground(new org.eclipse.swt.graphics.Color(50, 150, 50));
        gc.drawOval(input2.x - radius, input2.y - radius, radius * 2, radius * 2);

        // Draw output point on right side
        org.eclipse.swt.graphics.Point output = getOutputPoint();
        gc.setBackground(new org.eclipse.swt.graphics.Color(230, 255, 230));
        gc.fillOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setForeground(new org.eclipse.swt.graphics.Color(0, 100, 0));
        gc.drawOval(output.x - radius, output.y - radius, radius * 2, radius * 2);
        gc.setLineWidth(1);
    }

    @Override
    public String getDescription() {
        return "Histogram Visualization\ncv2.calcHist(images, channels, optionalMask, histSize, ranges)";
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
    protected int getPropertiesDialogColumns() {
        return 3;
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = columns;
        sep.setLayoutData(sepGd);

        // Mode selection
        new Label(dialog, SWT.NONE).setText("Display Mode:");
        Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modeCombo.setItems(MODES);
        modeCombo.select(modeIndex);
        GridData modeGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        modeGd.horizontalSpan = columns - 1;
        modeCombo.setLayoutData(modeGd);

        // Background selection
        new Label(dialog, SWT.NONE).setText("Background:");
        Combo bgCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        bgCombo.setItems(BACKGROUND_MODES);
        bgCombo.select(backgroundMode);
        GridData bgGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        bgGd.horizontalSpan = columns - 1;
        bgCombo.setLayoutData(bgGd);

        // Fill bars checkbox
        new Label(dialog, SWT.NONE).setText("Fill Bars:");
        Button fillCheck = new Button(dialog, SWT.CHECK);
        fillCheck.setSelection(fillBars);
        GridData fillGd = new GridData();
        fillGd.horizontalSpan = columns - 1;
        fillCheck.setLayoutData(fillGd);

        // Queues In Sync checkbox
        new Label(dialog, SWT.NONE).setText("Queues In Sync:");
        Button syncCheckbox = new Button(dialog, SWT.CHECK);
        syncCheckbox.setSelection(queuesInSync);
        syncCheckbox.setToolTipText("When checked, only process when both inputs receive new frames");
        GridData syncGd = new GridData();
        syncGd.horizontalSpan = columns - 1;
        syncCheckbox.setLayoutData(syncGd);

        // Line thickness
        new Label(dialog, SWT.NONE).setText("Line Thickness:");
        Scale thicknessScale = new Scale(dialog, SWT.HORIZONTAL);
        thicknessScale.setMinimum(1);
        thicknessScale.setMaximum(10);
        int thicknessSliderPos = Math.min(Math.max(lineThickness, 1), 10);
        thicknessScale.setSelection(thicknessSliderPos);
        thicknessScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label thicknessLabel = new Label(dialog, SWT.NONE);
        thicknessLabel.setText(String.valueOf(lineThickness));
        thicknessScale.addListener(SWT.Selection, e ->
            thicknessLabel.setText(String.valueOf(thicknessScale.getSelection())));

        return () -> {
            modeIndex = modeCombo.getSelectionIndex();
            backgroundMode = bgCombo.getSelectionIndex();
            fillBars = fillCheck.getSelection();
            queuesInSync = syncCheckbox.getSelection();
            lineThickness = thicknessScale.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("modeIndex", modeIndex);
        json.addProperty("backgroundMode", backgroundMode);
        json.addProperty("fillBars", fillBars);
        json.addProperty("lineThickness", lineThickness);
        json.addProperty("queuesInSync", queuesInSync);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("modeIndex")) modeIndex = json.get("modeIndex").getAsInt();
        if (json.has("backgroundMode")) backgroundMode = json.get("backgroundMode").getAsInt();
        if (json.has("fillBars")) fillBars = json.get("fillBars").getAsBoolean();
        if (json.has("lineThickness")) lineThickness = json.get("lineThickness").getAsInt();
        if (json.has("queuesInSync")) queuesInSync = json.get("queuesInSync").getAsBoolean();
    }
}
