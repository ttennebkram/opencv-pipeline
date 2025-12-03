package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Spinner;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Histogram visualization processor.
 * Displays histogram of the image in various modes.
 */
@FXProcessorInfo(
    nodeType = "Histogram",
    displayName = "Histogram",
    category = "Visualization",
    description = "Histogram visualization\nImgproc.calcHist() / custom render"
)
public class HistogramProcessor extends FXProcessorBase {

    // Properties with defaults
    private int modeIndex = 0;       // 0=Color (BGR), 1=Grayscale, 2=Per Channel
    private int backgroundMode = 0;  // 0=White, 1=Black, 2=Background Image
    private boolean fillBars = false;
    private int lineThickness = 4;
    private int minMaxEpsilon = 3;   // Threshold for ignoring near-zero bins in min/max calculation

    private static final String[] MODES = {"Color (BGR)", "Grayscale", "Per Channel"};
    private static final String[] BG_MODES = {"White", "Black", "Background Image"};

    @Override
    public String getNodeType() {
        return "Histogram";
    }

    @Override
    public String getCategory() {
        return "Analysis";
    }

    @Override
    public String getDescription() {
        return "Histogram Visualization\nImgproc.calcHist()";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        int histSize = 256;
        int histWidth = input.cols();
        int histHeight = input.rows();

        // Track temporary Mats for cleanup
        Mat histImage = null;
        MatOfFloat ranges = null;
        MatOfInt histSizeMat = null;
        Mat gray = null;
        List<Mat> bgrPlanes = new ArrayList<>();

        try {
            // Create background
            switch (backgroundMode) {
                case 1: // Black
                    histImage = new Mat(histHeight, histWidth, CvType.CV_8UC3, new Scalar(0, 0, 0));
                    break;
                case 2: // Background Image
                    histImage = input.clone();
                    break;
                default: // White
                    histImage = new Mat(histHeight, histWidth, CvType.CV_8UC3, new Scalar(255, 255, 255));
            }

            ranges = new MatOfFloat(0, 256);
            histSizeMat = new MatOfInt(histSize);

            if (modeIndex == 1) {
                // Grayscale histogram
                gray = new Mat();
                if (input.channels() == 3) {
                    Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
                } else {
                    input.copyTo(gray);
                }
                Mat hist = new Mat();
                try {
                    List<Mat> images = new ArrayList<>();
                    images.add(gray);
                    Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, histSizeMat, ranges);
                    Core.normalize(hist, hist, 0, histHeight, Core.NORM_MINMAX);

                    Scalar grayColor = new Scalar(128, 128, 128);
                    HistStats stats = drawHistogram(histImage, hist, grayColor, histWidth, histHeight);
                    drawMinMaxText(histImage, stats, grayColor, 0, histHeight);
                } finally {
                    hist.release();
                }
            } else {
                // Color or Per Channel
                if (input.channels() == 3) {
                    Core.split(input, bgrPlanes);

                    Scalar[] colors = {new Scalar(255, 0, 0), new Scalar(0, 255, 0), new Scalar(0, 0, 255)};
                    String[] channelNames = {"B", "G", "R"};

                    // Calculate line spacing based on text height (3% of image height + some padding)
                    int lineSpacing = (int)(histHeight * 0.04);

                    for (int i = 0; i < 3; i++) {
                        Mat hist = new Mat();
                        try {
                            List<Mat> images = new ArrayList<>();
                            images.add(bgrPlanes.get(i));
                            Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, histSizeMat, ranges);
                            Core.normalize(hist, hist, 0, histHeight, Core.NORM_MINMAX);

                            HistStats stats = drawHistogram(histImage, hist, colors[i], histWidth, histHeight);
                            // Stack text vertically for each channel
                            drawMinMaxTextWithLabel(histImage, channelNames[i], stats, colors[i], i * lineSpacing, histHeight);
                        } finally {
                            hist.release();
                        }
                    }
                } else {
                    // Single channel - treat as grayscale
                    Mat hist = new Mat();
                    try {
                        List<Mat> images = new ArrayList<>();
                        images.add(input);
                        Imgproc.calcHist(images, new MatOfInt(0), new Mat(), hist, histSizeMat, ranges);
                        Core.normalize(hist, hist, 0, histHeight, Core.NORM_MINMAX);
                        Scalar grayColor = new Scalar(128, 128, 128);
                        HistStats stats = drawHistogram(histImage, hist, grayColor, histWidth, histHeight);
                        drawMinMaxText(histImage, stats, grayColor, 0, histHeight);
                    } finally {
                        hist.release();
                    }
                }
            }

            return histImage;

        } catch (Exception e) {
            // On exception, release histImage since we won't be returning it
            if (histImage != null) histImage.release();
            throw e;
        } finally {
            // Always release temporary Mats
            if (ranges != null) ranges.release();
            if (histSizeMat != null) histSizeMat.release();
            if (gray != null) gray.release();
            for (Mat plane : bgrPlanes) plane.release();
        }
    }

    /**
     * Histogram stats: min bin, max bin, count at min, count at max,
     * and first/last peak (max value) positions.
     */
    private static class HistStats {
        int minBin = -1;
        int maxBin = -1;
        int minCount = 0;
        int maxCount = 0;
        // Peak positions (bins with the maximum histogram value)
        int firstPeakBin = -1;  // First bin with max value
        int lastPeakBin = -1;   // Last bin with max value
    }

    /**
     * Draw histogram and return stats (min/max bin indices, counts, and peak positions).
     * Uses epsilon threshold to ignore very small values near zero.
     */
    private HistStats drawHistogram(Mat histImage, Mat hist, Scalar color, int histWidth, int histHeight) {
        int binWidth = Math.max(1, histWidth / 256);
        double epsilon = minMaxEpsilon;  // Threshold for "near zero" values

        HistStats stats = new HistStats();

        // First pass: find min/max bins with values above epsilon threshold
        // Also find the maximum histogram value (peak height)
        double peakValue = 0;
        for (int i = 0; i < 256; i++) {
            double[] val = hist.get(i, 0);
            if (val != null && val[0] > epsilon) {
                if (stats.minBin == -1) stats.minBin = i;
                stats.maxBin = i;
                // Track maximum value for peak detection
                if (val[0] > peakValue) {
                    peakValue = val[0];
                }
            }
        }

        // Second pass: find first and last bins with the peak value,
        // and count bins at min/max positions
        if (stats.minBin >= 0 && stats.maxBin >= 0) {
            // Count bins at the extremes (within 1 bin of min/max)
            for (int i = 0; i <= stats.minBin && i < 256; i++) {
                double[] val = hist.get(i, 0);
                if (val != null && val[0] > epsilon) {
                    stats.minCount++;
                }
            }
            for (int i = stats.maxBin; i < 256; i++) {
                double[] val = hist.get(i, 0);
                if (val != null && val[0] > epsilon) {
                    stats.maxCount++;
                }
            }

            // Find first and last peak bins (bins with the maximum histogram value)
            // Allow small tolerance for floating point comparison
            double peakTolerance = 0.001;
            for (int i = 0; i < 256; i++) {
                double[] val = hist.get(i, 0);
                if (val != null && Math.abs(val[0] - peakValue) < peakTolerance) {
                    if (stats.firstPeakBin == -1) stats.firstPeakBin = i;
                    stats.lastPeakBin = i;
                }
            }
        }

        // Draw the histogram curve/bars
        for (int i = 1; i < 256; i++) {
            double[] val0 = hist.get(i - 1, 0);
            double[] val1 = hist.get(i, 0);
            if (val0 != null && val1 != null) {
                if (fillBars) {
                    Imgproc.rectangle(histImage,
                        new Point(binWidth * (i - 1), histHeight),
                        new Point(binWidth * i, histHeight - val1[0]),
                        color, -1);
                } else {
                    Imgproc.line(histImage,
                        new Point(binWidth * (i - 1), histHeight - val0[0]),
                        new Point(binWidth * i, histHeight - val1[0]),
                        color, lineThickness);
                }
            }
        }

        return stats;
    }

    /**
     * Calculate font scale so text height is approximately 3% of image height.
     */
    private double calculateFontScale(int histHeight) {
        // Target text height = 3% of image height
        double targetHeight = histHeight * 0.03;
        // FONT_HERSHEY_SIMPLEX at scale 1.0 is roughly 22 pixels tall
        return targetHeight / 22.0;
    }

    /**
     * Draw min/max text labels on the histogram image with counts and peak info.
     */
    private void drawMinMaxText(Mat histImage, HistStats stats, Scalar color, int yOffset, int histHeight) {
        if (stats.minBin < 0 || stats.maxBin < 0) return;

        double fontScale = calculateFontScale(histHeight);
        int thickness = Math.max(1, (int)(fontScale * 2));
        int font = Imgproc.FONT_HERSHEY_SIMPLEX;

        int margin = (int)(histHeight * 0.02);
        int textY = histHeight - margin - yOffset;

        // Draw min label at bottom-left area (with count)
        String minText = "min:" + stats.minBin + "(" + stats.minCount + ")";
        Imgproc.putText(histImage, minText, new Point(margin, textY),
                        font, fontScale, color, thickness);

        // Draw peak info in center area
        String peakText;
        if (stats.firstPeakBin == stats.lastPeakBin) {
            peakText = "peak:" + stats.firstPeakBin;
        } else {
            peakText = "peak:" + stats.firstPeakBin + "-" + stats.lastPeakBin;
        }
        int[] baselinePeak = new int[1];
        Size peakSize = Imgproc.getTextSize(peakText, font, fontScale, thickness, baselinePeak);
        double peakX = (histImage.cols() - peakSize.width) / 2;
        Imgproc.putText(histImage, peakText, new Point(peakX, textY),
                        font, fontScale, color, thickness);

        // Draw max label at bottom-right area (with count)
        String maxText = "max:" + stats.maxBin + "(" + stats.maxCount + ")";
        int[] baseline = new int[1];
        Size textSize = Imgproc.getTextSize(maxText, font, fontScale, thickness, baseline);
        Imgproc.putText(histImage, maxText, new Point(histImage.cols() - textSize.width - margin, textY),
                        font, fontScale, color, thickness);
    }

    /**
     * Draw min/max text labels with channel label prefix (e.g., "R min:0(1) peak:128-130 max:255(1)").
     */
    private void drawMinMaxTextWithLabel(Mat histImage, String label, HistStats stats, Scalar color, int yOffset, int histHeight) {
        if (stats.minBin < 0 || stats.maxBin < 0) return;

        double fontScale = calculateFontScale(histHeight);
        int thickness = Math.max(1, (int)(fontScale * 2));
        int font = Imgproc.FONT_HERSHEY_SIMPLEX;

        int margin = (int)(histHeight * 0.02);
        int textY = histHeight - margin - yOffset;

        // Build peak text
        String peakText;
        if (stats.firstPeakBin == stats.lastPeakBin) {
            peakText = "pk:" + stats.firstPeakBin;
        } else {
            peakText = "pk:" + stats.firstPeakBin + "-" + stats.lastPeakBin;
        }

        // Draw combined label with channel name at bottom-left (with counts and peak)
        String text = label + " min:" + stats.minBin + "(" + stats.minCount + ") " + peakText + " max:" + stats.maxBin + "(" + stats.maxCount + ")";
        Imgproc.putText(histImage, text, new Point(margin, textY),
                        font, fontScale, color, thickness);
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        ComboBox<String> modeCombo = dialog.addComboBox("Mode:", MODES, MODES[Math.min(modeIndex, MODES.length - 1)]);
        ComboBox<String> bgCombo = dialog.addComboBox("Background:", BG_MODES, BG_MODES[Math.min(backgroundMode, BG_MODES.length - 1)]);
        CheckBox fillCheck = dialog.addCheckbox("Fill Bars", fillBars);
        Spinner<Integer> thickSpinner = dialog.addSpinner("Line Thickness:", 1, 10, lineThickness);
        Spinner<Integer> epsilonSpinner = dialog.addSpinner("Min/Max Epsilon:", 0, 50, minMaxEpsilon);

        // Save callback
        dialog.setOnOk(() -> {
            modeIndex = modeCombo.getSelectionModel().getSelectedIndex();
            backgroundMode = bgCombo.getSelectionModel().getSelectedIndex();
            fillBars = fillCheck.isSelected();
            lineThickness = thickSpinner.getValue();
            minMaxEpsilon = epsilonSpinner.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("modeIndex", modeIndex);
        json.addProperty("backgroundMode", backgroundMode);
        json.addProperty("fillBars", fillBars);
        json.addProperty("lineThickness", lineThickness);
        json.addProperty("minMaxEpsilon", minMaxEpsilon);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        modeIndex = getJsonInt(json, "modeIndex", 0);
        backgroundMode = getJsonInt(json, "backgroundMode", 0);
        fillBars = json.has("fillBars") ? json.get("fillBars").getAsBoolean() : false;
        lineThickness = getJsonInt(json, "lineThickness", 4);
        minMaxEpsilon = getJsonInt(json, "minMaxEpsilon", 3);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        modeIndex = getInt(node.properties, "modeIndex", 0);
        backgroundMode = getInt(node.properties, "backgroundMode", 0);
        fillBars = getBool(node.properties, "fillBars", false);
        lineThickness = getInt(node.properties, "lineThickness", 4);
        minMaxEpsilon = getInt(node.properties, "minMaxEpsilon", 3);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("modeIndex", modeIndex);
        node.properties.put("backgroundMode", backgroundMode);
        node.properties.put("fillBars", fillBars);
        node.properties.put("lineThickness", lineThickness);
        node.properties.put("minMaxEpsilon", minMaxEpsilon);
    }

    private boolean getBool(java.util.Map<String, Object> props, String key, boolean defaultValue) {
        if (props.containsKey(key)) {
            Object val = props.get(key);
            if (val instanceof Boolean) return (Boolean) val;
        }
        return defaultValue;
    }
}
