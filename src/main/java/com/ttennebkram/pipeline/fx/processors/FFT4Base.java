package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.scene.layout.HBox;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for FFT 4-output processors (FFTLowPass4, FFTHighPass4).
 * Provides common FFT processing, mask creation, and visualization code.
 *
 * 4 Outputs:
 * [0] = Filtered image
 * [1] = Absolute difference (blocked frequencies)
 * [2] = FFT spectrum with filter overlay
 * [3] = Filter response curve graph
 */
public abstract class FFT4Base extends FXProcessorBase implements FXMultiOutputProcessor {

    // Properties
    protected int radius;
    protected int smoothness = 0;

    // Butterworth filter constants
    protected static final double BUTTERWORTH_ORDER_MAX = 10.0;
    protected static final double BUTTERWORTH_ORDER_MIN = 0.5;
    protected static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    protected static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    protected static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    protected static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    public FFT4Base() {
        this.radius = getDefaultRadius();
    }

    @Override
    public String getCategory() {
        return "FFT";
    }

    @Override
    public int getOutputCount() {
        return 4;
    }

    @Override
    public String[] getOutputLabels() {
        return new String[]{"Filtered", "Difference", "Spectrum", "Filter Curve"};
    }

    /**
     * Get the default radius for this filter type.
     */
    protected abstract int getDefaultRadius();

    /**
     * Whether this is a high-pass filter (vs low-pass).
     */
    protected abstract boolean isHighPass();

    @Override
    public Mat[] processMultiOutput(Mat input) {
        if (isInvalidInput(input)) {
            return new Mat[]{input != null ? input.clone() : null, null, null, null};
        }

        int origRows = input.rows();
        int origCols = input.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        // Track all Mats for cleanup on exception
        Mat mask = null;
        Mat filterVis = null;
        Mat filtered = null;
        Mat spectrum = null;
        Mat difference = null;
        List<Mat> channels = new ArrayList<>();
        List<Mat> filteredChannels = new ArrayList<>();

        try {
            // Create the filter mask at optimal size
            mask = isHighPass() ?
                createHighPassMask(optRows, optCols) :
                createLowPassMask(optRows, optCols);

            // Create filter curve visualization (for output 4)
            filterVis = createFilterCurveVisualization(origCols, origRows);

            if (input.channels() > 1) {
                // Process each BGR channel separately
                Core.split(input, channels);

                Mat spectrumAccum = null;

                for (int c = 0; c < channels.size(); c++) {
                    Mat channel = channels.get(c);

                    // Pad to optimal size
                    Mat padded = new Mat();
                    Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
                        Core.BORDER_CONSTANT, Scalar.all(0));

                    // Convert to float and create complex mat
                    Mat floatChannel = new Mat();
                    padded.convertTo(floatChannel, CvType.CV_32F);
                    padded.release();

                    Mat complexI = new Mat();
                    List<Mat> planes = new ArrayList<>();
                    planes.add(floatChannel);
                    planes.add(Mat.zeros(floatChannel.size(), CvType.CV_32F));
                    Core.merge(planes, complexI);
                    planes.get(1).release();
                    floatChannel.release();

                    // Compute DFT and shift
                    Core.dft(complexI, complexI);
                    fftShift(complexI);

                    // For spectrum visualization, use the first channel
                    if (c == 0) {
                        spectrumAccum = createSpectrumVisualization(complexI, origRows, origCols);
                    }

                    // Apply mask
                    List<Mat> dftPlanes = new ArrayList<>();
                    Core.split(complexI, dftPlanes);
                    complexI.release();

                    Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
                    Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

                    Mat maskedDft = new Mat();
                    Core.merge(dftPlanes, maskedDft);
                    for (Mat p : dftPlanes) p.release();

                    // Inverse shift and DFT
                    fftShift(maskedDft);
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
                channels.clear();
                for (Mat ch : filteredChannels) ch.release();
                filteredChannels.clear();
            } else {
                // Single channel (grayscale) processing
                Mat padded = new Mat();
                Core.copyMakeBorder(input, padded, 0, optRows - origRows, 0, optCols - origCols,
                    Core.BORDER_CONSTANT, Scalar.all(0));

                Mat floatInput = new Mat();
                padded.convertTo(floatInput, CvType.CV_32F);
                padded.release();

                Mat complexI = new Mat();
                List<Mat> planes = new ArrayList<>();
                planes.add(floatInput);
                planes.add(Mat.zeros(floatInput.size(), CvType.CV_32F));
                Core.merge(planes, complexI);
                planes.get(1).release();
                floatInput.release();

                Core.dft(complexI, complexI);
                fftShift(complexI);

                spectrum = createSpectrumVisualization(complexI, origRows, origCols);

                // Apply mask
                List<Mat> dftPlanes = new ArrayList<>();
                Core.split(complexI, dftPlanes);
                complexI.release();

                Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
                Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));

                Mat maskedDft = new Mat();
                Core.merge(dftPlanes, maskedDft);
                for (Mat p : dftPlanes) p.release();

                fftShift(maskedDft);
                Core.idft(maskedDft, maskedDft, Core.DFT_SCALE);

                List<Mat> idftPlanes = new ArrayList<>();
                Core.split(maskedDft, idftPlanes);
                maskedDft.release();

                Mat magnitude = idftPlanes.get(0);
                idftPlanes.get(1).release();

                Mat cropped = new Mat(magnitude, new Rect(0, 0, origCols, origRows));
                Mat result = cropped.clone();
                magnitude.release();

                Core.min(result, new Scalar(255), result);
                Core.max(result, new Scalar(0), result);
                filtered = new Mat();
                result.convertTo(filtered, CvType.CV_8U);
                result.release();

                // Convert grayscale to BGR for consistency
                Mat filteredBGR = new Mat();
                Imgproc.cvtColor(filtered, filteredBGR, Imgproc.COLOR_GRAY2BGR);
                filtered.release();
                filtered = filteredBGR;
            }

            // Calculate absolute difference
            difference = new Mat();
            Core.absdiff(input, filtered, difference);

            // Add filter overlay to spectrum
            addFilterOverlayToSpectrum(spectrum);

            // Release mask (no longer needed)
            mask.release();
            mask = null;

            return new Mat[]{filtered, difference, spectrum, filterVis};

        } catch (Exception e) {
            // On exception, release all output Mats we created
            if (filtered != null) filtered.release();
            if (difference != null) difference.release();
            if (spectrum != null) spectrum.release();
            if (filterVis != null) filterVis.release();
            throw e;
        } finally {
            // Always release temporary Mats
            if (mask != null) mask.release();
            for (Mat ch : channels) ch.release();
            for (Mat ch : filteredChannels) ch.release();
        }
    }

    /**
     * Add filter radius overlay to spectrum visualization.
     */
    protected abstract void addFilterOverlayToSpectrum(Mat spectrum);

    // ===== FFT Helper Methods =====

    private int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    protected int getOptimalFFTSize(int n) {
        int pow2 = nextPowerOf2(n);
        int optimal = Core.getOptimalDFTSize(n);

        // Use power of 2 if it adds less than 25% overhead
        if (pow2 <= optimal * 1.25) {
            return pow2;
        }
        // Otherwise use OpenCV's optimal (but ensure even for fftShift)
        return optimal % 2 == 0 ? optimal : optimal + 1;
    }

    protected void fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

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

    protected Mat createLowPassMask(int rows, int cols) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (radius == 0) {
            mask.setTo(new Scalar(0.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (smoothness == 0) {
            // Hard circle mask
            mask.setTo(new Scalar(0.0));
            Imgproc.circle(mask, new Point(ccol, crow), radius, new Scalar(1.0), -1);
        } else {
            // Butterworth filter
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

    protected Mat createHighPassMask(int rows, int cols) {
        Mat lowPass = createLowPassMask(rows, cols);
        // Create ones matrix and subtract low-pass to get high-pass
        Mat ones = Mat.ones(rows, cols, CvType.CV_32F);
        Mat highPass = new Mat();
        Core.subtract(ones, lowPass, highPass);
        ones.release();
        lowPass.release();
        return highPass;
    }

    protected Mat createSpectrumVisualization(Mat dftShift, int origRows, int origCols) {
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

        Mat spectrum8U = new Mat();
        logMag.convertTo(spectrum8U, CvType.CV_8U);

        // Crop to original size (center portion)
        int padRows = spectrum8U.rows();
        int padCols = spectrum8U.cols();
        int startRow = (padRows - origRows) / 2;
        int startCol = (padCols - origCols) / 2;
        Mat cropped = new Mat(spectrum8U, new Rect(startCol, startRow, origCols, origRows));
        Mat croppedClone = cropped.clone();

        // Convert to BGR for display
        Mat spectrumBGR = new Mat();
        Imgproc.cvtColor(croppedClone, spectrumBGR, Imgproc.COLOR_GRAY2BGR);

        // Cleanup
        for (Mat p : planes) p.release();
        magnitude.release();
        logMag.release();
        spectrum8U.release();
        croppedClone.release();

        return spectrumBGR;
    }

    protected double computeFilterValue(double distance) {
        if (radius == 0) {
            return isHighPass() ? 1.0 : 0.0;
        }

        double lowPassValue;
        if (smoothness == 0) {
            lowPassValue = distance <= radius ? 1.0 : 0.0;
        } else {
            double order = BUTTERWORTH_ORDER_MAX - (smoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) order = BUTTERWORTH_ORDER_MIN;

            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = radius * shiftFactor;

            double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
            lowPassValue = 1.0 / (1.0 + Math.pow(ratio, 2 * order));
        }

        return isHighPass() ? (1.0 - lowPassValue) : lowPassValue;
    }

    protected Mat createFilterCurveVisualization(int width, int height) {
        // Create a black background image
        Mat vis = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));

        // Graph margins
        int marginLeft = 50;
        int marginRight = 20;
        int marginTop = 30;
        int marginBottom = 40;

        int graphWidth = width - marginLeft - marginRight;
        int graphHeight = height - marginTop - marginBottom;

        if (graphWidth <= 0 || graphHeight <= 0) {
            return vis;
        }

        // Max distance for x-axis
        int maxDistance = Math.max(200, radius * 3);

        // Draw grid lines (gray)
        Scalar gridColor = new Scalar(60, 60, 60);
        for (int i = 0; i <= 4; i++) {
            int yPos = marginTop + (int) (graphHeight * (1.0 - i / 4.0));
            Imgproc.line(vis,
                new Point(marginLeft, yPos),
                new Point(width - marginRight, yPos),
                gridColor, 6);
        }
        for (int d = 0; d <= maxDistance; d += 50) {
            int xPos = marginLeft + (int) (graphWidth * d / (double) maxDistance);
            Imgproc.line(vis,
                new Point(xPos, marginTop),
                new Point(xPos, marginTop + graphHeight),
                gridColor, 6);
        }

        // Draw axes (white)
        Scalar axisColor = new Scalar(255, 255, 255);
        Imgproc.line(vis,
            new Point(marginLeft, marginTop),
            new Point(marginLeft, marginTop + graphHeight),
            axisColor, 12);
        Imgproc.line(vis,
            new Point(marginLeft, marginTop + graphHeight),
            new Point(width - marginRight, marginTop + graphHeight),
            axisColor, 12);

        // Draw the filter curve
        Scalar curveColor = isHighPass() ?
            new Scalar(100, 100, 255) :  // Light red for high-pass
            new Scalar(255, 100, 100);   // Light blue for low-pass

        Point prevPoint = null;
        for (int i = 0; i <= graphWidth; i++) {
            double distance = (i / (double) graphWidth) * maxDistance;
            double filterValue = computeFilterValue(distance);

            int xPos = marginLeft + i;
            int yPos = marginTop + (int) (graphHeight * (1.0 - filterValue));

            Point currentPoint = new Point(xPos, yPos);
            if (prevPoint != null) {
                Imgproc.line(vis, prevPoint, currentPoint, curveColor, 10);
            }
            prevPoint = currentPoint;
        }

        // Draw vertical line at radius (red)
        if (radius > 0 && radius <= maxDistance) {
            int radiusX = marginLeft + (int) (graphWidth * radius / (double) maxDistance);
            Scalar radiusColor = new Scalar(0, 0, 255);
            Imgproc.line(vis,
                new Point(radiusX, marginTop),
                new Point(radiusX, marginTop + graphHeight),
                radiusColor, 4);
        }

        // Draw labels
        Scalar textColor = new Scalar(200, 200, 200);
        double fontScale = 0.5;
        int fontFace = Imgproc.FONT_HERSHEY_SIMPLEX;

        String title = isHighPass() ? "High-Pass Response" : "Low-Pass Response";
        Imgproc.putText(vis, title,
            new Point(marginLeft + 10, 20), fontFace, fontScale, textColor, 1);

        // Y-axis labels
        Imgproc.putText(vis, "1.0",
            new Point(5, marginTop + 5), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "0.5",
            new Point(5, marginTop + graphHeight / 2 + 5), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, "0.0",
            new Point(5, marginTop + graphHeight + 5), fontFace, fontScale, textColor, 1);

        // X-axis labels
        Imgproc.putText(vis, "0",
            new Point(marginLeft - 5, height - 10), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, String.valueOf(maxDistance / 2),
            new Point(marginLeft + graphWidth / 2 - 10, height - 10), fontFace, fontScale, textColor, 1);
        Imgproc.putText(vis, String.valueOf(maxDistance),
            new Point(width - marginRight - 25, height - 10), fontFace, fontScale, textColor, 1);

        // Radius label
        if (radius > 0) {
            Imgproc.putText(vis, "R=" + radius,
                new Point(width - 80, marginTop + 15),
                fontFace, fontScale, new Scalar(0, 0, 255), 1);
        }

        return vis;
    }

    // ===== Properties Dialog =====

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Radius slider
        HBox radiusRow = new HBox(10);
        Label radiusLabel = new Label("Radius:");
        radiusLabel.setMinWidth(80);
        Slider radiusSlider = new Slider(0, 300, radius);
        radiusSlider.setPrefWidth(200);
        Label radiusValue = new Label(String.valueOf(radius));
        radiusValue.setMinWidth(40);
        radiusSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            radiusValue.setText(String.valueOf(newVal.intValue()));
        });
        radiusRow.getChildren().addAll(radiusLabel, radiusSlider, radiusValue);
        dialog.addCustomContent(radiusRow);

        // Smoothness slider
        HBox smoothRow = new HBox(10);
        Label smoothLabel = new Label("Smoothness:");
        smoothLabel.setMinWidth(80);
        Slider smoothSlider = new Slider(0, 100, smoothness);
        smoothSlider.setPrefWidth(200);
        Label smoothValue = new Label(String.valueOf(smoothness));
        smoothValue.setMinWidth(40);
        smoothSlider.valueProperty().addListener((obs, oldVal, newVal) -> {
            smoothValue.setText(String.valueOf(newVal.intValue()));
        });
        smoothRow.getChildren().addAll(smoothLabel, smoothSlider, smoothValue);
        dialog.addCustomContent(smoothRow);

        dialog.setOnOk(() -> {
            radius = (int) radiusSlider.getValue();
            smoothness = (int) smoothSlider.getValue();
        });
    }

    // ===== Serialization =====

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("radius", radius);
        json.addProperty("smoothness", smoothness);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        radius = getJsonInt(json, "radius", getDefaultRadius());
        smoothness = getJsonInt(json, "smoothness", 0);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        radius = getInt(node.properties, "radius", getDefaultRadius());
        smoothness = getInt(node.properties, "smoothness", 0);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("radius", radius);
        node.properties.put("smoothness", smoothness);
    }
}
