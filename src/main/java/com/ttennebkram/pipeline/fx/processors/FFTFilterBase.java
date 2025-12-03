package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.Slider;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Base class for FFT filter processors.
 * Contains common FFT processing code.
 */
public abstract class FFTFilterBase extends FXProcessorBase {

    // Butterworth filter constants
    protected static final double BUTTERWORTH_ORDER_MAX = 10.0;
    protected static final double BUTTERWORTH_ORDER_MIN = 0.5;
    protected static final double BUTTERWORTH_ORDER_RANGE = 9.5;
    protected static final double BUTTERWORTH_SMOOTHNESS_SCALE = 100.0;
    protected static final double BUTTERWORTH_TARGET_ATTENUATION = 0.03;
    protected static final double BUTTERWORTH_DIVISION_EPSILON = 1e-10;

    // Properties with defaults
    protected int radius = 100;
    protected int smoothness = 0;

    @Override
    public String getCategory() {
        return "FFT";
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        Slider radiusSlider = dialog.addSlider("Radius:", 0, 200, radius, "%.0f");
        Slider smoothnessSlider = dialog.addSlider("Smoothness:", 0, 100, smoothness, "%.0f");
        dialog.addDescription("Note: FFT processing is computationally expensive (it's slow!). Consider resizing the image before input to FFT.");

        dialog.setOnOk(() -> {
            radius = (int) radiusSlider.getValue();
            smoothness = (int) smoothnessSlider.getValue();
        });
    }

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

    protected abstract int getDefaultRadius();

    // ===== FFT Helper Methods =====

    protected int nextPowerOf2(int n) {
        int power = 1;
        while (power < n) {
            power *= 2;
        }
        return power;
    }

    protected int getOptimalFFTSize(int n) {
        int pow2 = nextPowerOf2(n);
        int optimal = Core.getOptimalDFTSize(n);
        if (pow2 <= optimal * 1.25) {
            return pow2;
        }
        return optimal % 2 == 0 ? optimal : optimal + 1;
    }

    protected void fftShift(Mat input) {
        int cx = input.cols() / 2;
        int cy = input.rows() / 2;

        Mat q0 = new Mat(input, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(input, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(input, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(input, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);

        tmp.release();
    }

    protected Mat createLowPassMask(int rows, int cols, int maskRadius, int maskSmoothness) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (maskRadius == 0) {
            mask.setTo(new Scalar(0.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (maskSmoothness == 0) {
            mask.setTo(new Scalar(0.0));
            Imgproc.circle(mask, new Point(ccol, crow), maskRadius, new Scalar(1.0), -1);
        } else {
            float[] maskData = new float[rows * cols];
            double order = BUTTERWORTH_ORDER_MAX - (maskSmoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) {
                order = BUTTERWORTH_ORDER_MIN;
            }
            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = maskRadius * shiftFactor;
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

    protected Mat createHighPassMask(int rows, int cols, int maskRadius, int maskSmoothness) {
        Mat mask = new Mat(rows, cols, CvType.CV_32F);

        if (maskRadius == 0) {
            mask.setTo(new Scalar(1.0));
            return mask;
        }

        int crow = rows / 2;
        int ccol = cols / 2;

        if (maskSmoothness == 0) {
            mask.setTo(new Scalar(1.0));
            Imgproc.circle(mask, new Point(ccol, crow), maskRadius, new Scalar(0.0), -1);
        } else {
            float[] maskData = new float[rows * cols];
            double order = BUTTERWORTH_ORDER_MAX - (maskSmoothness / BUTTERWORTH_SMOOTHNESS_SCALE) * BUTTERWORTH_ORDER_RANGE;
            if (order < BUTTERWORTH_ORDER_MIN) {
                order = BUTTERWORTH_ORDER_MIN;
            }
            double shiftFactor = Math.pow(1.0 / BUTTERWORTH_TARGET_ATTENUATION - 1.0, 1.0 / (2.0 * order));
            double effectiveCutoff = maskRadius * shiftFactor;
            double twoN = 2 * order;

            for (int y = 0; y < rows; y++) {
                int dy = y - crow;
                int dy2 = dy * dy;
                for (int x = 0; x < cols; x++) {
                    int dx = x - ccol;
                    double distance = Math.sqrt(dx * dx + dy2);
                    double ratio = (distance + BUTTERWORTH_DIVISION_EPSILON) / effectiveCutoff;
                    float lpValue = (float) (1.0 / (1.0 + Math.pow(ratio, twoN)));
                    maskData[y * cols + x] = 1.0f - lpValue;
                }
            }
            mask.put(0, 0, maskData);
        }

        return mask;
    }

    protected Mat applyFFTFilterToChannel(Mat channel, boolean isLowPass) {
        int origRows = channel.rows();
        int origCols = channel.cols();

        int optRows = getOptimalFFTSize(origRows);
        int optCols = getOptimalFFTSize(origCols);

        Mat padded = new Mat();
        Core.copyMakeBorder(channel, padded, 0, optRows - origRows, 0, optCols - origCols,
            Core.BORDER_CONSTANT, Scalar.all(0));

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

        Core.dft(complexI, complexI);
        fftShift(complexI);

        Mat mask = isLowPass ? createLowPassMask(optRows, optCols, radius, smoothness)
                            : createHighPassMask(optRows, optCols, radius, smoothness);

        List<Mat> dftPlanes = new ArrayList<>();
        Core.split(complexI, dftPlanes);
        complexI.release();

        Core.multiply(dftPlanes.get(0), mask, dftPlanes.get(0));
        Core.multiply(dftPlanes.get(1), mask, dftPlanes.get(1));
        mask.release();

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

        Mat output = new Mat();
        result.convertTo(output, CvType.CV_8U);
        result.release();

        return output;
    }

    protected Mat applyFFTFilter(Mat input, boolean isLowPass) {
        if (input == null || input.empty()) {
            return input;
        }

        List<Mat> channels = new ArrayList<>();
        Core.split(input, channels);

        try {
            List<Mat> filteredChannels = new ArrayList<>();
            try {
                for (Mat channel : channels) {
                    Mat filtered = applyFFTFilterToChannel(channel, isLowPass);
                    filteredChannels.add(filtered);
                }

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
}
