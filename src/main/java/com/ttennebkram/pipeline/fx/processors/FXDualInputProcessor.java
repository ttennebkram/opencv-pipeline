package com.ttennebkram.pipeline.fx.processors;

import com.ttennebkram.pipeline.processing.DualImageProcessor;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Base class for dual-input processors.
 * Extends FXProcessorBase with dual-input specific functionality.
 */
public abstract class FXDualInputProcessor extends FXProcessorBase {

    /**
     * Process two input images.
     * Either input may be null if not yet received.
     *
     * @param input1 First input image (may be null)
     * @param input2 Second input image (may be null)
     * @return Processed output image
     */
    public abstract Mat processDual(Mat input1, Mat input2);

    /**
     * Single-input process() is not used for dual-input processors.
     * Throws UnsupportedOperationException.
     */
    @Override
    public Mat process(Mat input) {
        throw new UnsupportedOperationException(
                "Dual-input processors must use processDual(Mat, Mat)");
    }

    /**
     * Create a DualImageProcessor lambda for use with DualInputProcessor.
     */
    public DualImageProcessor createDualImageProcessor() {
        return this::processDual;
    }

    /**
     * Helper to resize input2 to match input1 dimensions if needed.
     * Returns a new Mat if resize was needed, otherwise returns input2 unchanged.
     * Caller must release the returned Mat if it's different from input2.
     */
    protected Mat resizeToMatch(Mat input1, Mat input2) {
        if (input1.width() != input2.width() || input1.height() != input2.height()) {
            Mat resized = new Mat();
            Imgproc.resize(input2, resized, new Size(input1.width(), input1.height()));
            return resized;
        }
        return input2;
    }

    /**
     * Helper to convert input2 to match input1's type if needed.
     * Returns a new Mat if conversion was needed, otherwise returns input2 unchanged.
     * Caller must release the returned Mat if it's different from input2.
     */
    protected Mat convertToMatch(Mat input1, Mat input2) {
        if (input1.type() != input2.type()) {
            Mat converted = new Mat();
            input2.convertTo(converted, input1.type());
            return converted;
        }
        return input2;
    }

    /**
     * Helper to prepare input2 to match input1 (resize and convert type).
     * Returns an array: [preparedMat, needsRelease]
     * If needsRelease is true, caller must release preparedMat when done.
     */
    protected PreparedInput prepareInput2(Mat input1, Mat input2) {
        Mat prepared = input2;
        boolean needsRelease = false;

        // Resize if dimensions don't match
        if (input1.width() != input2.width() || input1.height() != input2.height()) {
            prepared = new Mat();
            Imgproc.resize(input2, prepared, new Size(input1.width(), input1.height()));
            needsRelease = true;
        }

        // Convert type if needed
        if (input1.type() != prepared.type()) {
            Mat converted = new Mat();
            prepared.convertTo(converted, input1.type());
            if (needsRelease) {
                prepared.release();
            }
            prepared = converted;
            needsRelease = true;
        }

        return new PreparedInput(prepared, needsRelease);
    }

    /**
     * Helper class to track whether a prepared input needs to be released.
     */
    protected static class PreparedInput {
        public final Mat mat;
        public final boolean needsRelease;

        public PreparedInput(Mat mat, boolean needsRelease) {
            this.mat = mat;
            this.needsRelease = needsRelease;
        }

        public void releaseIfNeeded() {
            if (needsRelease && mat != null) {
                mat.release();
            }
        }
    }
}
