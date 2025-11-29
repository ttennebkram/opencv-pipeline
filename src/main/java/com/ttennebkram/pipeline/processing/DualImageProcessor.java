package com.ttennebkram.pipeline.processing;

import org.opencv.core.Mat;

/**
 * Functional interface for processing two input images.
 * Used by dual-input nodes like AddClamp, SubtractClamp, BitwiseAnd, etc.
 */
@FunctionalInterface
public interface DualImageProcessor {
    /**
     * Process two input images and return the result.
     * Either input may be null if not yet received.
     *
     * @param input1 First input image (may be null)
     * @param input2 Second input image (may be null)
     * @return Processed output image, or null if processing failed
     */
    Mat process(Mat input1, Mat input2);
}
