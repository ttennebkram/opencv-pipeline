package com.ttennebkram.pipeline.processing;

import org.opencv.core.Mat;

/**
 * Interface for pure OpenCV image processing operations.
 * No UI dependencies - just takes a Mat and returns a processed Mat.
 */
@FunctionalInterface
public interface ImageProcessor {
    /**
     * Process an input image and return the result.
     *
     * @param input The input image (caller owns this Mat)
     * @return The processed output image (caller must release when done)
     */
    Mat process(Mat input);
}
