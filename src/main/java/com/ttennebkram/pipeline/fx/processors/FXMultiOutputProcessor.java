package com.ttennebkram.pipeline.fx.processors;

import org.opencv.core.Mat;

/**
 * Interface for processors that produce multiple outputs.
 * Extends FXProcessor with multi-output processing capability.
 *
 * Examples: FFTLowPass4, FFTHighPass4 (4 outputs each)
 */
public interface FXMultiOutputProcessor extends FXProcessor {

    /**
     * Get the number of outputs this processor produces.
     * @return Number of outputs (typically 2-4)
     */
    int getOutputCount();

    /**
     * Process an input image and return multiple outputs.
     *
     * @param input The input Mat (do not modify or release)
     * @return Array of output Mats (caller will release each)
     */
    Mat[] processMultiOutput(Mat input);

    /**
     * Default single-output process returns the first output.
     * This allows multi-output processors to work with single-output systems.
     */
    @Override
    default Mat process(Mat input) {
        Mat[] outputs = processMultiOutput(input);
        if (outputs != null && outputs.length > 0 && outputs[0] != null) {
            // Release other outputs since caller only expects one
            for (int i = 1; i < outputs.length; i++) {
                if (outputs[i] != null) {
                    outputs[i].release();
                }
            }
            return outputs[0];
        }
        return input != null ? input.clone() : null;
    }

    /**
     * Get labels for each output (for UI display and tooltips).
     * @return Array of output labels matching getOutputCount()
     */
    String[] getOutputLabels();
}
