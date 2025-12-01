package com.ttennebkram.pipeline.fx.processors;

import org.opencv.core.Mat;

/**
 * FFT Low Pass filter processor.
 * Removes high frequency details, creating a blur effect.
 */
@FXProcessorInfo(nodeType = "FFTLowPass", category = "FFT")
public class FFTLowPassProcessor extends FFTFilterBase {

    @Override
    public String getNodeType() {
        return "FFTLowPass";
    }

    @Override
    public String getDescription() {
        return "FFT Low Pass Filter\nRemoves high frequencies (blurs image)";
    }

    @Override
    protected int getDefaultRadius() {
        return 100;
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }
        return applyFFTFilter(input, true);
    }
}
