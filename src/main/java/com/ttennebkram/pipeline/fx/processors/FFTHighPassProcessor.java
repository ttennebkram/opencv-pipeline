package com.ttennebkram.pipeline.fx.processors;

import org.opencv.core.Mat;

/**
 * FFT High Pass filter processor.
 * Removes low frequency content, enhancing edges.
 */
@FXProcessorInfo(
    nodeType = "FFTHighPass",
    displayName = "FFT High-Pass",
    category = "Filter",
    description = "FFT High-Pass Filter\nCore.dft() / Core.idft()"
)
public class FFTHighPassProcessor extends FFTFilterBase {

    @Override
    public String getNodeType() {
        return "FFTHighPass";
    }

    @Override
    public String getDescription() {
        return "FFT High Pass Filter\nRemoves low frequencies (enhances edges)";
    }

    @Override
    protected int getDefaultRadius() {
        return 30;
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }
        return applyFFTFilter(input, false);
    }
}
