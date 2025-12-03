package com.ttennebkram.pipeline.fx.processors;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * FFT Low-Pass 4-output processor.
 * Removes high frequency details (blur effect).
 *
 * Outputs:
 * [0] = Filtered image (blurred)
 * [1] = Absolute difference (removed detail)
 * [2] = FFT spectrum with filter overlay
 * [3] = Filter response curve graph
 */
@FXProcessorInfo(
    nodeType = "FFTLowPass4",
    displayName = "FFT Low-Pass 4",
    category = "Filter",
    description = "FFT Low-Pass (4 outputs)\nCore.dft() / Core.idft()",
    outputCount = 4
)
public class FFTLowPass4Processor extends FFT4Base {

    @Override
    public String getNodeType() {
        return "FFTLowPass4";
    }

    @Override
    public String getDescription() {
        return "FFT Low-Pass Filter (4 outputs)\nRemoves high frequencies (blur)";
    }

    @Override
    protected int getDefaultRadius() {
        return 100;
    }

    @Override
    protected boolean isHighPass() {
        return false;
    }

    @Override
    protected void addFilterOverlayToSpectrum(Mat spectrum) {
        if (spectrum == null || spectrum.empty()) {
            return;
        }

        int centerX = spectrum.cols() / 2;
        int centerY = spectrum.rows() / 2;

        if (radius <= 0) {
            return;
        }

        // For low-pass: blocked area is OUTSIDE the radius
        // Create a mask for the PASSED region (inside circle)
        Mat circleMask = Mat.zeros(spectrum.rows(), spectrum.cols(), CvType.CV_8U);
        Imgproc.circle(circleMask, new Point(centerX, centerY),
            radius, new Scalar(255), -1);  // White filled circle = passed frequencies

        // Invert to get blocked region (outside circle)
        Mat blockedMask = new Mat();
        Core.bitwise_not(circleMask, blockedMask);

        // Split spectrum into channels
        List<Mat> channels = new ArrayList<>();
        Core.split(spectrum, channels);

        // Blue channel: darken outside circle (multiply by 0.3)
        Mat blueInside = new Mat();
        Mat blueOutside = new Mat();
        channels.get(0).copyTo(blueInside);
        channels.get(0).copyTo(blueOutside);
        Core.multiply(blueOutside, new Scalar(0.3), blueOutside);

        Mat blueResult = new Mat();
        blueInside.copyTo(blueResult, circleMask);
        blueOutside.copyTo(blueResult, blockedMask);
        channels.set(0, blueResult);
        blueInside.release();
        blueOutside.release();

        // Green channel: darken outside circle
        Mat greenInside = new Mat();
        Mat greenOutside = new Mat();
        channels.get(1).copyTo(greenInside);
        channels.get(1).copyTo(greenOutside);
        Core.multiply(greenOutside, new Scalar(0.3), greenOutside);

        Mat greenResult = new Mat();
        greenInside.copyTo(greenResult, circleMask);
        greenOutside.copyTo(greenResult, blockedMask);
        channels.set(1, greenResult);
        greenInside.release();
        greenOutside.release();

        // Red channel: boost outside circle (add 100 for red tint)
        Mat redInside = new Mat();
        Mat redOutside = new Mat();
        channels.get(2).copyTo(redInside);
        channels.get(2).copyTo(redOutside);
        Core.add(redOutside, new Scalar(100), redOutside);

        Mat redResult = new Mat();
        redInside.copyTo(redResult, circleMask);
        redOutside.copyTo(redResult, blockedMask);
        channels.set(2, redResult);
        redInside.release();
        redOutside.release();

        // Merge channels back
        Core.merge(channels, spectrum);

        // Cleanup
        circleMask.release();
        blockedMask.release();
        for (Mat ch : channels) ch.release();

        // Draw bright green outline at radius (the filter boundary)
        Imgproc.circle(spectrum, new Point(centerX, centerY),
            radius, new Scalar(0, 255, 0), 2);
    }
}
