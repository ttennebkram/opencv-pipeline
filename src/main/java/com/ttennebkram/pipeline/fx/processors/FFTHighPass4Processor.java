package com.ttennebkram.pipeline.fx.processors;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * FFT High-Pass 4-output processor.
 * Removes low frequency content (edge enhancement).
 *
 * Outputs:
 * [0] = Filtered image (edges enhanced)
 * [1] = Absolute difference (removed smoothness)
 * [2] = FFT spectrum with filter overlay
 * [3] = Filter response curve graph
 */
@FXProcessorInfo(nodeType = "FFTHighPass4", category = "FFT")
public class FFTHighPass4Processor extends FFT4Base {

    @Override
    public String getNodeType() {
        return "FFTHighPass4";
    }

    @Override
    public String getDescription() {
        return "FFT High-Pass Filter (4 outputs)\nRemoves low frequencies (edges)";
    }

    @Override
    protected int getDefaultRadius() {
        return 30;
    }

    @Override
    protected boolean isHighPass() {
        return true;
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

        // For high-pass: blocked area is INSIDE the radius (center)
        // Create a mask for the blocked region (inside circle)
        Mat circleMask = Mat.zeros(spectrum.rows(), spectrum.cols(), CvType.CV_8U);
        Imgproc.circle(circleMask, new Point(centerX, centerY),
            radius, new Scalar(255), -1);  // White filled circle

        // Inverse mask (for keeping original values outside)
        Mat inverseMask = new Mat();
        Core.bitwise_not(circleMask, inverseMask);

        // Split spectrum into channels
        List<Mat> channels = new ArrayList<>();
        Core.split(spectrum, channels);

        // Blue channel: darken inside circle (multiply by 0.3)
        Mat blueInside = new Mat();
        channels.get(0).copyTo(blueInside);
        Core.multiply(blueInside, new Scalar(0.3), blueInside);

        Mat blueResult = new Mat();
        channels.get(0).copyTo(blueResult, inverseMask);  // Keep original outside
        blueInside.copyTo(blueResult, circleMask);        // Use darkened inside
        channels.set(0, blueResult);
        blueInside.release();

        // Green channel: darken inside circle
        Mat greenInside = new Mat();
        channels.get(1).copyTo(greenInside);
        Core.multiply(greenInside, new Scalar(0.3), greenInside);

        Mat greenResult = new Mat();
        channels.get(1).copyTo(greenResult, inverseMask);
        greenInside.copyTo(greenResult, circleMask);
        channels.set(1, greenResult);
        greenInside.release();

        // Red channel: boost inside circle (add 100 for red tint)
        Mat redInside = new Mat();
        channels.get(2).copyTo(redInside);
        Core.add(redInside, new Scalar(100), redInside);

        Mat redResult = new Mat();
        channels.get(2).copyTo(redResult, inverseMask);
        redInside.copyTo(redResult, circleMask);
        channels.set(2, redResult);
        redInside.release();

        // Merge channels back
        Core.merge(channels, spectrum);

        // Cleanup
        circleMask.release();
        inverseMask.release();
        for (Mat ch : channels) ch.release();

        // Draw bright green outline at radius (the filter boundary)
        Imgproc.circle(spectrum, new Point(centerX, centerY),
            radius, new Scalar(0, 255, 0), 2);
    }
}
