package com.ttennebkram.pipeline.fx;

import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.ByteBuffer;

/**
 * Utility methods for converting between OpenCV Mat and JavaFX Image.
 */
public class FXImageUtils {

    /**
     * Convert an OpenCV Mat to a JavaFX Image.
     * Handles BGR, BGRA, and grayscale formats.
     *
     * @param mat The OpenCV Mat to convert
     * @return A JavaFX Image, or null if conversion fails
     */
    public static Image matToImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        int width = mat.width();
        int height = mat.height();
        int channels = mat.channels();

        // Convert to RGB/RGBA format for JavaFX
        Mat converted = new Mat();
        try {
            if (channels == 1) {
                // Grayscale to RGB
                Imgproc.cvtColor(mat, converted, Imgproc.COLOR_GRAY2RGB);
                channels = 3;
            } else if (channels == 3) {
                // BGR to RGB
                Imgproc.cvtColor(mat, converted, Imgproc.COLOR_BGR2RGB);
            } else if (channels == 4) {
                // BGRA to RGBA
                Imgproc.cvtColor(mat, converted, Imgproc.COLOR_BGRA2RGBA);
            } else {
                System.err.println("Unsupported channel count: " + channels);
                return null;
            }

            // Create byte array from Mat
            int bufferSize = width * height * channels;
            byte[] buffer = new byte[bufferSize];
            converted.get(0, 0, buffer);

            // Create WritableImage
            WritableImage image = new WritableImage(width, height);
            PixelWriter writer = image.getPixelWriter();

            if (channels == 3) {
                // RGB format
                writer.setPixels(0, 0, width, height,
                    PixelFormat.getByteRgbInstance(),
                    buffer, 0, width * 3);
            } else if (channels == 4) {
                // RGBA format
                writer.setPixels(0, 0, width, height,
                    PixelFormat.getByteBgraInstance(),
                    buffer, 0, width * 4);
            }

            return image;
        } finally {
            converted.release();
        }
    }

    /**
     * Convert an OpenCV Mat to a JavaFX Image with scaling.
     *
     * @param mat The OpenCV Mat to convert
     * @param maxWidth Maximum width for the output image
     * @param maxHeight Maximum height for the output image
     * @return A scaled JavaFX Image, or null if conversion fails
     */
    public static Image matToImage(Mat mat, int maxWidth, int maxHeight) {
        if (mat == null || mat.empty()) {
            return null;
        }

        // Calculate scale to fit within bounds
        double scale = Math.min(
            (double) maxWidth / mat.width(),
            (double) maxHeight / mat.height()
        );

        if (scale >= 1.0) {
            // No scaling needed
            return matToImage(mat);
        }

        // Scale the image
        Mat scaled = new Mat();
        try {
            Imgproc.resize(mat, scaled,
                new org.opencv.core.Size(mat.width() * scale, mat.height() * scale));
            return matToImage(scaled);
        } finally {
            scaled.release();
        }
    }

    /**
     * Create a thumbnail image from a Mat.
     *
     * @param mat The source Mat
     * @param thumbWidth Thumbnail width
     * @param thumbHeight Thumbnail height
     * @return A thumbnail JavaFX Image
     */
    public static Image createThumbnail(Mat mat, int thumbWidth, int thumbHeight) {
        return matToImage(mat, thumbWidth, thumbHeight);
    }
}
