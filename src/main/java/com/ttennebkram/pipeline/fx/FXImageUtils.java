package com.ttennebkram.pipeline.fx;

import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.Image;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.nio.ByteBuffer;

/**
 * Utility methods for converting between OpenCV Mat and JavaFX Image.
 */
public class FXImageUtils {

    // Set to true to use pure JavaFX approach (no AWT/Swing dependency)
    // Set to false to use BufferedImage + SwingFXUtils (more reliable but requires javafx-swing)
    private static final boolean USE_PURE_JAVAFX = true;

    /**
     * Convert an OpenCV Mat to a JavaFX Image.
     *
     * @param mat The OpenCV Mat to convert
     * @return A JavaFX Image, or null if conversion fails
     */
    public static Image matToImage(Mat mat) {
        if (mat == null || mat.empty()) {
            return null;
        }

        return USE_PURE_JAVAFX ? matToImagePureJavaFX(mat) : matToImageViaBufferedImage(mat);
    }

    /**
     * Pure JavaFX implementation - no AWT/Swing dependency.
     * Uses WritableImage and PixelWriter directly.
     */
    private static Image matToImagePureJavaFX(Mat mat) {
        try {
            int width = mat.width();
            int height = mat.height();
            int channels = mat.channels();

            // Convert BGR to RGB for 3-channel images
            Mat rgbMat;
            if (channels == 3) {
                rgbMat = new Mat();
                Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGR2RGB);
            } else if (channels == 4) {
                rgbMat = new Mat();
                Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_BGRA2RGBA);
            } else if (channels == 1) {
                // Convert grayscale to RGB
                rgbMat = new Mat();
                Imgproc.cvtColor(mat, rgbMat, Imgproc.COLOR_GRAY2RGB);
            } else {
                return null;
            }

            // Get pixel data
            int bufferSize = rgbMat.channels() * width * height;
            byte[] buffer = new byte[bufferSize];
            rgbMat.get(0, 0, buffer);

            // Create WritableImage
            WritableImage image = new WritableImage(width, height);
            PixelWriter pw = image.getPixelWriter();

            // Use appropriate pixel format based on channels
            if (rgbMat.channels() == 3) {
                pw.setPixels(0, 0, width, height,
                    PixelFormat.getByteRgbInstance(),
                    buffer, 0, width * 3);
            } else if (rgbMat.channels() == 4) {
                pw.setPixels(0, 0, width, height,
                    PixelFormat.getByteBgraInstance(),
                    buffer, 0, width * 4);
            }

            // Release temp mat if we created one
            if (rgbMat != mat) {
                rgbMat.release();
            }

            return image;

        } catch (Exception e) {
            System.err.println("matToImagePureJavaFX error: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    /**
     * BufferedImage + SwingFXUtils implementation.
     * More reliable but requires javafx-swing dependency.
     */
    private static Image matToImageViaBufferedImage(Mat mat) {
        try {
            int width = mat.width();
            int height = mat.height();
            int channels = mat.channels();

            // Determine BufferedImage type
            int bufferedImageType;
            Mat converted;

            if (channels == 1) {
                // Grayscale
                bufferedImageType = BufferedImage.TYPE_BYTE_GRAY;
                converted = mat;
            } else if (channels == 3) {
                // BGR to RGB
                bufferedImageType = BufferedImage.TYPE_3BYTE_BGR;
                converted = mat; // Keep as BGR for BufferedImage
            } else if (channels == 4) {
                // BGRA
                bufferedImageType = BufferedImage.TYPE_4BYTE_ABGR;
                converted = new Mat();
                Imgproc.cvtColor(mat, converted, Imgproc.COLOR_BGRA2RGBA);
            } else {
                System.err.println("Unsupported channel count: " + channels);
                return null;
            }

            // Create BufferedImage
            BufferedImage bufferedImage = new BufferedImage(width, height, bufferedImageType);
            byte[] targetPixels = ((DataBufferByte) bufferedImage.getRaster().getDataBuffer()).getData();

            // Copy Mat data to BufferedImage
            converted.get(0, 0, targetPixels);

            // Release converted mat if we created one
            if (converted != mat) {
                converted.release();
            }

            // Convert BufferedImage to JavaFX Image
            return SwingFXUtils.toFXImage(bufferedImage, null);

        } catch (Exception e) {
            System.err.println("matToImageViaBufferedImage error: " + e.getMessage());
            e.printStackTrace();
            return null;
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

        // Scale the image using INTER_AREA for better quality when shrinking
        Mat scaled = new Mat();
        try {
            Imgproc.resize(mat, scaled,
                new org.opencv.core.Size(mat.width() * scale, mat.height() * scale),
                0, 0, Imgproc.INTER_AREA);
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
