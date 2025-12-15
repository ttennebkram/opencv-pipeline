package com.ttennebkram.pipeline.effects.refraction;

import com.ttennebkram.pipeline.effects.BaseEffect;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Cut Glass Refraction Effect
 *
 * Static geometric facets that refract the image, creating a cut-glass or
 * prismatic effect with square-framed facets and beveled edges.
 *
 * Port of the provided Python/OpenCV implementation.
 */
public class RefractionCutGlass extends BaseEffect {

    private final int facetSize = 120;

    // Precomputed displacement + alpha
    private Mat offsetX32F;   // CV_32FC1
    private Mat offsetY32F;   // CV_32FC1
    private Mat alpha32F;     // CV_32FC1

    // Precomputed remap coordinate maps
    private Mat mapX32F;      // CV_32FC1 (x + offsetX)
    private Mat mapY32F;      // CV_32FC1 (y + offsetY)

    // Precomputed 3-channel alpha for blending
    private Mat alpha3_32F;   // CV_32FC3

    // Reusable temporaries to reduce allocations
    private final Mat refracted8U = new Mat();
    private final Mat frame32F = new Mat();
    private final Mat refracted32F = new Mat();
    private final Mat invAlpha3_32F = new Mat();
    private final Mat partA = new Mat();
    private final Mat partB = new Mat();
    private final Mat out32F = new Mat();

    public RefractionCutGlass(int width, int height) {
        super(width, height);
        createCutGlassPatternAndMaps();
    }

    public static String getName() {
        return "Cut Glass";
    }

    public static String getDescription() {
        return "Static cut-glass pattern with prismatic refraction effects";
    }

    public static String getCategory() {
        return "refraction";
    }

    @Override
    public void update() {
        // Static effect: nothing to animate
    }

    @Override
    public Mat draw(Mat frame, Mat faceMask /* nullable */) {
        // This effect ignores faceMask and applies globally (matches Python).

        // Safety: ensure expected size
        // (If your pipeline can change resolution dynamically, you should rebuild maps.)
        if (frame.cols() != width || frame.rows() != height) {
            // Best effort: operate on actual frame size by falling back to direct return.
            // You could also rebuild all maps here.
            return frame;
        }

        // 1) Refract using remap (equivalent to Python advanced indexing)
        Imgproc.remap(
                frame,                  // src
                refracted8U,            // dst
                mapX32F,                // map1
                mapY32F,                // map2
                Imgproc.INTER_LINEAR,
                Core.BORDER_REPLICATE,
                new Scalar(0, 0, 0)
        );

        // 2) Alpha blend: result = refracted * alpha + frame * (1 - alpha)
        frame.convertTo(frame32F, CvType.CV_32FC3, 1.0 / 255.0);
        refracted8U.convertTo(refracted32F, CvType.CV_32FC3, 1.0 / 255.0);

        // invAlpha3 = 1 - alpha3
        Mat ones = Mat.ones(alpha3_32F.rows(), alpha3_32F.cols(), alpha3_32F.type());
        Core.subtract(ones, alpha3_32F, invAlpha3_32F);
        ones.release();

        // partA = refracted * alpha
        Core.multiply(refracted32F, alpha3_32F, partA);

        // partB = frame * (1 - alpha)
        Core.multiply(frame32F, invAlpha3_32F, partB);

        // out = partA + partB
        Core.add(partA, partB, out32F);

        Mat out8U = new Mat();
        out32F.convertTo(out8U, CvType.CV_8UC3, 255.0);
        return out8U;
    }

    private void createCutGlassPatternAndMaps() {
        offsetX32F = Mat.zeros(height, width, CvType.CV_32FC1);
        offsetY32F = Mat.zeros(height, width, CvType.CV_32FC1);
        alpha32F   = Mat.zeros(height, width, CvType.CV_32FC1);

        // Build facets on a regular grid
        for (int row = 0; row < height; row += facetSize) {
            for (int col = 0; col < width; col += facetSize) {
                addSquareFacet(offsetX32F, offsetY32F, alpha32F, col, row, facetSize);
            }
        }

        // Blur displacement for smoother transitions (matches Python GaussianBlur(5x5))
        Imgproc.GaussianBlur(offsetX32F, offsetX32F, new Size(5, 5), 0);
        Imgproc.GaussianBlur(offsetY32F, offsetY32F, new Size(5, 5), 0);

        // Precompute mapX/mapY grids: mapX = x + offsetX, mapY = y + offsetY
        mapX32F = new Mat(height, width, CvType.CV_32FC1);
        mapY32F = new Mat(height, width, CvType.CV_32FC1);

        // Fill coordinate grids + offsets
        float[] oxRow = new float[width];
        float[] oyRow = new float[width];
        float[] mxRow = new float[width];
        float[] myRow = new float[width];

        for (int y = 0; y < height; y++) {
            offsetX32F.get(y, 0, oxRow);
            offsetY32F.get(y, 0, oyRow);

            for (int x = 0; x < width; x++) {
                float sx = x + oxRow[x];
                float sy = y + oyRow[x];

                // clip(0, width-1/height-1) as in Python
                if (sx < 0) sx = 0;
                if (sx > width - 1) sx = width - 1;
                if (sy < 0) sy = 0;
                if (sy > height - 1) sy = height - 1;

                mxRow[x] = sx;
                myRow[x] = sy;
            }

            mapX32F.put(y, 0, mxRow);
            mapY32F.put(y, 0, myRow);
        }

        // Precompute alpha3 for blending
        alpha3_32F = new Mat();
        List<Mat> chans = new ArrayList<>(3);
        chans.add(alpha32F);
        chans.add(alpha32F);
        chans.add(alpha32F);
        Core.merge(chans, alpha3_32F);
    }

    private void addSquareFacet(Mat offsetX, Mat offsetY, Mat alpha,
                               int cornerX, int cornerY, int size) {

        final int centerX = cornerX + size / 2;
        final int centerY = cornerY + size / 2;

        final int frameThickness = 0; // No dark frame between squares
        final int flatBorder = 8;
        final int transitionZone = size / 16;

        // We will operate row-by-row via float arrays for speed and simplicity.
        // (Still a faithful port; avoids per-pixel Mat.get/put overhead.)
        float[] oxRow = new float[width];
        float[] oyRow = new float[width];
        float[] aRow  = new float[width];

        int yStart = Math.max(0, cornerY);
        int yEnd   = Math.min(height, cornerY + size);
        int xStart = Math.max(0, cornerX);
        int xEnd   = Math.min(width, cornerX + size);

        for (int y = yStart; y < yEnd; y++) {

            // Pull current rows
            offsetX.get(y, 0, oxRow);
            offsetY.get(y, 0, oyRow);
            alpha.get(y, 0, aRow);

            for (int x = xStart; x < xEnd; x++) {
                int distLeft   = x - cornerX;
                int distRight  = (cornerX + size) - x;
                int distTop    = y - cornerY;
                int distBottom = (cornerY + size) - y;

                int dx = x - centerX;
                int dy = y - centerY;

                int minEdgeDist = Math.min(Math.min(distLeft, distRight), Math.min(distTop, distBottom));

                float refractX = 0.0f;
                float refractY = 0.0f;

                if (minEdgeDist < frameThickness) {
                    aRow[x] = 0.0f;
                } else if (minEdgeDist < flatBorder) {
                    oxRow[x] = 0.0f;
                    oyRow[x] = 0.0f;
                    aRow[x] = 0.0f;
                } else if (minEdgeDist < flatBorder + transitionZone) {
                    float transitionProgress = (minEdgeDist - flatBorder) / (float) transitionZone;
                    float bevelStrength = 1.0f - transitionProgress;

                    if (distLeft == minEdgeDist) {
                        refractX = -35.0f * bevelStrength;
                        refractY = 0.0f;
                    } else if (distRight == minEdgeDist) {
                        refractX = 35.0f * bevelStrength;
                        refractY = 0.0f;
                    } else if (distTop == minEdgeDist) {
                        refractX = 0.0f;
                        refractY = -35.0f * bevelStrength;
                    } else {
                        refractX = 0.0f;
                        refractY = 35.0f * bevelStrength;
                    }

                    oxRow[x] = refractX;
                    oyRow[x] = refractY;
                    aRow[x] = 1.0f * bevelStrength;

                } else {
                    // Valley center: crosshatch diagonal pattern
                    int diagonalSpacing = 40;
                    int diagonalWidth = 1;

                    int diagonalPos1 = mod(dx + dy, diagonalSpacing);
                    int diagonalPos2 = mod(dx - dy, diagonalSpacing);

                    float distFromRidge1 = Math.abs(diagonalPos1 - diagonalSpacing / 2.0f);
                    float distFromRidge2 = Math.abs(diagonalPos2 - diagonalSpacing / 2.0f);
                    float distFromRidge = Math.min(distFromRidge1, distFromRidge2);

                    if (distFromRidge < diagonalWidth) {
                        float ridgeRoundness = 1.0f - (distFromRidge / diagonalWidth);
                        float ridgeHeight = ridgeRoundness * 0.6f;

                        refractX = 3.0f * (1.0f - ridgeHeight);
                        refractY = 3.0f * (1.0f - ridgeHeight);
                        aRow[x] = 0.2f + 0.5f * ridgeHeight;
                    } else {
                        float denom = (diagonalSpacing / 2.0f - diagonalWidth);
                        float valleyDepth = (distFromRidge - diagonalWidth) / denom;
                        if (valleyDepth > 1.0f) valleyDepth = 1.0f;

                        // Directional refraction based on dominant axis (matches Python)
                        if (Math.abs(dx) > Math.abs(dy)) {
                            if (dx > 0) {
                                refractX = 70.0f * valleyDepth;
                                refractY = (Math.abs(dx) > 0)
                                        ? (dy / (float) Math.abs(dx) * 45.0f * valleyDepth)
                                        : 0.0f;
                            } else {
                                refractX = -70.0f * valleyDepth;
                                refractY = (Math.abs(dx) > 0)
                                        ? (dy / (float) Math.abs(dx) * 45.0f * valleyDepth)
                                        : 0.0f;
                            }
                        } else {
                            if (dy > 0) {
                                refractY = 70.0f * valleyDepth;
                                refractX = (Math.abs(dy) > 0)
                                        ? (dx / (float) Math.abs(dy) * 45.0f * valleyDepth)
                                        : 0.0f;
                            } else {
                                refractY = -70.0f * valleyDepth;
                                refractX = (Math.abs(dy) > 0)
                                        ? (dx / (float) Math.abs(dy) * 45.0f * valleyDepth)
                                        : 0.0f;
                            }
                        }

                        aRow[x] = 1.0f * valleyDepth;
                    }

                    oxRow[x] = refractX;
                    oyRow[x] = refractY;
                }
            }

            // Write rows back
            offsetX.put(y, 0, oxRow);
            offsetY.put(y, 0, oyRow);
            alpha.put(y, 0, aRow);
        }
    }

    private static int mod(int a, int m) {
        int r = a % m;
        return (r < 0) ? (r + m) : r;
    }
}


