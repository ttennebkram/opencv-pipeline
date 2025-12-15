package com.ttennebkram.pipeline.effects;

import org.opencv.core.Mat;

/**
 * Abstract base class for standalone webcam filter effects.
 *
 * Port of Python BaseEffect from webcam-filters project.
 * Effects that are NOT pipeline processors should extend this class.
 */
public abstract class BaseEffect {

    protected final int width;
    protected final int height;

    /**
     * Initialize effect with frame dimensions.
     *
     * @param width  Frame width in pixels
     * @param height Frame height in pixels
     */
    public BaseEffect(int width, int height) {
        this.width = width;
        this.height = height;
    }

    /**
     * Return the display name of this effect.
     *
     * @return Human-readable name (e.g., "Cut Glass")
     */
    public static String getName() {
        return "Unnamed Effect";
    }

    /**
     * Return a brief description of this effect.
     *
     * @return One-line description of what the effect does
     */
    public static String getDescription() {
        return "";
    }

    /**
     * Return the category this effect belongs to.
     *
     * @return Category name (e.g., "refraction", "misc")
     */
    public static String getCategory() {
        return "misc";
    }

    /**
     * Update animation state.
     * Called once per frame before draw(). Override to update positions,
     * timers, and other animation state.
     */
    public void update() {
        // Default: no animation
    }

    /**
     * Apply the effect to a frame.
     *
     * @param frame    Input frame (BGR format)
     * @param faceMask Optional face detection mask (may be null)
     * @return Processed frame (BGR format)
     */
    public abstract Mat draw(Mat frame, Mat faceMask);

    /**
     * Clean up resources when effect is no longer needed.
     * Override if your effect needs to release Mats, close windows, etc.
     */
    public void cleanup() {
        // Default: nothing to clean up
    }

    // Getters
    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
}
