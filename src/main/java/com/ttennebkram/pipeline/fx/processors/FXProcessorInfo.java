package com.ttennebkram.pipeline.fx.processors;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for FXProcessor classes to declare their metadata.
 * Used for auto-registration at runtime - no compile-time registration needed.
 * The FXNodeRegistry uses this annotation to auto-discover processors.
 *
 * Example usage:
 * <pre>
 * {@literal @}FXProcessorInfo(
 *     nodeType = "GaussianBlur",
 *     displayName = "Gaussian Blur",
 *     category = "Blur",
 *     description = "Gaussian blur\nImgproc.GaussianBlur(src, dst, ksize, sigmaX)"
 * )
 * public class GaussianBlurProcessor extends FXProcessorBase { ... }
 * </pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface FXProcessorInfo {

    /**
     * The node type name (e.g., "GaussianBlur", "CannyEdge").
     * Must match the nodeType used in serialization.
     */
    String nodeType();

    /**
     * Display name shown in the node title (e.g., "Gaussian Blur").
     * If empty, defaults to nodeType.
     */
    String displayName() default "";

    /**
     * Short name for toolbar buttons.
     * If empty, defaults to displayName.
     */
    String buttonName() default "";

    /**
     * Category for grouping (e.g., "Blur", "Edges", "Basic").
     */
    String category();

    /**
     * Description/method signature shown in tooltips.
     */
    String description() default "";

    /**
     * Whether this is a source node (no input required).
     */
    boolean isSource() default false;

    /**
     * Whether this is a dual-input processor.
     * Dual-input processors extend FXDualInputProcessor.
     */
    boolean dualInput() default false;

    /**
     * Whether this is a container node (holds sub-pipeline).
     */
    boolean isContainer() default false;

    /**
     * Number of outputs this node produces.
     */
    int outputCount() default 1;

    /**
     * Whether this node can be disabled.
     * If false, the enable/disable checkbox is not shown.
     * Default is true (most nodes can be disabled).
     */
    boolean canBeDisabled() default true;
}
