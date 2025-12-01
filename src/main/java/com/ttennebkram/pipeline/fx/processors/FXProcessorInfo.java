package com.ttennebkram.pipeline.fx.processors;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for FXProcessor classes to declare their metadata.
 * Used for auto-registration at runtime - no compile-time registration needed.
 *
 * Example usage:
 * <pre>
 * {@literal @}FXProcessorInfo(
 *     nodeType = "GaussianBlur",
 *     category = "Blur",
 *     dualInput = false
 * )
 * public class GaussianBlurProcessor extends FXProcessorBase { ... }
 * </pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface FXProcessorInfo {

    /**
     * The node type name (e.g., "GaussianBlur", "CannyEdge").
     * Must match the nodeType used in FXNodeRegistry and serialization.
     */
    String nodeType();

    /**
     * Category for grouping (e.g., "Blur", "Edges", "Basic").
     */
    String category();

    /**
     * Whether this is a dual-input processor.
     * Dual-input processors extend FXDualInputProcessor.
     */
    boolean dualInput() default false;
}
