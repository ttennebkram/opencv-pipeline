package com.ttennebkram.pipeline.registry;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Annotation for pipeline node classes to declare their metadata.
 * Used for auto-registration and serialization.
 *
 * Example usage:
 * <pre>
 * {@literal @}NodeInfo(
 *     name = "GaussianBlur",
 *     category = "Blur",
 *     aliases = {"Gaussian Blur"}
 * )
 * public class GaussianBlurNode extends ProcessingNode { ... }
 * </pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface NodeInfo {

    /**
     * Canonical name used for serialization (e.g., "GaussianBlur").
     * This is the primary identifier stored in JSON files.
     */
    String name();

    /**
     * Category for toolbar grouping (e.g., "Blur", "Edge Detection").
     */
    String category();

    /**
     * Additional aliases for backward compatibility and user convenience.
     * The registry automatically adds "Untitled: {name}" and "Unknown: {name}" aliases.
     */
    String[] aliases() default {};
}
