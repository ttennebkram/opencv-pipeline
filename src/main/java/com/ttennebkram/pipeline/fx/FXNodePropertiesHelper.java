package com.ttennebkram.pipeline.fx;

import javafx.scene.control.*;
import javafx.stage.Window;
import java.util.Map;

/**
 * Helper class for populating FXPropertiesDialog with node-specific properties.
 * This centralizes the property dialog configuration for all node types.
 */
public class FXNodePropertiesHelper {

    /**
     * Add properties to dialog based on node type.
     * Returns true if properties were added, false if node type has no custom properties.
     */
    public static boolean addPropertiesForNode(FXPropertiesDialog dialog, FXNode node) {
        String nodeType = node.nodeType;
        Map<String, Object> props = node.properties;

        switch (nodeType) {
            case "Arrow":
            case "Line":
                addArrowProperties(dialog, props);
                return true;
            case "BlobDetector":
                addBlobDetectorProperties(dialog, props);
                return true;
            case "BoxBlur":
                addBoxBlurProperties(dialog, props);
                return true;
            case "CannyEdge":
                addCannyEdgeProperties(dialog, props);
                return true;
            case "Clone":
                addCloneProperties(dialog, props);
                return true;
            case "ConnectedComponents":
                addConnectedComponentsProperties(dialog, props);
                return true;
            case "Contours":
                addContoursProperties(dialog, props);
                return true;
            case "Ellipse":
                addEllipseProperties(dialog, props);
                return true;
            case "FileSource":
                addFileSourceProperties(dialog, props);
                return true;
            case "GaussianBlur":
                addGaussianBlurProperties(dialog, props);
                return true;
            case "HarrisCorners":
                addHarrisCornersProperties(dialog, props);
                return true;
            case "Histogram":
                addHistogramProperties(dialog, props);
                return true;
            case "HoughCircles":
                addHoughCirclesProperties(dialog, props);
                return true;
            case "HoughLines":
                addHoughLinesProperties(dialog, props);
                return true;
            case "MatchTemplate":
                addMatchTemplateProperties(dialog, props);
                return true;
            case "MeanShift":
                addMeanShiftProperties(dialog, props);
                return true;
            case "MorphologyEx":
                addMorphologyExProperties(dialog, props);
                return true;
            case "ORBFeatures":
                addORBFeaturesProperties(dialog, props);
                return true;
            case "SIFTFeatures":
                addSIFTFeaturesProperties(dialog, props);
                return true;
            case "ShiTomasi":
                addShiTomasiProperties(dialog, props);
                return true;
            case "Text":
                addTextProperties(dialog, props);
                return true;
            case "Threshold":
                addThresholdProperties(dialog, props);
                return true;
            case "WarpAffine":
                addWarpAffineProperties(dialog, props);
                return true;
            case "WebcamSource":
                addWebcamSourceProperties(dialog, props);
                return true;
            default:
                return false;
        }
    }

    /**
     * Save properties from dialog controls back to node.
     */
    public static void savePropertiesForNode(FXNode node) {
        Map<String, Object> props = node.properties;
        String nodeType = node.nodeType;

        // Each node type has its controls stored with underscore prefix
        // The calling code should call this after dialog closes with OK

        switch (nodeType) {
            case "Arrow":
            case "Line":
                saveArrowProperties(props);
                break;
            case "BlobDetector":
                saveBlobDetectorProperties(props);
                break;
            case "BoxBlur":
                saveBoxBlurProperties(props);
                break;
            case "CannyEdge":
                saveCannyEdgeProperties(props);
                break;
            case "Clone":
                saveCloneProperties(props);
                break;
            case "ConnectedComponents":
                saveConnectedComponentsProperties(props);
                break;
            case "Contours":
                saveContoursProperties(props);
                break;
            case "Ellipse":
                saveEllipseProperties(props);
                break;
            case "FileSource":
                saveFileSourceProperties(props);
                break;
            case "GaussianBlur":
                saveGaussianBlurProperties(props);
                break;
            case "HarrisCorners":
                saveHarrisCornersProperties(props);
                break;
            case "Histogram":
                saveHistogramProperties(props);
                break;
            case "HoughCircles":
                saveHoughCirclesProperties(props);
                break;
            case "HoughLines":
                saveHoughLinesProperties(props);
                break;
            case "MatchTemplate":
                saveMatchTemplateProperties(props);
                break;
            case "MeanShift":
                saveMeanShiftProperties(props);
                break;
            case "MorphologyEx":
                saveMorphologyExProperties(props);
                break;
            case "ORBFeatures":
                saveORBFeaturesProperties(props);
                break;
            case "SIFTFeatures":
                saveSIFTFeaturesProperties(props);
                break;
            case "ShiTomasi":
                saveShiTomasiProperties(props);
                break;
            case "Text":
                saveTextProperties(props);
                break;
            case "Threshold":
                saveThresholdProperties(props);
                break;
            case "WarpAffine":
                saveWarpAffineProperties(props);
                break;
            case "WebcamSource":
                saveWebcamSourceProperties(props);
                break;
        }
    }

    // ========== ARROW / LINE ==========
    private static void addArrowProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int x1 = getInt(props, "x1", 50);
        int y1 = getInt(props, "y1", 50);
        int x2 = getInt(props, "x2", 200);
        int y2 = getInt(props, "y2", 150);
        int colorR = getInt(props, "colorR", 0);
        int colorG = getInt(props, "colorG", 255);
        int colorB = getInt(props, "colorB", 0);
        int thickness = getInt(props, "thickness", 2);

        Spinner<Integer> x1Spinner = dialog.addSpinner("X1:", -4096, 4096, x1);
        Spinner<Integer> y1Spinner = dialog.addSpinner("Y1:", -4096, 4096, y1);
        Spinner<Integer> x2Spinner = dialog.addSpinner("X2:", -4096, 4096, x2);
        Spinner<Integer> y2Spinner = dialog.addSpinner("Y2:", -4096, 4096, y2);
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);

        props.put("_x1Spinner", x1Spinner);
        props.put("_y1Spinner", y1Spinner);
        props.put("_x2Spinner", x2Spinner);
        props.put("_y2Spinner", y2Spinner);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
        props.put("_thicknessSpinner", thicknessSpinner);
    }

    private static void saveArrowProperties(Map<String, Object> props) {
        if (props.containsKey("_x1Spinner")) {
            props.put("x1", ((Spinner<Integer>) props.get("_x1Spinner")).getValue());
            props.put("y1", ((Spinner<Integer>) props.get("_y1Spinner")).getValue());
            props.put("x2", ((Spinner<Integer>) props.get("_x2Spinner")).getValue());
            props.put("y2", ((Spinner<Integer>) props.get("_y2Spinner")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            props.put("thickness", ((Spinner<Integer>) props.get("_thicknessSpinner")).getValue());
            cleanupControls(props, "_x1Spinner", "_y1Spinner", "_x2Spinner", "_y2Spinner",
                           "_colorRSpinner", "_colorGSpinner", "_colorBSpinner", "_thicknessSpinner");
        }
    }

    // ========== BLOB DETECTOR ==========
    private static void addBlobDetectorProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int minThreshold = getInt(props, "minThreshold", 10);
        int maxThreshold = getInt(props, "maxThreshold", 200);
        boolean showOriginal = getBool(props, "showOriginal", true);
        boolean filterByArea = getBool(props, "filterByArea", true);
        int minArea = getInt(props, "minArea", 100);
        int maxArea = getInt(props, "maxArea", 5000);
        boolean filterByCircularity = getBool(props, "filterByCircularity", false);
        int minCircularity = getInt(props, "minCircularity", 10);
        boolean filterByConvexity = getBool(props, "filterByConvexity", false);
        int minConvexity = getInt(props, "minConvexity", 87);
        boolean filterByInertia = getBool(props, "filterByInertia", false);
        int minInertiaRatio = getInt(props, "minInertiaRatio", 1);
        boolean filterByColor = getBool(props, "filterByColor", false);
        int blobColor = getInt(props, "blobColor", 0);

        CheckBox showOrigCheck = dialog.addCheckbox("Show Original Background", showOriginal);
        Slider minThreshSlider = dialog.addSlider("Min Threshold:", 0, 255, minThreshold, "%.0f");
        Slider maxThreshSlider = dialog.addSlider("Max Threshold:", 0, 255, maxThreshold, "%.0f");
        CheckBox areaCheck = dialog.addCheckbox("Filter by Area", filterByArea);
        Spinner<Integer> minAreaSpinner = dialog.addSpinner("Min Area:", 1, 10000, minArea);
        Spinner<Integer> maxAreaSpinner = dialog.addSpinner("Max Area:", 1, 50000, maxArea);
        CheckBox circCheck = dialog.addCheckbox("Filter by Circularity", filterByCircularity);
        Slider circSlider = dialog.addSlider("Min Circularity %:", 1, 100, minCircularity, "%.0f");
        CheckBox convCheck = dialog.addCheckbox("Filter by Convexity", filterByConvexity);
        Slider convSlider = dialog.addSlider("Min Convexity %:", 1, 100, minConvexity, "%.0f");
        CheckBox inertiaCheck = dialog.addCheckbox("Filter by Inertia", filterByInertia);
        Slider inertiaSlider = dialog.addSlider("Min Inertia %:", 1, 100, minInertiaRatio, "%.0f");
        CheckBox colorCheck = dialog.addCheckbox("Filter by Color", filterByColor);
        String[] colorOptions = {"Dark (0)", "Light (255)"};
        ComboBox<String> colorCombo = dialog.addComboBox("Blob Color:", colorOptions, colorOptions[blobColor == 0 ? 0 : 1]);

        props.put("_showOrigCheck", showOrigCheck);
        props.put("_minThreshSlider", minThreshSlider);
        props.put("_maxThreshSlider", maxThreshSlider);
        props.put("_filterByAreaCheck", areaCheck);
        props.put("_minAreaSpinner", minAreaSpinner);
        props.put("_maxAreaSpinner", maxAreaSpinner);
        props.put("_filterByCircularityCheck", circCheck);
        props.put("_minCircularitySlider", circSlider);
        props.put("_filterByConvexityCheck", convCheck);
        props.put("_minConvexitySlider", convSlider);
        props.put("_filterByInertiaCheck", inertiaCheck);
        props.put("_minInertiaRatioSlider", inertiaSlider);
        props.put("_filterByColorCheck", colorCheck);
        props.put("_blobColorCombo", colorCombo);
    }

    private static void saveBlobDetectorProperties(Map<String, Object> props) {
        if (props.containsKey("_showOrigCheck")) {
            props.put("showOriginal", ((CheckBox) props.get("_showOrigCheck")).isSelected());
            props.put("minThreshold", (int) ((Slider) props.get("_minThreshSlider")).getValue());
            props.put("maxThreshold", (int) ((Slider) props.get("_maxThreshSlider")).getValue());
            props.put("filterByArea", ((CheckBox) props.get("_filterByAreaCheck")).isSelected());
            props.put("minArea", ((Spinner<Integer>) props.get("_minAreaSpinner")).getValue());
            props.put("maxArea", ((Spinner<Integer>) props.get("_maxAreaSpinner")).getValue());
            props.put("filterByCircularity", ((CheckBox) props.get("_filterByCircularityCheck")).isSelected());
            props.put("minCircularity", (int) ((Slider) props.get("_minCircularitySlider")).getValue());
            props.put("filterByConvexity", ((CheckBox) props.get("_filterByConvexityCheck")).isSelected());
            props.put("minConvexity", (int) ((Slider) props.get("_minConvexitySlider")).getValue());
            props.put("filterByInertia", ((CheckBox) props.get("_filterByInertiaCheck")).isSelected());
            props.put("minInertiaRatio", (int) ((Slider) props.get("_minInertiaRatioSlider")).getValue());
            props.put("filterByColor", ((CheckBox) props.get("_filterByColorCheck")).isSelected());
            @SuppressWarnings("unchecked")
            ComboBox<String> colorCombo = (ComboBox<String>) props.get("_blobColorCombo");
            props.put("blobColor", colorCombo.getSelectionModel().getSelectedIndex() == 0 ? 0 : 255);
            cleanupControls(props, "_showOrigCheck", "_minThreshSlider", "_maxThreshSlider",
                           "_filterByAreaCheck", "_minAreaSpinner", "_maxAreaSpinner",
                           "_filterByCircularityCheck", "_minCircularitySlider",
                           "_filterByConvexityCheck", "_minConvexitySlider",
                           "_filterByInertiaCheck", "_minInertiaRatioSlider",
                           "_filterByColorCheck", "_blobColorCombo");
        }
    }

    // ========== BOX BLUR ==========
    private static void addBoxBlurProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int kx = getInt(props, "kernelSizeX", getInt(props, "ksize", 5));
        int ky = getInt(props, "kernelSizeY", getInt(props, "ksize", 5));
        Slider kxSlider = dialog.addOddKernelSlider("Kernel Size X:", kx);
        Slider kySlider = dialog.addOddKernelSlider("Kernel Size Y:", ky);
        props.put("_kxSlider", kxSlider);
        props.put("_kySlider", kySlider);
    }

    private static void saveBoxBlurProperties(Map<String, Object> props) {
        if (props.containsKey("_kxSlider")) {
            int kx = (int) ((Slider) props.get("_kxSlider")).getValue();
            int ky = (int) ((Slider) props.get("_kySlider")).getValue();
            props.put("kernelSizeX", kx);
            props.put("kernelSizeY", ky);
            props.remove("ksize");  // Remove old property name
            cleanupControls(props, "_kxSlider", "_kySlider");
        }
    }

    // ========== CANNY EDGE ==========
    private static void addCannyEdgeProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int threshold1 = getInt(props, "threshold1", 30);
        int threshold2 = getInt(props, "threshold2", 150);
        int apertureIndex = getInt(props, "apertureIndex", 0);
        boolean l2Gradient = getBool(props, "l2Gradient", false);

        Slider t1Slider = dialog.addSlider("Threshold 1 (lower):", 0, 500, threshold1, "%.0f");
        Slider t2Slider = dialog.addSlider("Threshold 2 (upper):", 0, 500, threshold2, "%.0f");
        String[] apertures = {"3", "5", "7"};
        ComboBox<String> apertureCombo = dialog.addComboBox("Aperture Size:", apertures, apertures[apertureIndex]);
        CheckBox l2Check = dialog.addCheckbox("L2 Gradient", l2Gradient);

        props.put("_threshold1Slider", t1Slider);
        props.put("_threshold2Slider", t2Slider);
        props.put("_apertureCombo", apertureCombo);
        props.put("_l2GradientCheck", l2Check);
    }

    private static void saveCannyEdgeProperties(Map<String, Object> props) {
        if (props.containsKey("_threshold1Slider")) {
            props.put("threshold1", (int) ((Slider) props.get("_threshold1Slider")).getValue());
            props.put("threshold2", (int) ((Slider) props.get("_threshold2Slider")).getValue());
            @SuppressWarnings("unchecked")
            ComboBox<String> combo = (ComboBox<String>) props.get("_apertureCombo");
            props.put("apertureIndex", combo.getSelectionModel().getSelectedIndex());
            props.put("l2Gradient", ((CheckBox) props.get("_l2GradientCheck")).isSelected());
            cleanupControls(props, "_threshold1Slider", "_threshold2Slider", "_apertureCombo", "_l2GradientCheck");
        }
    }

    // ========== CLONE ==========
    private static void addCloneProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int numOutputs = getInt(props, "numOutputs", 2);
        String[] outputs = {"2", "3", "4"};
        ComboBox<String> outputCombo = dialog.addComboBox("Number of Outputs:", outputs, outputs[numOutputs - 2]);
        props.put("_numOutputsCombo", outputCombo);
    }

    private static void saveCloneProperties(Map<String, Object> props) {
        if (props.containsKey("_numOutputsCombo")) {
            @SuppressWarnings("unchecked")
            ComboBox<String> combo = (ComboBox<String>) props.get("_numOutputsCombo");
            props.put("numOutputs", combo.getSelectionModel().getSelectedIndex() + 2);
            cleanupControls(props, "_numOutputsCombo");
        }
    }

    // ========== CONNECTED COMPONENTS ==========
    private static void addConnectedComponentsProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int threshold = getInt(props, "threshold", 127);
        boolean invertThreshold = getBool(props, "invertThreshold", false);
        int connectivity = getInt(props, "connectivity", 8);
        int minArea = getInt(props, "minArea", 0);

        Slider threshSlider = dialog.addSlider("Threshold:", 0, 255, threshold, "%.0f");
        CheckBox invertCheck = dialog.addCheckbox("Invert Threshold", invertThreshold);
        String[] connOptions = {"4", "8"};
        ComboBox<String> connCombo = dialog.addComboBox("Connectivity:", connOptions, connectivity == 4 ? "4" : "8");
        Spinner<Integer> minAreaSpinner = dialog.addSpinner("Min Area:", 0, 10000, minArea);

        props.put("_thresholdSlider", threshSlider);
        props.put("_invertThresholdCheck", invertCheck);
        props.put("_connectivityCombo", connCombo);
        props.put("_minAreaSpinner", minAreaSpinner);
    }

    private static void saveConnectedComponentsProperties(Map<String, Object> props) {
        if (props.containsKey("_thresholdSlider")) {
            props.put("threshold", (int) ((Slider) props.get("_thresholdSlider")).getValue());
            props.put("invertThreshold", ((CheckBox) props.get("_invertThresholdCheck")).isSelected());
            @SuppressWarnings("unchecked")
            ComboBox<String> combo = (ComboBox<String>) props.get("_connectivityCombo");
            props.put("connectivity", combo.getSelectionModel().getSelectedIndex() == 0 ? 4 : 8);
            props.put("minArea", ((Spinner<Integer>) props.get("_minAreaSpinner")).getValue());
            cleanupControls(props, "_thresholdSlider", "_invertThresholdCheck", "_connectivityCombo", "_minAreaSpinner");
        }
    }

    // ========== CONTOURS ==========
    private static void addContoursProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        boolean showOriginal = getBool(props, "showOriginal", false);
        int thickness = getInt(props, "thickness", 2);
        int colorR = getInt(props, "colorR", 0);
        int colorG = getInt(props, "colorG", 255);
        int colorB = getInt(props, "colorB", 0);

        CheckBox showOrigCheckBox = dialog.addCheckbox("Show Original Image", showOriginal);
        Slider thicknessSlider = dialog.addSlider("Thickness:", 1, 10, thickness, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        props.put("_showOrigCheckBox", showOrigCheckBox);
        props.put("_thicknessSlider", thicknessSlider);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
    }

    private static void saveContoursProperties(Map<String, Object> props) {
        if (props.containsKey("_showOrigCheckBox")) {
            props.put("showOriginal", ((CheckBox) props.get("_showOrigCheckBox")).isSelected());
            props.put("thickness", (int) ((Slider) props.get("_thicknessSlider")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            cleanupControls(props, "_showOrigCheckBox", "_thicknessSlider",
                           "_colorRSpinner", "_colorGSpinner", "_colorBSpinner");
        }
    }

    // ========== ELLIPSE ==========
    private static void addEllipseProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int centerX = getInt(props, "centerX", 100);
        int centerY = getInt(props, "centerY", 100);
        int axisX = getInt(props, "axisX", 60);
        int axisY = getInt(props, "axisY", 40);
        int angle = getInt(props, "angle", 0);
        int colorR = getInt(props, "colorR", 0);
        int colorG = getInt(props, "colorG", 255);
        int colorB = getInt(props, "colorB", 0);
        int thickness = getInt(props, "thickness", 2);
        boolean filled = getBool(props, "filled", false);

        Spinner<Integer> cxSpinner = dialog.addSpinner("Center X:", -4096, 4096, centerX);
        Spinner<Integer> cySpinner = dialog.addSpinner("Center Y:", -4096, 4096, centerY);
        Spinner<Integer> axisXSpinner = dialog.addSpinner("Axis X:", 1, 2000, axisX);
        Spinner<Integer> axisYSpinner = dialog.addSpinner("Axis Y:", 1, 2000, axisY);
        Slider angleSlider = dialog.addSlider("Angle:", 0, 360, angle, "%.0f°");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 50, thickness);
        CheckBox filledCheckBox = dialog.addCheckbox("Filled", filled);

        props.put("_centerXSpinner", cxSpinner);
        props.put("_centerYSpinner", cySpinner);
        props.put("_axisXSpinner", axisXSpinner);
        props.put("_axisYSpinner", axisYSpinner);
        props.put("_angleSlider", angleSlider);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
        props.put("_thicknessSpinner", thicknessSpinner);
        props.put("_filledCheckBox", filledCheckBox);
    }

    private static void saveEllipseProperties(Map<String, Object> props) {
        if (props.containsKey("_centerXSpinner")) {
            props.put("centerX", ((Spinner<Integer>) props.get("_centerXSpinner")).getValue());
            props.put("centerY", ((Spinner<Integer>) props.get("_centerYSpinner")).getValue());
            props.put("axisX", ((Spinner<Integer>) props.get("_axisXSpinner")).getValue());
            props.put("axisY", ((Spinner<Integer>) props.get("_axisYSpinner")).getValue());
            props.put("angle", (int) ((Slider) props.get("_angleSlider")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            props.put("thickness", ((Spinner<Integer>) props.get("_thicknessSpinner")).getValue());
            props.put("filled", ((CheckBox) props.get("_filledCheckBox")).isSelected());
            cleanupControls(props, "_centerXSpinner", "_centerYSpinner", "_axisXSpinner", "_axisYSpinner",
                           "_angleSlider", "_colorRSpinner", "_colorGSpinner", "_colorBSpinner",
                           "_thicknessSpinner", "_filledCheckBox");
        }
    }

    // ========== FILE SOURCE ==========
    private static void addFileSourceProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        String imagePath = getString(props, "imagePath", "");
        int fpsMode = getInt(props, "fpsMode", 1);
        boolean loopVideo = getBool(props, "loopVideo", true);

        TextField pathField = dialog.addTextField("Image/Video Path:", imagePath);
        String[] fpsModes = {"Just Once", "Automatic", "1 fps", "5 fps", "10 fps", "15 fps", "24 fps", "30 fps", "60 fps"};
        ComboBox<String> fpsCombo = dialog.addComboBox("FPS Mode:", fpsModes, fpsModes[fpsMode]);
        CheckBox loopCheck = dialog.addCheckbox("Loop Video", loopVideo);

        props.put("_imagePathField", pathField);
        props.put("_fpsModeCombo", fpsCombo);
        props.put("_loopVideoCheck", loopCheck);
    }

    private static void saveFileSourceProperties(Map<String, Object> props) {
        if (props.containsKey("_imagePathField")) {
            props.put("imagePath", ((TextField) props.get("_imagePathField")).getText());
            @SuppressWarnings("unchecked")
            ComboBox<String> combo = (ComboBox<String>) props.get("_fpsModeCombo");
            props.put("fpsMode", combo.getSelectionModel().getSelectedIndex());
            props.put("loopVideo", ((CheckBox) props.get("_loopVideoCheck")).isSelected());
            cleanupControls(props, "_imagePathField", "_fpsModeCombo", "_loopVideoCheck");
        }
    }

    // ========== GAUSSIAN BLUR ==========
    private static void addGaussianBlurProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int kx = getInt(props, "kernelSizeX", getInt(props, "ksize", 7));
        int ky = getInt(props, "kernelSizeY", getInt(props, "ksize", 7));
        double sigmaX = getDouble(props, "sigmaX", 0.0);

        Slider kxSlider = dialog.addOddKernelSlider("Kernel Size X:", kx);
        Slider kySlider = dialog.addOddKernelSlider("Kernel Size Y:", ky);
        Slider sigmaSlider = dialog.addSlider("Sigma X (0=auto):", 0, 100, sigmaX * 10, "%.1f");

        props.put("_kxSlider", kxSlider);
        props.put("_kySlider", kySlider);
        props.put("_sigmaXSlider", sigmaSlider);
    }

    private static void saveGaussianBlurProperties(Map<String, Object> props) {
        if (props.containsKey("_kxSlider")) {
            int kx = (int) ((Slider) props.get("_kxSlider")).getValue();
            int ky = (int) ((Slider) props.get("_kySlider")).getValue();
            props.put("kernelSizeX", kx);
            props.put("kernelSizeY", ky);
            props.put("sigmaX", ((Slider) props.get("_sigmaXSlider")).getValue() / 10.0);
            props.remove("ksize");  // Remove old property name
            cleanupControls(props, "_kxSlider", "_kySlider", "_sigmaXSlider");
        }
    }

    // ========== HARRIS CORNERS ==========
    private static void addHarrisCornersProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        boolean showOriginal = getBool(props, "showOriginal", true);
        int blockSize = getInt(props, "blockSize", 2);
        int ksize = getInt(props, "ksize", 3);
        int kPercent = getInt(props, "kPercent", 4);
        int thresholdPercent = getInt(props, "thresholdPercent", 1);
        int markerSize = getInt(props, "markerSize", 5);
        int colorR = getInt(props, "colorR", 255);
        int colorG = getInt(props, "colorG", 0);
        int colorB = getInt(props, "colorB", 0);

        CheckBox showOrigCheckBox = dialog.addCheckbox("Show Original Image", showOriginal);
        Slider blockSizeSlider = dialog.addSlider("Block Size:", 2, 10, blockSize, "%.0f");
        String[] ksizes = {"3", "5", "7"};
        int ksizeIdx = (ksize == 5) ? 1 : (ksize == 7) ? 2 : 0;
        ComboBox<String> ksizeCombo = dialog.addComboBox("Kernel Size:", ksizes, ksizes[ksizeIdx]);
        Slider kSlider = dialog.addSlider("K (%):", 1, 10, kPercent, "%.0f%%");
        Slider threshSlider = dialog.addSlider("Threshold (%):", 1, 100, thresholdPercent, "%.0f%%");
        Slider markerSlider = dialog.addSlider("Marker Size:", 1, 15, markerSize, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        props.put("_showOrigCheckBox", showOrigCheckBox);
        props.put("_blockSizeSlider", blockSizeSlider);
        props.put("_ksizeCombo", ksizeCombo);
        props.put("_kSlider", kSlider);
        props.put("_threshSlider", threshSlider);
        props.put("_markerSlider", markerSlider);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
    }

    private static void saveHarrisCornersProperties(Map<String, Object> props) {
        if (props.containsKey("_showOrigCheckBox")) {
            props.put("showOriginal", ((CheckBox) props.get("_showOrigCheckBox")).isSelected());
            props.put("blockSize", (int) ((Slider) props.get("_blockSizeSlider")).getValue());
            @SuppressWarnings("unchecked")
            ComboBox<String> ksizeCombo = (ComboBox<String>) props.get("_ksizeCombo");
            String[] ksizes = {"3", "5", "7"};
            int ksizeIdx = ksizeCombo.getSelectionModel().getSelectedIndex();
            props.put("ksize", Integer.parseInt(ksizes[ksizeIdx]));
            props.put("kPercent", (int) ((Slider) props.get("_kSlider")).getValue());
            props.put("thresholdPercent", (int) ((Slider) props.get("_threshSlider")).getValue());
            props.put("markerSize", (int) ((Slider) props.get("_markerSlider")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            cleanupControls(props, "_showOrigCheckBox", "_blockSizeSlider", "_ksizeCombo",
                           "_kSlider", "_threshSlider", "_markerSlider",
                           "_colorRSpinner", "_colorGSpinner", "_colorBSpinner");
        }
    }

    // ========== HISTOGRAM ==========
    private static void addHistogramProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int modeIndex = getInt(props, "modeIndex", 0);
        int backgroundMode = getInt(props, "backgroundMode", 0);
        boolean fillBars = getBool(props, "fillBars", false);
        int lineThickness = getInt(props, "lineThickness", 4);

        String[] modes = {"Color (BGR)", "Grayscale", "Per Channel"};
        ComboBox<String> modeCombo = dialog.addComboBox("Mode:", modes, modes[modeIndex]);
        String[] bgModes = {"White", "Black", "Background Image"};
        ComboBox<String> bgCombo = dialog.addComboBox("Background:", bgModes, bgModes[backgroundMode]);
        CheckBox fillCheck = dialog.addCheckbox("Fill Bars", fillBars);
        Spinner<Integer> thickSpinner = dialog.addSpinner("Line Thickness:", 1, 10, lineThickness);

        props.put("_modeIndexCombo", modeCombo);
        props.put("_backgroundModeCombo", bgCombo);
        props.put("_fillBarsCheck", fillCheck);
        props.put("_lineThicknessSpinner", thickSpinner);
    }

    private static void saveHistogramProperties(Map<String, Object> props) {
        if (props.containsKey("_modeIndexCombo")) {
            @SuppressWarnings("unchecked")
            ComboBox<String> modeCombo = (ComboBox<String>) props.get("_modeIndexCombo");
            props.put("modeIndex", modeCombo.getSelectionModel().getSelectedIndex());
            @SuppressWarnings("unchecked")
            ComboBox<String> bgCombo = (ComboBox<String>) props.get("_backgroundModeCombo");
            props.put("backgroundMode", bgCombo.getSelectionModel().getSelectedIndex());
            props.put("fillBars", ((CheckBox) props.get("_fillBarsCheck")).isSelected());
            props.put("lineThickness", ((Spinner<Integer>) props.get("_lineThicknessSpinner")).getValue());
            cleanupControls(props, "_modeIndexCombo", "_backgroundModeCombo", "_fillBarsCheck", "_lineThicknessSpinner");
        }
    }

    // ========== HOUGH CIRCLES ==========
    private static void addHoughCirclesProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        boolean showOriginal = getBool(props, "showOriginal", true);
        int minDist = getInt(props, "minDist", 50);
        int param1 = getInt(props, "param1", 100);
        int param2 = getInt(props, "param2", 30);
        int minRadius = getInt(props, "minRadius", 10);
        int maxRadius = getInt(props, "maxRadius", 100);
        int thickness = getInt(props, "thickness", 2);
        int colorR = getInt(props, "colorR", 0);
        int colorG = getInt(props, "colorG", 255);
        int colorB = getInt(props, "colorB", 0);

        CheckBox showOrigCheckBox = dialog.addCheckbox("Show Original Image", showOriginal);
        Slider minDistSlider = dialog.addSlider("Min Distance:", 1, 200, minDist, "%.0f");
        Slider param1Slider = dialog.addSlider("Param1 (Canny):", 1, 300, param1, "%.0f");
        Slider param2Slider = dialog.addSlider("Param2 (Accumulator):", 1, 100, param2, "%.0f");
        Slider minRadiusSlider = dialog.addSlider("Min Radius:", 0, 200, minRadius, "%.0f");
        Slider maxRadiusSlider = dialog.addSlider("Max Radius:", 0, 500, maxRadius, "%.0f");
        Slider thicknessSlider = dialog.addSlider("Thickness:", 1, 10, thickness, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        props.put("_showOrigCheckBox", showOrigCheckBox);
        props.put("_minDistSlider", minDistSlider);
        props.put("_param1Slider", param1Slider);
        props.put("_param2Slider", param2Slider);
        props.put("_minRadiusSlider", minRadiusSlider);
        props.put("_maxRadiusSlider", maxRadiusSlider);
        props.put("_thicknessSlider", thicknessSlider);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
    }

    private static void saveHoughCirclesProperties(Map<String, Object> props) {
        if (props.containsKey("_showOrigCheckBox")) {
            props.put("showOriginal", ((CheckBox) props.get("_showOrigCheckBox")).isSelected());
            props.put("minDist", (int) ((Slider) props.get("_minDistSlider")).getValue());
            props.put("param1", (int) ((Slider) props.get("_param1Slider")).getValue());
            props.put("param2", (int) ((Slider) props.get("_param2Slider")).getValue());
            props.put("minRadius", (int) ((Slider) props.get("_minRadiusSlider")).getValue());
            props.put("maxRadius", (int) ((Slider) props.get("_maxRadiusSlider")).getValue());
            props.put("thickness", (int) ((Slider) props.get("_thicknessSlider")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            cleanupControls(props, "_showOrigCheckBox", "_minDistSlider", "_param1Slider", "_param2Slider",
                           "_minRadiusSlider", "_maxRadiusSlider", "_thicknessSlider",
                           "_colorRSpinner", "_colorGSpinner", "_colorBSpinner");
        }
    }

    // ========== HOUGH LINES ==========
    private static void addHoughLinesProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int threshold = getInt(props, "threshold", 50);
        int minLineLength = getInt(props, "minLineLength", 50);
        int maxLineGap = getInt(props, "maxLineGap", 10);
        int thickness = getInt(props, "thickness", 2);
        int colorR = getInt(props, "colorR", 255);
        int colorG = getInt(props, "colorG", 0);
        int colorB = getInt(props, "colorB", 0);

        Slider thresholdSlider = dialog.addSlider("Threshold:", 1, 200, threshold, "%.0f");
        Slider minLengthSlider = dialog.addSlider("Min Line Length:", 1, 200, minLineLength, "%.0f");
        Slider maxGapSlider = dialog.addSlider("Max Line Gap:", 1, 100, maxLineGap, "%.0f");
        Slider thicknessSlider = dialog.addSlider("Thickness:", 1, 10, thickness, "%.0f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);

        props.put("_thresholdSlider", thresholdSlider);
        props.put("_minLengthSlider", minLengthSlider);
        props.put("_maxGapSlider", maxGapSlider);
        props.put("_thicknessSlider", thicknessSlider);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
    }

    private static void saveHoughLinesProperties(Map<String, Object> props) {
        if (props.containsKey("_thresholdSlider")) {
            props.put("threshold", (int) ((Slider) props.get("_thresholdSlider")).getValue());
            props.put("minLineLength", (int) ((Slider) props.get("_minLengthSlider")).getValue());
            props.put("maxLineGap", (int) ((Slider) props.get("_maxGapSlider")).getValue());
            props.put("thickness", (int) ((Slider) props.get("_thicknessSlider")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            cleanupControls(props, "_thresholdSlider", "_minLengthSlider", "_maxGapSlider",
                           "_thicknessSlider", "_colorRSpinner", "_colorGSpinner", "_colorBSpinner");
        }
    }

    // ========== MATCH TEMPLATE ==========
    private static void addMatchTemplateProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int method = getInt(props, "method", 5);
        int outputMode = getInt(props, "outputMode", 0);
        int rectColorR = getInt(props, "rectColorR", 0);
        int rectColorG = getInt(props, "rectColorG", 255);
        int rectColorB = getInt(props, "rectColorB", 0);
        int rectThickness = getInt(props, "rectThickness", 5);

        String[] methods = {"TM_SQDIFF", "TM_SQDIFF_NORMED", "TM_CCORR", "TM_CCORR_NORMED", "TM_CCOEFF", "TM_CCOEFF_NORMED"};
        ComboBox<String> methodCombo = dialog.addComboBox("Method:", methods, methods[method]);
        String[] outputs = {"Result Matrix Only", "Source + Rectangle", "Source + Result Overlay"};
        ComboBox<String> outputCombo = dialog.addComboBox("Output Mode:", outputs, outputs[outputMode]);
        Spinner<Integer> rSpinner = dialog.addSpinner("Rect Red:", 0, 255, rectColorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Rect Green:", 0, 255, rectColorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Rect Blue:", 0, 255, rectColorB);
        Spinner<Integer> thickSpinner = dialog.addSpinner("Rect Thickness:", 1, 20, rectThickness);

        props.put("_methodCombo", methodCombo);
        props.put("_outputModeCombo", outputCombo);
        props.put("_rectColorRSpinner", rSpinner);
        props.put("_rectColorGSpinner", gSpinner);
        props.put("_rectColorBSpinner", bSpinner);
        props.put("_rectThicknessSpinner", thickSpinner);
    }

    private static void saveMatchTemplateProperties(Map<String, Object> props) {
        if (props.containsKey("_methodCombo")) {
            @SuppressWarnings("unchecked")
            ComboBox<String> methodCombo = (ComboBox<String>) props.get("_methodCombo");
            props.put("method", methodCombo.getSelectionModel().getSelectedIndex());
            @SuppressWarnings("unchecked")
            ComboBox<String> outputCombo = (ComboBox<String>) props.get("_outputModeCombo");
            props.put("outputMode", outputCombo.getSelectionModel().getSelectedIndex());
            props.put("rectColorR", ((Spinner<Integer>) props.get("_rectColorRSpinner")).getValue());
            props.put("rectColorG", ((Spinner<Integer>) props.get("_rectColorGSpinner")).getValue());
            props.put("rectColorB", ((Spinner<Integer>) props.get("_rectColorBSpinner")).getValue());
            props.put("rectThickness", ((Spinner<Integer>) props.get("_rectThicknessSpinner")).getValue());
            cleanupControls(props, "_methodCombo", "_outputModeCombo", "_rectColorRSpinner",
                           "_rectColorGSpinner", "_rectColorBSpinner", "_rectThicknessSpinner");
        }
    }

    // ========== MEAN SHIFT ==========
    private static void addMeanShiftProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int spatialRadius = getInt(props, "spatialRadius", 20);
        int colorRadius = getInt(props, "colorRadius", 40);
        int maxLevel = getInt(props, "maxLevel", 1);

        Slider spatialSlider = dialog.addSlider("Spatial Radius:", 1, 100, spatialRadius, "%.0f");
        Slider colorSlider = dialog.addSlider("Color Radius:", 1, 100, colorRadius, "%.0f");
        Slider levelSlider = dialog.addSlider("Max Pyramid Level:", 0, 4, maxLevel, "%.0f");

        props.put("_spatialRadiusSlider", spatialSlider);
        props.put("_colorRadiusSlider", colorSlider);
        props.put("_maxLevelSlider", levelSlider);
    }

    private static void saveMeanShiftProperties(Map<String, Object> props) {
        if (props.containsKey("_spatialRadiusSlider")) {
            props.put("spatialRadius", (int) ((Slider) props.get("_spatialRadiusSlider")).getValue());
            props.put("colorRadius", (int) ((Slider) props.get("_colorRadiusSlider")).getValue());
            props.put("maxLevel", (int) ((Slider) props.get("_maxLevelSlider")).getValue());
            cleanupControls(props, "_spatialRadiusSlider", "_colorRadiusSlider", "_maxLevelSlider");
        }
    }

    // ========== MORPHOLOGY EX ==========
    private static void addMorphologyExProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int operationIndex = getInt(props, "operationIndex", 0);
        int shapeIndex = getInt(props, "shapeIndex", 0);
        int kernelWidth = getInt(props, "kernelWidth", 3);
        int kernelHeight = getInt(props, "kernelHeight", 3);
        int iterations = getInt(props, "iterations", 1);

        String[] operations = {"Erode", "Dilate", "Open", "Close", "Gradient", "Top Hat", "Black Hat"};
        String[] shapes = {"Rectangle", "Cross", "Ellipse"};

        ComboBox<String> opCombo = dialog.addComboBox("Operation:", operations, operations[operationIndex]);
        ComboBox<String> shapeCombo = dialog.addComboBox("Shape:", shapes, shapes[shapeIndex]);
        Slider widthSlider = dialog.addSlider("Kernel Width:", 1, 31, kernelWidth, "%.0f");
        Slider heightSlider = dialog.addSlider("Kernel Height:", 1, 31, kernelHeight, "%.0f");
        Slider iterSlider = dialog.addSlider("Iterations:", 1, 20, iterations, "%.0f");

        props.put("_operationCombo", opCombo);
        props.put("_shapeCombo", shapeCombo);
        props.put("_kernelWidthSlider", widthSlider);
        props.put("_kernelHeightSlider", heightSlider);
        props.put("_iterationsSlider", iterSlider);
    }

    private static void saveMorphologyExProperties(Map<String, Object> props) {
        if (props.containsKey("_operationCombo")) {
            @SuppressWarnings("unchecked")
            ComboBox<String> opCombo = (ComboBox<String>) props.get("_operationCombo");
            props.put("operationIndex", opCombo.getSelectionModel().getSelectedIndex());
            @SuppressWarnings("unchecked")
            ComboBox<String> shapeCombo = (ComboBox<String>) props.get("_shapeCombo");
            props.put("shapeIndex", shapeCombo.getSelectionModel().getSelectedIndex());
            props.put("kernelWidth", (int) ((Slider) props.get("_kernelWidthSlider")).getValue());
            props.put("kernelHeight", (int) ((Slider) props.get("_kernelHeightSlider")).getValue());
            props.put("iterations", (int) ((Slider) props.get("_iterationsSlider")).getValue());
            cleanupControls(props, "_operationCombo", "_shapeCombo", "_kernelWidthSlider",
                           "_kernelHeightSlider", "_iterationsSlider");
        }
    }

    // ========== ORB FEATURES ==========
    private static void addORBFeaturesProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int nFeatures = getInt(props, "nFeatures", 500);
        int fastThreshold = getInt(props, "fastThreshold", 20);
        int nLevels = getInt(props, "nLevels", 8);
        boolean showRich = getBool(props, "showRich", true);

        Spinner<Integer> featSpinner = dialog.addSpinner("Max Features:", 10, 5000, nFeatures);
        Spinner<Integer> threshSpinner = dialog.addSpinner("FAST Threshold:", 1, 100, fastThreshold);
        Spinner<Integer> levelSpinner = dialog.addSpinner("Pyramid Levels:", 1, 16, nLevels);
        CheckBox richCheck = dialog.addCheckbox("Show Rich Keypoints", showRich);

        props.put("_nFeaturesSpinner", featSpinner);
        props.put("_fastThresholdSpinner", threshSpinner);
        props.put("_nLevelsSpinner", levelSpinner);
        props.put("_showRichCheck", richCheck);
    }

    private static void saveORBFeaturesProperties(Map<String, Object> props) {
        if (props.containsKey("_nFeaturesSpinner")) {
            props.put("nFeatures", ((Spinner<Integer>) props.get("_nFeaturesSpinner")).getValue());
            props.put("fastThreshold", ((Spinner<Integer>) props.get("_fastThresholdSpinner")).getValue());
            props.put("nLevels", ((Spinner<Integer>) props.get("_nLevelsSpinner")).getValue());
            props.put("showRich", ((CheckBox) props.get("_showRichCheck")).isSelected());
            cleanupControls(props, "_nFeaturesSpinner", "_fastThresholdSpinner", "_nLevelsSpinner", "_showRichCheck");
        }
    }

    // ========== SIFT FEATURES ==========
    private static void addSIFTFeaturesProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int nFeatures = getInt(props, "nFeatures", 500);
        int nOctaveLayers = getInt(props, "nOctaveLayers", 3);
        int contrastThreshold = getInt(props, "contrastThreshold", 4);  // 0.04 * 100
        int edgeThreshold = getInt(props, "edgeThreshold", 10);
        int sigma = getInt(props, "sigma", 16);  // 1.6 * 10
        boolean showRich = getBool(props, "showRichKeypoints", true);
        int colorIndex = getInt(props, "colorIndex", 0);

        Spinner<Integer> featSpinner = dialog.addSpinner("Max Features (0=all):", 0, 5000, nFeatures);
        Spinner<Integer> octaveSpinner = dialog.addSpinner("Octave Layers:", 1, 10, nOctaveLayers);
        Slider contrastSlider = dialog.addSlider("Contrast Thresh (x100):", 1, 20, contrastThreshold, "%.0f");
        Spinner<Integer> edgeSpinner = dialog.addSpinner("Edge Threshold:", 1, 50, edgeThreshold);
        Slider sigmaSlider = dialog.addSlider("Sigma (x10):", 10, 30, sigma, "%.0f");
        CheckBox richCheck = dialog.addCheckbox("Show Rich Keypoints", showRich);
        String[] colors = {"Green", "Red", "Blue", "Yellow", "White"};
        ComboBox<String> colorCombo = dialog.addComboBox("Keypoint Color:", colors, colors[colorIndex]);

        props.put("_nFeaturesSpinner", featSpinner);
        props.put("_nOctaveLayersSpinner", octaveSpinner);
        props.put("_contrastThresholdSlider", contrastSlider);
        props.put("_edgeThresholdSpinner", edgeSpinner);
        props.put("_sigmaSlider", sigmaSlider);
        props.put("_showRichKeypointsCheck", richCheck);
        props.put("_colorIndexCombo", colorCombo);
    }

    private static void saveSIFTFeaturesProperties(Map<String, Object> props) {
        if (props.containsKey("_nFeaturesSpinner")) {
            props.put("nFeatures", ((Spinner<Integer>) props.get("_nFeaturesSpinner")).getValue());
            props.put("nOctaveLayers", ((Spinner<Integer>) props.get("_nOctaveLayersSpinner")).getValue());
            props.put("contrastThreshold", (int) ((Slider) props.get("_contrastThresholdSlider")).getValue());
            props.put("edgeThreshold", ((Spinner<Integer>) props.get("_edgeThresholdSpinner")).getValue());
            props.put("sigma", (int) ((Slider) props.get("_sigmaSlider")).getValue());
            props.put("showRichKeypoints", ((CheckBox) props.get("_showRichKeypointsCheck")).isSelected());
            @SuppressWarnings("unchecked")
            ComboBox<String> colorCombo = (ComboBox<String>) props.get("_colorIndexCombo");
            props.put("colorIndex", colorCombo.getSelectionModel().getSelectedIndex());
            cleanupControls(props, "_nFeaturesSpinner", "_nOctaveLayersSpinner", "_contrastThresholdSlider",
                           "_edgeThresholdSpinner", "_sigmaSlider", "_showRichKeypointsCheck", "_colorIndexCombo");
        }
    }

    // ========== SHI-TOMASI ==========
    private static void addShiTomasiProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int maxCorners = getInt(props, "maxCorners", 100);
        int qualityLevel = getInt(props, "qualityLevel", 1);  // 0.01 * 100
        int minDistance = getInt(props, "minDistance", 10);
        int blockSize = getInt(props, "blockSize", 3);
        boolean useHarris = getBool(props, "useHarrisDetector", false);
        int kPercent = getInt(props, "kPercent", 4);  // 0.04 * 100
        int markerSize = getInt(props, "markerSize", 5);
        boolean drawFeatures = getBool(props, "drawFeatures", true);

        Spinner<Integer> cornersSpinner = dialog.addSpinner("Max Corners:", 1, 1000, maxCorners);
        Slider qualitySlider = dialog.addSlider("Quality Level (x100):", 1, 100, qualityLevel, "%.0f");
        Spinner<Integer> minDistSpinner = dialog.addSpinner("Min Distance:", 1, 100, minDistance);
        Spinner<Integer> blockSpinner = dialog.addSpinner("Block Size:", 2, 31, blockSize);
        CheckBox harrisCheck = dialog.addCheckbox("Use Harris Detector", useHarris);
        Slider kSlider = dialog.addSlider("K Parameter (%):", 1, 20, kPercent, "%.0f");
        Spinner<Integer> markerSpinner = dialog.addSpinner("Marker Size:", 1, 20, markerSize);
        CheckBox drawCheck = dialog.addCheckbox("Draw Features", drawFeatures);

        props.put("_maxCornersSpinner", cornersSpinner);
        props.put("_qualityLevelSlider", qualitySlider);
        props.put("_minDistanceSpinner", minDistSpinner);
        props.put("_blockSizeSpinner", blockSpinner);
        props.put("_useHarrisDetectorCheck", harrisCheck);
        props.put("_kPercentSlider", kSlider);
        props.put("_markerSizeSpinner", markerSpinner);
        props.put("_drawFeaturesCheck", drawCheck);
    }

    private static void saveShiTomasiProperties(Map<String, Object> props) {
        if (props.containsKey("_maxCornersSpinner")) {
            props.put("maxCorners", ((Spinner<Integer>) props.get("_maxCornersSpinner")).getValue());
            props.put("qualityLevel", (int) ((Slider) props.get("_qualityLevelSlider")).getValue());
            props.put("minDistance", ((Spinner<Integer>) props.get("_minDistanceSpinner")).getValue());
            props.put("blockSize", ((Spinner<Integer>) props.get("_blockSizeSpinner")).getValue());
            props.put("useHarrisDetector", ((CheckBox) props.get("_useHarrisDetectorCheck")).isSelected());
            props.put("kPercent", (int) ((Slider) props.get("_kPercentSlider")).getValue());
            props.put("markerSize", ((Spinner<Integer>) props.get("_markerSizeSpinner")).getValue());
            props.put("drawFeatures", ((CheckBox) props.get("_drawFeaturesCheck")).isSelected());
            cleanupControls(props, "_maxCornersSpinner", "_qualityLevelSlider", "_minDistanceSpinner",
                           "_blockSizeSpinner", "_useHarrisDetectorCheck", "_kPercentSlider",
                           "_markerSizeSpinner", "_drawFeaturesCheck");
        }
    }

    // ========== TEXT ==========
    private static void addTextProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        String text = getString(props, "text", "Hello");
        int posX = getInt(props, "posX", 50);
        int posY = getInt(props, "posY", 100);
        int fontIndex = getInt(props, "fontIndex", 0);
        double fontScale = getDouble(props, "fontScale", 1.0);
        int colorR = getInt(props, "colorR", 0);
        int colorG = getInt(props, "colorG", 255);
        int colorB = getInt(props, "colorB", 0);
        int thickness = getInt(props, "thickness", 2);

        TextField textField = dialog.addTextField("Text:", text);
        Spinner<Integer> xSpinner = dialog.addSpinner("Position X:", -4096, 4096, posX);
        Spinner<Integer> ySpinner = dialog.addSpinner("Position Y:", -4096, 4096, posY);
        String[] fonts = {"Simplex", "Plain", "Duplex", "Complex", "Triplex", "Complex Small", "Script Simplex", "Script Complex"};
        ComboBox<String> fontCombo = dialog.addComboBox("Font:", fonts, fonts[Math.min(fontIndex, fonts.length - 1)]);
        Slider scaleSlider = dialog.addSlider("Font Scale:", 0.1, 10.0, fontScale, "%.1f");
        Spinner<Integer> rSpinner = dialog.addSpinner("Color R:", 0, 255, colorR);
        Spinner<Integer> gSpinner = dialog.addSpinner("Color G:", 0, 255, colorG);
        Spinner<Integer> bSpinner = dialog.addSpinner("Color B:", 0, 255, colorB);
        Spinner<Integer> thicknessSpinner = dialog.addSpinner("Thickness:", 1, 20, thickness);

        props.put("_textField", textField);
        props.put("_posXSpinner", xSpinner);
        props.put("_posYSpinner", ySpinner);
        props.put("_fontCombo", fontCombo);
        props.put("_fontScaleSlider", scaleSlider);
        props.put("_colorRSpinner", rSpinner);
        props.put("_colorGSpinner", gSpinner);
        props.put("_colorBSpinner", bSpinner);
        props.put("_thicknessSpinner", thicknessSpinner);
    }

    private static void saveTextProperties(Map<String, Object> props) {
        if (props.containsKey("_textField")) {
            props.put("text", ((TextField) props.get("_textField")).getText());
            props.put("posX", ((Spinner<Integer>) props.get("_posXSpinner")).getValue());
            props.put("posY", ((Spinner<Integer>) props.get("_posYSpinner")).getValue());
            @SuppressWarnings("unchecked")
            ComboBox<String> fontCombo = (ComboBox<String>) props.get("_fontCombo");
            props.put("fontIndex", fontCombo.getSelectionModel().getSelectedIndex());
            props.put("fontScale", ((Slider) props.get("_fontScaleSlider")).getValue());
            props.put("colorR", ((Spinner<Integer>) props.get("_colorRSpinner")).getValue());
            props.put("colorG", ((Spinner<Integer>) props.get("_colorGSpinner")).getValue());
            props.put("colorB", ((Spinner<Integer>) props.get("_colorBSpinner")).getValue());
            props.put("thickness", ((Spinner<Integer>) props.get("_thicknessSpinner")).getValue());
            cleanupControls(props, "_textField", "_posXSpinner", "_posYSpinner", "_fontCombo",
                           "_fontScaleSlider", "_colorRSpinner", "_colorGSpinner", "_colorBSpinner",
                           "_thicknessSpinner");
        }
    }

    // ========== THRESHOLD ==========
    private static void addThresholdProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        // Handle both old "threshold" and new "threshValue" property names
        int threshValue = getInt(props, "threshValue", getInt(props, "threshold", 127));
        int maxValue = getInt(props, "maxValue", 255);
        int typeIndex = getInt(props, "typeIndex", 0);
        int modifierIndex = getInt(props, "modifierIndex", 0);

        Slider threshSlider = dialog.addSlider("Threshold:", 0, 255, threshValue, "%.0f");
        Slider maxSlider = dialog.addSlider("Max Value:", 0, 255, maxValue, "%.0f");
        String[] types = {"BINARY", "BINARY_INV", "TRUNC", "TOZERO", "TOZERO_INV"};
        ComboBox<String> typeCombo = dialog.addComboBox("Type:", types, types[typeIndex]);
        String[] modifiers = {"None", "OTSU", "TRIANGLE"};
        ComboBox<String> modCombo = dialog.addComboBox("Modifier:", modifiers, modifiers[modifierIndex]);

        props.put("_threshValueSlider", threshSlider);
        props.put("_maxValueSlider", maxSlider);
        props.put("_typeCombo", typeCombo);
        props.put("_modifierCombo", modCombo);
    }

    private static void saveThresholdProperties(Map<String, Object> props) {
        if (props.containsKey("_threshValueSlider")) {
            // Use correct property name "threshValue" (not "threshold")
            props.put("threshValue", (int) ((Slider) props.get("_threshValueSlider")).getValue());
            props.put("maxValue", (int) ((Slider) props.get("_maxValueSlider")).getValue());
            @SuppressWarnings("unchecked")
            ComboBox<String> typeCombo = (ComboBox<String>) props.get("_typeCombo");
            props.put("typeIndex", typeCombo.getSelectionModel().getSelectedIndex());
            @SuppressWarnings("unchecked")
            ComboBox<String> modCombo = (ComboBox<String>) props.get("_modifierCombo");
            props.put("modifierIndex", modCombo.getSelectionModel().getSelectedIndex());
            // Remove old property name if exists
            props.remove("threshold");
            cleanupControls(props, "_threshValueSlider", "_maxValueSlider", "_typeCombo", "_modifierCombo");
        }
    }

    // ========== WARP AFFINE ==========
    private static void addWarpAffineProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        double scaleX = getDouble(props, "scaleX", 1.0);
        double scaleY = getDouble(props, "scaleY", 1.0);
        double rotation = getDouble(props, "rotation", 0.0);
        double translateX = getDouble(props, "translateX", 0.0);
        double translateY = getDouble(props, "translateY", 0.0);

        Slider scaleXSlider = dialog.addSlider("Scale X:", 0.1, 5.0, scaleX, "%.2f");
        Slider scaleYSlider = dialog.addSlider("Scale Y:", 0.1, 5.0, scaleY, "%.2f");
        Slider rotSlider = dialog.addSlider("Rotation:", -180, 180, rotation, "%.1f°");
        Slider txSlider = dialog.addSlider("Translate X:", -500, 500, translateX, "%.0f");
        Slider tySlider = dialog.addSlider("Translate Y:", -500, 500, translateY, "%.0f");

        props.put("_scaleXSlider", scaleXSlider);
        props.put("_scaleYSlider", scaleYSlider);
        props.put("_rotationSlider", rotSlider);
        props.put("_translateXSlider", txSlider);
        props.put("_translateYSlider", tySlider);
    }

    private static void saveWarpAffineProperties(Map<String, Object> props) {
        if (props.containsKey("_scaleXSlider")) {
            props.put("scaleX", ((Slider) props.get("_scaleXSlider")).getValue());
            props.put("scaleY", ((Slider) props.get("_scaleYSlider")).getValue());
            props.put("rotation", ((Slider) props.get("_rotationSlider")).getValue());
            props.put("translateX", ((Slider) props.get("_translateXSlider")).getValue());
            props.put("translateY", ((Slider) props.get("_translateYSlider")).getValue());
            cleanupControls(props, "_scaleXSlider", "_scaleYSlider", "_rotationSlider",
                           "_translateXSlider", "_translateYSlider");
        }
    }

    // ========== WEBCAM SOURCE ==========
    private static void addWebcamSourceProperties(FXPropertiesDialog dialog, Map<String, Object> props) {
        int cameraIndex = getInt(props, "cameraIndex", -1);
        int resolutionIndex = getInt(props, "resolutionIndex", 1);
        boolean mirrorHorizontal = getBool(props, "mirrorHorizontal", true);
        int fpsIndex = getInt(props, "fpsIndex", 0);

        Spinner<Integer> camSpinner = dialog.addSpinner("Camera Index (-1=auto):", -1, 10, cameraIndex);
        String[] resolutions = {"320x240", "640x480", "1280x720", "1920x1080"};
        ComboBox<String> resCombo = dialog.addComboBox("Resolution:", resolutions, resolutions[resolutionIndex]);
        CheckBox mirrorCheck = dialog.addCheckbox("Mirror Horizontal", mirrorHorizontal);
        String[] fpsOptions = {"1 fps", "5 fps", "10 fps", "15 fps", "30 fps"};
        ComboBox<String> fpsCombo = dialog.addComboBox("FPS:", fpsOptions, fpsOptions[fpsIndex]);

        props.put("_cameraIndexSpinner", camSpinner);
        props.put("_resolutionIndexCombo", resCombo);
        props.put("_mirrorHorizontalCheck", mirrorCheck);
        props.put("_fpsIndexCombo", fpsCombo);
    }

    private static void saveWebcamSourceProperties(Map<String, Object> props) {
        if (props.containsKey("_cameraIndexSpinner")) {
            props.put("cameraIndex", ((Spinner<Integer>) props.get("_cameraIndexSpinner")).getValue());
            @SuppressWarnings("unchecked")
            ComboBox<String> resCombo = (ComboBox<String>) props.get("_resolutionIndexCombo");
            props.put("resolutionIndex", resCombo.getSelectionModel().getSelectedIndex());
            props.put("mirrorHorizontal", ((CheckBox) props.get("_mirrorHorizontalCheck")).isSelected());
            @SuppressWarnings("unchecked")
            ComboBox<String> fpsCombo = (ComboBox<String>) props.get("_fpsIndexCombo");
            props.put("fpsIndex", fpsCombo.getSelectionModel().getSelectedIndex());
            cleanupControls(props, "_cameraIndexSpinner", "_resolutionIndexCombo", "_mirrorHorizontalCheck", "_fpsIndexCombo");
        }
    }

    // ========== HELPER METHODS ==========

    private static int getInt(Map<String, Object> props, String key, int defaultVal) {
        Object val = props.get(key);
        if (val == null) return defaultVal;
        if (val instanceof Number) return ((Number) val).intValue();
        try {
            return Integer.parseInt(val.toString());
        } catch (NumberFormatException e) {
            return defaultVal;
        }
    }

    private static double getDouble(Map<String, Object> props, String key, double defaultVal) {
        Object val = props.get(key);
        if (val == null) return defaultVal;
        if (val instanceof Number) return ((Number) val).doubleValue();
        try {
            return Double.parseDouble(val.toString());
        } catch (NumberFormatException e) {
            return defaultVal;
        }
    }

    private static boolean getBool(Map<String, Object> props, String key, boolean defaultVal) {
        Object val = props.get(key);
        if (val == null) return defaultVal;
        if (val instanceof Boolean) return (Boolean) val;
        return Boolean.parseBoolean(val.toString());
    }

    private static String getString(Map<String, Object> props, String key, String defaultVal) {
        Object val = props.get(key);
        if (val == null) return defaultVal;
        return val.toString();
    }

    private static void cleanupControls(Map<String, Object> props, String... keys) {
        for (String key : keys) {
            props.remove(key);
        }
    }
}
