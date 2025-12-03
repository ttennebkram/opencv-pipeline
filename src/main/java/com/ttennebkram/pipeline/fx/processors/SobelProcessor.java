package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import javafx.scene.control.ComboBox;
import javafx.scene.control.ToggleGroup;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Sobel edge detection processor.
 * Computes first-order derivatives for edge detection.
 */
@FXProcessorInfo(
    nodeType = "Sobel",
    displayName = "Sobel Edges",
    buttonName = "Sobel",
    category = "Edges",
    description = "Sobel derivatives\nImgproc.Sobel(src, dst, ddepth, dx, dy, ksize)"
)
public class SobelProcessor extends FXProcessorBase {

    // Properties with defaults
    private int dx = 1;
    private int dy = 0;
    private int kernelSizeIndex = 1;  // Index into {1, 3, 5, 7}

    private static final int[] KERNEL_SIZES = {1, 3, 5, 7};

    @Override
    public String getNodeType() {
        return "Sobel";
    }

    @Override
    public String getCategory() {
        return "Edge Detection";
    }

    @Override
    public String getDescription() {
        return "Sobel Derivatives\nImgproc.Sobel(src, dst, ddepth, dx, dy, ksize)";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        // Convert to grayscale if needed
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        int ksize = KERNEL_SIZES[Math.min(kernelSizeIndex, KERNEL_SIZES.length - 1)];

        // Ensure dx + dy >= 1
        int useDx = dx;
        int useDy = dy;
        if (useDx + useDy < 1) {
            useDx = 1;
        }

        Mat gradX = new Mat();
        Mat gradY = new Mat();
        Mat output = new Mat();

        if (useDx > 0 && useDy > 0) {
            // Both directions - compute separately and combine
            Imgproc.Sobel(gray, gradX, CvType.CV_16S, useDx, 0, ksize);
            Imgproc.Sobel(gray, gradY, CvType.CV_16S, 0, useDy, ksize);
            Core.convertScaleAbs(gradX, gradX);
            Core.convertScaleAbs(gradY, gradY);
            Core.addWeighted(gradX, 0.5, gradY, 0.5, 0, output);
        } else if (useDx > 0) {
            Imgproc.Sobel(gray, output, CvType.CV_16S, useDx, 0, ksize);
            Core.convertScaleAbs(output, output);
        } else {
            Imgproc.Sobel(gray, output, CvType.CV_16S, 0, useDy, ksize);
            Core.convertScaleAbs(output, output);
        }

        gray.release();
        gradX.release();
        gradY.release();

        // Convert back to BGR for display
        Mat bgr = new Mat();
        Imgproc.cvtColor(output, bgr, Imgproc.COLOR_GRAY2BGR);
        output.release();

        return bgr;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        String[] derivOrders = {"0", "1", "2"};
        ToggleGroup dxGroup = dialog.addRadioButtons("dx (X derivative):", derivOrders, dx);
        ToggleGroup dyGroup = dialog.addRadioButtons("dy (Y derivative):", derivOrders, dy);

        String[] kernelSizes = {"1", "3", "5", "7"};
        ComboBox<String> kernelSizeCombo = dialog.addComboBox("Kernel Size:", kernelSizes,
                kernelSizes[Math.min(kernelSizeIndex, kernelSizes.length - 1)]);

        dialog.addDescription("Note: dx + dy must be >= 1");

        // Save callback
        dialog.setOnOk(() -> {
            dx = dxGroup.getSelectedToggle() != null ? (Integer) dxGroup.getSelectedToggle().getUserData() : 1;
            dy = dyGroup.getSelectedToggle() != null ? (Integer) dyGroup.getSelectedToggle().getUserData() : 0;
            kernelSizeIndex = kernelSizeCombo.getSelectionModel().getSelectedIndex();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("dx", dx);
        json.addProperty("dy", dy);
        json.addProperty("kernelSizeIndex", kernelSizeIndex);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        dx = getJsonInt(json, "dx", 1);
        dy = getJsonInt(json, "dy", 0);
        kernelSizeIndex = getJsonInt(json, "kernelSizeIndex", 1);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        dx = getInt(node.properties, "dx", 1);
        dy = getInt(node.properties, "dy", 0);
        kernelSizeIndex = getInt(node.properties, "kernelSizeIndex", 1);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("dx", dx);
        node.properties.put("dy", dy);
        node.properties.put("kernelSizeIndex", kernelSizeIndex);
    }
}
