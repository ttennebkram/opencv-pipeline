package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Erode (Morphological) node.
 */
@NodeInfo(name = "Erode", category = "Morphology", aliases = {})
public class ErodeNode extends ProcessingNode {
    private static final String[] SHAPE_NAMES = {"Rectangle", "Ellipse", "Cross"};
    private static final int[] SHAPE_VALUES = {Imgproc.MORPH_RECT, Imgproc.MORPH_ELLIPSE, Imgproc.MORPH_CROSS};

    private int kernelSize = 5;
    private int shapeIndex = 0;
    private int iterations = 1;

    public ErodeNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Erode", x, y);
    }

    // Getters/setters for serialization
    public int getKernelSize() { return kernelSize; }
    public void setKernelSize(int v) { kernelSize = v; }
    public int getShapeIndex() { return shapeIndex; }
    public void setShapeIndex(int v) { shapeIndex = v; }
    public int getIterations() { return iterations; }
    public void setIterations(int v) { iterations = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Ensure odd kernel size
        int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;

        // Create structuring element
        Mat kernel = Imgproc.getStructuringElement(SHAPE_VALUES[shapeIndex], new Size(ksize, ksize));

        // Apply erosion
        Mat output = new Mat();
        Imgproc.erode(input, output, kernel, new org.opencv.core.Point(-1, -1), iterations);

        kernel.release();
        return output;
    }

    @Override
    public String getDescription() {
        return "Morphological Erosion\ncv2.erode(src, kernel, iterations)";
    }

    @Override
    public String getDisplayName() {
        return "Erode";
    }

    @Override
    public String getCategory() {
        return "Morphological";
    }

    @Override
    protected int getPropertiesDialogColumns() {
        return 3;
    }

    @Override
    protected Runnable addPropertiesContent(Shell dialog, int columns) {
        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = columns;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = columns;
        sep.setLayoutData(sepGd);

        // Kernel Size
        new Label(dialog, SWT.NONE).setText("Kernel Size:");
        Scale kScale = new Scale(dialog, SWT.HORIZONTAL);
        kScale.setMinimum(1);
        kScale.setMaximum(31);
        // Clamp slider position to valid range, but keep actual value
        int kSliderPos = Math.min(Math.max(kernelSize, 1), 31);
        kScale.setSelection(kSliderPos);
        kScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label kLabel = new Label(dialog, SWT.NONE);
        kLabel.setText(String.valueOf(kernelSize)); // Show real value
        kScale.addListener(SWT.Selection, e -> kLabel.setText(String.valueOf(kScale.getSelection())));

        // Kernel Shape
        new Label(dialog, SWT.NONE).setText("Kernel Shape:");
        Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        shapeCombo.setItems(SHAPE_NAMES);
        shapeCombo.select(shapeIndex);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        shapeCombo.setLayoutData(comboGd);

        // Iterations
        new Label(dialog, SWT.NONE).setText("Iterations:");
        Scale iterScale = new Scale(dialog, SWT.HORIZONTAL);
        iterScale.setMinimum(1);
        iterScale.setMaximum(10);
        // Clamp slider position to valid range, but keep actual value
        int iterSliderPos = Math.min(Math.max(iterations, 1), 10);
        iterScale.setSelection(iterSliderPos);
        iterScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label iterLabel = new Label(dialog, SWT.NONE);
        iterLabel.setText(String.valueOf(iterations)); // Show real value
        iterScale.addListener(SWT.Selection, e -> iterLabel.setText(String.valueOf(iterScale.getSelection())));

        return () -> {
            kernelSize = kScale.getSelection();
            shapeIndex = shapeCombo.getSelectionIndex();
            iterations = iterScale.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("kernelSize", kernelSize);
        json.addProperty("shapeIndex", shapeIndex);
        json.addProperty("iterations", iterations);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("kernelSize")) kernelSize = json.get("kernelSize").getAsInt();
        if (json.has("shapeIndex")) shapeIndex = json.get("shapeIndex").getAsInt();
        if (json.has("iterations")) iterations = json.get("iterations").getAsInt();
    }
}
