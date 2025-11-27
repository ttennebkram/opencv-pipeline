package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * MorphologyEx node - performs advanced morphological operations.
 * Combines getStructuringElement and morphologyEx into one node.
 */
@NodeInfo(name = "MorphologyEx", category = "Morphology", aliases = {"Morphology Ex"})
public class MorphologyExNode extends ProcessingNode {
    private int operationIndex = 0; // 0=Erode, 1=Dilate, 2=Open, 3=Close, 4=Gradient, 5=TopHat, 6=BlackHat
    private int shapeIndex = 0; // 0=Rectangle, 1=Cross, 2=Ellipse
    private int kernelWidth = 3;
    private int kernelHeight = 3;
    private int iterations = 1;
    private int anchorX = -1; // -1 means center
    private int anchorY = -1; // -1 means center

    private static final String[] OPERATIONS = {
        "Erode", "Dilate", "Open", "Close", "Gradient", "Top Hat / White Hat", "Black Hat"
    };

    private static final int[] OPERATION_CODES = {
        Imgproc.MORPH_ERODE,
        Imgproc.MORPH_DILATE,
        Imgproc.MORPH_OPEN,
        Imgproc.MORPH_CLOSE,
        Imgproc.MORPH_GRADIENT,
        Imgproc.MORPH_TOPHAT,
        Imgproc.MORPH_BLACKHAT
    };

    private static final String[] SHAPES = {
        "Rectangle", "Cross", "Ellipse"
    };

    private static final int[] SHAPE_CODES = {
        Imgproc.MORPH_RECT,
        Imgproc.MORPH_CROSS,
        Imgproc.MORPH_ELLIPSE
    };

    public MorphologyExNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Gradient/MorphEx", x, y);
    }

    // Getters/setters for serialization
    public int getOperationIndex() { return operationIndex; }
    public void setOperationIndex(int v) { operationIndex = v; }
    public int getShapeIndex() { return shapeIndex; }
    public void setShapeIndex(int v) { shapeIndex = v; }
    public int getKernelWidth() { return kernelWidth; }
    public void setKernelWidth(int v) { kernelWidth = v; }
    public int getKernelHeight() { return kernelHeight; }
    public void setKernelHeight(int v) { kernelHeight = v; }
    public int getIterations() { return iterations; }
    public void setIterations(int v) { iterations = v; }
    public int getAnchorX() { return anchorX; }
    public void setAnchorX(int v) { anchorX = v; }
    public int getAnchorY() { return anchorY; }
    public void setAnchorY(int v) { anchorY = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Ensure odd kernel sizes
        int kWidth = (kernelWidth % 2 == 0) ? kernelWidth + 1 : kernelWidth;
        int kHeight = (kernelHeight % 2 == 0) ? kernelHeight + 1 : kernelHeight;

        // Create structuring element
        Mat kernel = Imgproc.getStructuringElement(
            SHAPE_CODES[shapeIndex],
            new Size(kWidth, kHeight)
        );

        Mat output = new Mat();
        Imgproc.morphologyEx(
            input,
            output,
            OPERATION_CODES[operationIndex],
            kernel,
            new org.opencv.core.Point(anchorX, anchorY),
            iterations
        );

        kernel.release();
        return output;
    }

    @Override
    public String getDescription() {
        return "MorphologyEx\ncv2.morphologyEx(src, op, kernel)";
    }

    @Override
    public String getDisplayName() {
        return "Gradient/MorphEx";
    }

    @Override
    public String getCategory() {
        return "Morphological";
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

        // Operation selector
        new Label(dialog, SWT.NONE).setText("Operation:");
        Combo opCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        opCombo.setItems(OPERATIONS);
        opCombo.select(operationIndex);
        opCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Shape selector
        new Label(dialog, SWT.NONE).setText("Kernel Shape:");
        Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        shapeCombo.setItems(SHAPES);
        shapeCombo.select(shapeIndex);
        shapeCombo.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Kernel width
        new Label(dialog, SWT.NONE).setText("Kernel Width:");
        Composite widthComp = new Composite(dialog, SWT.NONE);
        widthComp.setLayout(new GridLayout(2, false));
        widthComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Scale widthScale = new Scale(widthComp, SWT.HORIZONTAL);
        widthScale.setMinimum(1);
        widthScale.setMaximum(31);
        int widthSliderPos = Math.min(Math.max(kernelWidth, 1), 31);
        widthScale.setSelection(widthSliderPos);
        widthScale.setLayoutData(new GridData(120, SWT.DEFAULT));

        Label widthLabel = new Label(widthComp, SWT.NONE);
        widthLabel.setText(String.valueOf(kernelWidth));
        widthScale.addListener(SWT.Selection, e -> widthLabel.setText(String.valueOf(widthScale.getSelection())));

        // Kernel height
        new Label(dialog, SWT.NONE).setText("Kernel Height:");
        Composite heightComp = new Composite(dialog, SWT.NONE);
        heightComp.setLayout(new GridLayout(2, false));
        heightComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Scale heightScale = new Scale(heightComp, SWT.HORIZONTAL);
        heightScale.setMinimum(1);
        heightScale.setMaximum(31);
        int heightSliderPos = Math.min(Math.max(kernelHeight, 1), 31);
        heightScale.setSelection(heightSliderPos);
        heightScale.setLayoutData(new GridData(120, SWT.DEFAULT));

        Label heightLabel = new Label(heightComp, SWT.NONE);
        heightLabel.setText(String.valueOf(kernelHeight));
        heightScale.addListener(SWT.Selection, e -> heightLabel.setText(String.valueOf(heightScale.getSelection())));

        // Iterations
        new Label(dialog, SWT.NONE).setText("Iterations:");
        Composite iterComp = new Composite(dialog, SWT.NONE);
        iterComp.setLayout(new GridLayout(2, false));
        iterComp.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        Scale iterScale = new Scale(iterComp, SWT.HORIZONTAL);
        iterScale.setMinimum(1);
        iterScale.setMaximum(20);
        int iterSliderPos = Math.min(Math.max(iterations, 1), 20);
        iterScale.setSelection(iterSliderPos);
        iterScale.setLayoutData(new GridData(120, SWT.DEFAULT));

        Label iterLabel = new Label(iterComp, SWT.NONE);
        iterLabel.setText(String.valueOf(iterations));
        iterScale.addListener(SWT.Selection, e -> iterLabel.setText(String.valueOf(iterScale.getSelection())));

        // Anchor X
        new Label(dialog, SWT.NONE).setText("Anchor X (-1=center):");
        Spinner anchorXSpinner = new Spinner(dialog, SWT.BORDER);
        anchorXSpinner.setMinimum(-1);
        anchorXSpinner.setMaximum(30);
        anchorXSpinner.setSelection(anchorX);
        anchorXSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        // Anchor Y
        new Label(dialog, SWT.NONE).setText("Anchor Y (-1=center):");
        Spinner anchorYSpinner = new Spinner(dialog, SWT.BORDER);
        anchorYSpinner.setMinimum(-1);
        anchorYSpinner.setMaximum(30);
        anchorYSpinner.setSelection(anchorY);
        anchorYSpinner.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));

        return () -> {
            operationIndex = opCombo.getSelectionIndex();
            shapeIndex = shapeCombo.getSelectionIndex();
            kernelWidth = widthScale.getSelection();
            kernelHeight = heightScale.getSelection();
            iterations = iterScale.getSelection();
            anchorX = anchorXSpinner.getSelection();
            anchorY = anchorYSpinner.getSelection();
        };
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("operationIndex", operationIndex);
        json.addProperty("shapeIndex", shapeIndex);
        json.addProperty("kernelWidth", kernelWidth);
        json.addProperty("kernelHeight", kernelHeight);
        json.addProperty("iterations", iterations);
        json.addProperty("anchorX", anchorX);
        json.addProperty("anchorY", anchorY);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("operationIndex")) operationIndex = json.get("operationIndex").getAsInt();
        if (json.has("shapeIndex")) shapeIndex = json.get("shapeIndex").getAsInt();
        if (json.has("kernelWidth")) kernelWidth = json.get("kernelWidth").getAsInt();
        if (json.has("kernelHeight")) kernelHeight = json.get("kernelHeight").getAsInt();
        if (json.has("iterations")) iterations = json.get("iterations").getAsInt();
        if (json.has("anchorX")) anchorX = json.get("anchorX").getAsInt();
        if (json.has("anchorY")) anchorY = json.get("anchorY").getAsInt();
    }
}
