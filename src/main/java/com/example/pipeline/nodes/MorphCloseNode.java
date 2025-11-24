package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Morph Close (Morphological) node.
 */
public class MorphCloseNode extends ProcessingNode {
    private static final String[] SHAPE_NAMES = {"Rectangle", "Ellipse", "Cross"};
    private static final int[] SHAPE_VALUES = {Imgproc.MORPH_RECT, Imgproc.MORPH_ELLIPSE, Imgproc.MORPH_CROSS};

    private int kernelSize = 5;
    private int shapeIndex = 0;
    private int iterations = 1;

    public MorphCloseNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Morph Close", x, y);
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

        int ksize = (kernelSize % 2 == 0) ? kernelSize + 1 : kernelSize;
        Mat kernel = Imgproc.getStructuringElement(SHAPE_VALUES[shapeIndex], new Size(ksize, ksize));

        Mat output = new Mat();
        Imgproc.morphologyEx(input, output, Imgproc.MORPH_CLOSE, kernel, new org.opencv.core.Point(-1, -1), iterations);

        kernel.release();
        return output;
    }

    @Override
    public String getDescription() {
        return "Morphological Closing\ncv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)";
    }

    @Override
    public String getDisplayName() {
        return "Morph Close";
    }

    @Override
    public String getCategory() {
        return "Morphological";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Morph Close Properties");
        dialog.setLayout(new GridLayout(3, false));

        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 3;
        sep.setLayoutData(sepGd);

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

        new Label(dialog, SWT.NONE).setText("Kernel Shape:");
        Combo shapeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        shapeCombo.setItems(SHAPE_NAMES);
        shapeCombo.select(shapeIndex);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        shapeCombo.setLayoutData(comboGd);

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

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            kernelSize = kScale.getSelection();
            shapeIndex = shapeCombo.getSelectionIndex();
            iterations = iterScale.getSelection();
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }
}
