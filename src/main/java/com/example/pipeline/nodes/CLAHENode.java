package com.example.pipeline.nodes;

import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

/**
 * CLAHE (Contrast Limited Adaptive Histogram Equalization) node.
 */
public class CLAHENode extends ProcessingNode {
    private static final String[] COLOR_MODES = {"LAB", "HSV", "Grayscale"};

    private double clipLimit = 2.0;
    private int tileSize = 8;
    private int colorModeIndex = 0; // LAB

    public CLAHENode(Display display, Shell shell, int x, int y) {
        super(display, shell, "CLAHE Contrast", x, y);
    }

    // Getters/setters for serialization
    public double getClipLimit() { return clipLimit; }
    public void setClipLimit(double v) { clipLimit = v; }
    public int getTileSize() { return tileSize; }
    public void setTileSize(int v) { tileSize = v; }
    public int getColorModeIndex() { return colorModeIndex; }
    public void setColorModeIndex(int v) { colorModeIndex = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        CLAHE clahe = Imgproc.createCLAHE(clipLimit, new Size(tileSize, tileSize));
        Mat output = new Mat();

        if (colorModeIndex == 2) { // Grayscale
            Mat gray = new Mat();
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
            Mat result = new Mat();
            clahe.apply(gray, result);
            Imgproc.cvtColor(result, output, Imgproc.COLOR_GRAY2BGR);
            gray.release();
            result.release();
        } else if (colorModeIndex == 0) { // LAB
            Mat lab = new Mat();
            Imgproc.cvtColor(input, lab, Imgproc.COLOR_BGR2Lab);
            java.util.List<Mat> channels = new java.util.ArrayList<>();
            Core.split(lab, channels);
            Mat lChannel = new Mat();
            clahe.apply(channels.get(0), lChannel);
            channels.set(0, lChannel);
            Core.merge(channels, lab);
            Imgproc.cvtColor(lab, output, Imgproc.COLOR_Lab2BGR);
            for (Mat ch : channels) ch.release();
            lab.release();
        } else { // HSV
            Mat hsv = new Mat();
            Imgproc.cvtColor(input, hsv, Imgproc.COLOR_BGR2HSV);
            java.util.List<Mat> channels = new java.util.ArrayList<>();
            Core.split(hsv, channels);
            Mat vChannel = new Mat();
            clahe.apply(channels.get(2), vChannel);
            channels.set(2, vChannel);
            Core.merge(channels, hsv);
            Imgproc.cvtColor(hsv, output, Imgproc.COLOR_HSV2BGR);
            for (Mat ch : channels) ch.release();
            hsv.release();
        }

        return output;
    }

    @Override
    public String getDescription() {
        return "CLAHE: Contrast Limited Adaptive Histogram Equalization\ncv2.createCLAHE(clipLimit, tileGridSize)";
    }

    @Override
    public String getDisplayName() {
        return "CLAHE Contrast";
    }

    @Override
    public String getCategory() {
        return "Basic";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("CLAHE Properties");
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

        // Clip Limit
        new Label(dialog, SWT.NONE).setText("Clip Limit:");
        Scale clipScale = new Scale(dialog, SWT.HORIZONTAL);
        clipScale.setMinimum(10);
        clipScale.setMaximum(400);
        clipScale.setSelection((int)(clipLimit * 10));
        clipScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label clipLabel = new Label(dialog, SWT.NONE);
        clipLabel.setText(String.format("%.1f", clipLimit));
        clipScale.addListener(SWT.Selection, e -> {
            double val = clipScale.getSelection() / 10.0;
            clipLabel.setText(String.format("%.1f", val));
        });

        // Tile Size
        new Label(dialog, SWT.NONE).setText("Tile Size:");
        Scale tileScale = new Scale(dialog, SWT.HORIZONTAL);
        tileScale.setMinimum(2);
        tileScale.setMaximum(32);
        tileScale.setSelection(tileSize);
        tileScale.setLayoutData(new GridData(200, SWT.DEFAULT));

        Label tileLabel = new Label(dialog, SWT.NONE);
        tileLabel.setText(String.valueOf(tileSize));
        tileScale.addListener(SWT.Selection, e -> tileLabel.setText(String.valueOf(tileScale.getSelection())));

        // Color Mode
        new Label(dialog, SWT.NONE).setText("Apply to:");
        Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modeCombo.setItems(COLOR_MODES);
        modeCombo.select(colorModeIndex);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        modeCombo.setLayoutData(comboGd);

        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            clipLimit = clipScale.getSelection() / 10.0;
            tileSize = tileScale.getSelection();
            colorModeIndex = modeCombo.getSelectionIndex();
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
