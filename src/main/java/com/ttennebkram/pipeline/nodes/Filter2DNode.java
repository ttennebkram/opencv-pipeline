package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * Filter2D node - applies a custom convolution kernel to the image.
 * User can select kernel size (3x3, 5x5, 7x7, 9x9) and edit individual kernel values.
 */
@NodeInfo(name = "Filter2D", category = "Filter", aliases = {"Filter 2D"})
public class Filter2DNode extends ProcessingNode {
    private int kernelSize = 3;
    private int[] kernelValues; // Stored as flat array

    public Filter2DNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Filter2D w/Kernel", x, y);
        // Initialize with identity kernel (center = 1, rest = 0)
        kernelValues = new int[9];
        kernelValues[4] = 1; // Center element
    }

    // Getters/setters for serialization
    public int getKernelSize() { return kernelSize; }
    public void setKernelSize(int v) {
        kernelSize = v;
        // Resize kernel array if needed
        int newSize = v * v;
        if (kernelValues == null || kernelValues.length != newSize) {
            int[] newValues = new int[newSize];
            // Copy old values if possible, or initialize to identity
            if (kernelValues != null) {
                int oldSize = (int) Math.sqrt(kernelValues.length);
                for (int i = 0; i < v && i < oldSize; i++) {
                    for (int j = 0; j < v && j < oldSize; j++) {
                        newValues[i * v + j] = kernelValues[i * oldSize + j];
                    }
                }
            } else {
                newValues[newSize / 2] = 1; // Center = 1
            }
            kernelValues = newValues;
        }
    }

    public int[] getKernelValues() { return kernelValues; }
    public void setKernelValues(int[] v) { kernelValues = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) {
            return input;
        }

        // Create kernel Mat from values
        Mat kernel = new Mat(kernelSize, kernelSize, CvType.CV_32F);
        for (int i = 0; i < kernelSize; i++) {
            for (int j = 0; j < kernelSize; j++) {
                kernel.put(i, j, kernelValues[i * kernelSize + j]);
            }
        }

        Mat output = new Mat();
        Imgproc.filter2D(input, output, -1, kernel);

        kernel.release();
        return output;
    }

    @Override
    public String getDescription() {
        return "Filter2D\ncv2.filter2D(src, ddepth, kernel)";
    }

    @Override
    public String getDisplayName() {
        return "Filter2D w/Kernel";
    }

    @Override
    public String getCategory() {
        return "Filter";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Filter2D Properties");
        dialog.setLayout(new GridLayout(2, false));

        // Name field at top
        Text nameText = addNameField(dialog, 2);

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 2;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Kernel size selector
        new Label(dialog, SWT.NONE).setText("Kernel Size:");
        Combo sizeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        sizeCombo.setItems(new String[]{"3x3", "5x5", "7x7", "9x9"});
        int sizeIndex = (kernelSize - 3) / 2;
        sizeCombo.select(sizeIndex >= 0 && sizeIndex < 4 ? sizeIndex : 0);

        // Kernel grid container
        Composite gridContainer = new Composite(dialog, SWT.NONE);
        GridData gridGd = new GridData(SWT.CENTER, SWT.CENTER, true, true);
        gridGd.horizontalSpan = 2;
        gridContainer.setLayoutData(gridGd);

        // Array to hold text fields
        Text[][] kernelFields = new Text[9][9]; // Max size
        int[] currentSize = {kernelSize};

        // Function to rebuild grid
        Runnable rebuildGrid = () -> {
            // Dispose old children
            for (Control child : gridContainer.getChildren()) {
                child.dispose();
            }

            int size = currentSize[0];
            gridContainer.setLayout(new GridLayout(size, true));

            // Ensure kernel values array is correct size
            if (kernelValues == null || kernelValues.length != size * size) {
                int[] newValues = new int[size * size];
                // Copy old values centered, or use identity
                if (kernelValues != null) {
                    int oldSize = (int) Math.sqrt(kernelValues.length);
                    int offset = (size - oldSize) / 2;
                    for (int i = 0; i < oldSize && i < size; i++) {
                        for (int j = 0; j < oldSize && j < size; j++) {
                            int newI = i + (offset > 0 ? offset : 0);
                            int newJ = j + (offset > 0 ? offset : 0);
                            int oldI = i + (offset < 0 ? -offset : 0);
                            int oldJ = j + (offset < 0 ? -offset : 0);
                            if (newI < size && newJ < size && oldI < oldSize && oldJ < oldSize) {
                                newValues[newI * size + newJ] = kernelValues[oldI * oldSize + oldJ];
                            }
                        }
                    }
                } else {
                    newValues[size * size / 2] = 1;
                }
                kernelValues = newValues;
            }

            // Create grid of text fields
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    Text field = new Text(gridContainer, SWT.BORDER | SWT.CENTER);
                    field.setText(String.valueOf(kernelValues[i * size + j]));
                    GridData gd = new GridData(SWT.FILL, SWT.CENTER, false, false);
                    gd.widthHint = 40;
                    field.setLayoutData(gd);
                    kernelFields[i][j] = field;

                    // Validate integer input
                    final int row = i, col = j;
                    field.addVerifyListener(e -> {
                        String newText = field.getText().substring(0, e.start) + e.text +
                                        field.getText().substring(e.end);
                        if (!newText.isEmpty() && !newText.equals("-")) {
                            try {
                                Integer.parseInt(newText);
                            } catch (NumberFormatException ex) {
                                e.doit = false;
                            }
                        }
                    });
                }
            }

            gridContainer.layout(true);
            dialog.pack();
        };

        // Initial grid build
        rebuildGrid.run();

        // Size change listener
        sizeCombo.addListener(SWT.Selection, e -> {
            // Save current values before resize
            int oldSize = currentSize[0];
            int[] oldValues = new int[oldSize * oldSize];
            for (int i = 0; i < oldSize; i++) {
                for (int j = 0; j < oldSize; j++) {
                    try {
                        oldValues[i * oldSize + j] = Integer.parseInt(kernelFields[i][j].getText());
                    } catch (NumberFormatException ex) {
                        oldValues[i * oldSize + j] = 0;
                    }
                }
            }
            kernelValues = oldValues;

            // Update size
            currentSize[0] = 3 + sizeCombo.getSelectionIndex() * 2;
            rebuildGrid.run();
        });

        // Predefined Kernels label
        Label presetLabel = new Label(dialog, SWT.NONE);
        presetLabel.setText("Predefined Kernels:");
        GridData presetLabelGd = new GridData(SWT.LEFT, SWT.CENTER, true, false);
        presetLabelGd.horizontalSpan = 2;
        presetLabel.setLayoutData(presetLabelGd);

        // Preset buttons
        Composite presetComp = new Composite(dialog, SWT.NONE);
        presetComp.setLayout(new GridLayout(3, true));
        GridData presetGd = new GridData(SWT.CENTER, SWT.CENTER, true, false);
        presetGd.horizontalSpan = 2;
        presetComp.setLayoutData(presetGd);

        Button identityBtn = new Button(presetComp, SWT.PUSH);
        identityBtn.setText("Identity");
        identityBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        identityBtn.addListener(SWT.Selection, e -> {
            int size = currentSize[0];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    kernelFields[i][j].setText(i == size/2 && j == size/2 ? "1" : "0");
                }
            }
        });

        Button sharpenBtn = new Button(presetComp, SWT.PUSH);
        sharpenBtn.setText("Sharpen/Edge");
        sharpenBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        sharpenBtn.addListener(SWT.Selection, e -> {
            int size = currentSize[0];
            int center = size / 2;
            // Count cross neighbors (same row or column as center, excluding center)
            int neighbors = (size - 1) * 2;
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    if (i == center && j == center) {
                        kernelFields[i][j].setText(String.valueOf(neighbors + 1));
                    } else if (i == center || j == center) {
                        kernelFields[i][j].setText("-1");
                    } else {
                        kernelFields[i][j].setText("0");
                    }
                }
            }
        });

        Button boxBtn = new Button(presetComp, SWT.PUSH);
        boxBtn.setText("Box Blur");
        boxBtn.setLayoutData(new GridData(SWT.FILL, SWT.CENTER, true, false));
        boxBtn.addListener(SWT.Selection, e -> {
            int size = currentSize[0];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    kernelFields[i][j].setText("1");
                }
            }
        });

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData buttonGd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        buttonGd.horizontalSpan = 2;
        buttonComp.setLayoutData(buttonGd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        dialog.setDefaultButton(okBtn);
        okBtn.addListener(SWT.Selection, e -> {
            // Save name field
            saveNameField(nameText);
            // Save kernel values
            int size = currentSize[0];
            kernelSize = size;
            kernelValues = new int[size * size];
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    try {
                        kernelValues[i * size + j] = Integer.parseInt(kernelFields[i][j].getText());
                    } catch (NumberFormatException ex) {
                        kernelValues[i * size + j] = 0;
                    }
                }
            }
            dialog.dispose();
            notifyChanged();
        });

        Button cancelBtn = new Button(buttonComp, SWT.PUSH);
        cancelBtn.setText("Cancel");
        cancelBtn.addListener(SWT.Selection, e -> dialog.dispose());

        dialog.pack();
        // Position dialog near cursor
        Point cursor = shell.getDisplay().getCursorLocation();
        dialog.setLocation(cursor.x, cursor.y);
        dialog.open();
    }

    @Override
    public void serializeProperties(JsonObject json) {
        super.serializeProperties(json);
        json.addProperty("kernelSize", kernelSize);
        JsonArray arr = new JsonArray();
        for (int val : kernelValues) arr.add(val);
        json.add("kernelValues", arr);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        super.deserializeProperties(json);
        if (json.has("kernelSize")) kernelSize = json.get("kernelSize").getAsInt();
        if (json.has("kernelValues")) {
            JsonArray arr = json.getAsJsonArray("kernelValues");
            kernelValues = new int[arr.size()];
            for (int i = 0; i < arr.size(); i++) kernelValues[i] = arr.get(i).getAsInt();
        }
    }
}
