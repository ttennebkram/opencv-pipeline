package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Color;
import org.eclipse.swt.graphics.Font;
import org.eclipse.swt.graphics.GC;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.Mat;

/**
 * Clone node - duplicates input to multiple outputs.
 * Supports 2-4 configurable outputs.
 */
@NodeInfo(
    name = "Clone",
    category = "Utility",
    aliases = {"Duplicate", "Split", "Tee"}
)
public class CloneNode extends MultiOutputNode {
    private static final int MIN_OUTPUTS = 2;
    private static final int MAX_OUTPUTS = 4;

    public CloneNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Clone", x, y);
        initMultiOutputQueues(2); // Default to 2 outputs
    }

    /**
     * Set the number of outputs (2-4).
     */
    public void setNumOutputs(int count) {
        count = Math.max(MIN_OUTPUTS, Math.min(MAX_OUTPUTS, count));
        initMultiOutputQueues(count);
    }

    @Override
    public Mat process(Mat input) {
        // Clone node just passes through - the base class handles sending to all outputs
        return input != null ? input.clone() : null;
    }

    @Override
    public void paint(GC gc) {
        // Draw node background - light purple for utility nodes
        gc.setBackground(new Color(230, 230, 255));
        gc.fillRoundRectangle(x, y, width, height, 10, 10);

        // Draw border
        gc.setForeground(new Color(80, 80, 150));
        gc.setLineWidth(2);
        gc.drawRoundRectangle(x, y, width, height, 10, 10);

        // Draw title with output count
        gc.setForeground(display.getSystemColor(SWT.COLOR_BLACK));
        Font boldFont = new Font(display, "Arial", 10, SWT.BOLD);
        gc.setFont(boldFont);
        gc.drawString(name + " (" + numOutputs + ")", x + 10, y + 5, true);
        boldFont.dispose();

        // Draw thread priority label
        Font smallFont = new Font(display, "Arial", 8, SWT.NORMAL);
        gc.setFont(smallFont);
        gc.setForeground(display.getSystemColor(SWT.COLOR_DARK_GRAY));
        gc.drawString(getThreadPriorityLabel(), x + 10, y + 20, true);
        smallFont.dispose();

        // Draw thumbnail if available
        if (thumbnail != null && !thumbnail.isDisposed()) {
            int thumbX = x + 40;
            int thumbY = y + 35;
            gc.drawImage(thumbnail, thumbX, thumbY);
        } else {
            gc.setForeground(display.getSystemColor(SWT.COLOR_GRAY));
            gc.drawString("(no output)", x + 45, y + 50, true);
        }

        // Draw connection points (from MultiOutputNode)
        drawMultiOutputConnectionPoints(gc);
    }

    @Override
    public String getDescription() {
        return "Clone/Duplicate\nSends input to multiple outputs";
    }

    @Override
    public String getDisplayName() {
        return "Clone";
    }

    @Override
    public String getCategory() {
        return "Utility";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Clone Properties");
        dialog.setLayout(new GridLayout(2, false));

        Label infoLabel = new Label(dialog, SWT.NONE);
        infoLabel.setText("Duplicates input to multiple outputs.");
        GridData infoGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        infoGd.horizontalSpan = 2;
        infoLabel.setLayoutData(infoGd);

        // Separator
        Label sep = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 2;
        sep.setLayoutData(sepGd);

        // Output count
        new Label(dialog, SWT.NONE).setText("Number of outputs:");
        Combo outputCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        outputCombo.setItems("2", "3", "4");
        outputCombo.select(numOutputs - MIN_OUTPUTS);

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 2;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            int newCount = outputCombo.getSelectionIndex() + MIN_OUTPUTS;
            setNumOutputs(newCount);
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

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("numOutputs", numOutputs);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("numOutputs")) {
            setNumOutputs(json.get("numOutputs").getAsInt());
        }
    }
}
