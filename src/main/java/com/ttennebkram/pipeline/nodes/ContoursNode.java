package com.ttennebkram.pipeline.nodes;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.registry.NodeInfo;
import org.eclipse.swt.SWT;
import org.eclipse.swt.graphics.Point;
import org.eclipse.swt.layout.GridData;
import org.eclipse.swt.layout.GridLayout;
import org.eclipse.swt.widgets.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * Enhanced Contours detection node with sorting, filtering, and multiple draw modes.
 */
@NodeInfo(
    name = "Contours",
    category = "Detection",
    aliases = {}
)
public class ContoursNode extends ProcessingNode {
    private int thresholdValue = 127;
    private int retrievalMode = 0; // 0=EXTERNAL, 1=LIST, 2=CCOMP, 3=TREE
    private int approxMethod = 1;  // 0=NONE, 1=SIMPLE, 2=TC89_L1, 3=TC89_KCOS
    private int thickness = 2;
    private int colorR = 0, colorG = 255, colorB = 0;
    private boolean showOriginal = true;
    private int sortMethod = 0;    // Sorting method index
    private int minIndex = 0;      // Minimum contour index to draw
    private int maxIndex = -1;     // Maximum contour index (-1 = all)
    private int drawMode = 0;      // Draw mode index

    private static final String[] RETRIEVAL_MODES = {"External", "List", "Two-level", "Tree"};
    private static final int[] RETRIEVAL_VALUES = {
        Imgproc.RETR_EXTERNAL, Imgproc.RETR_LIST, Imgproc.RETR_CCOMP, Imgproc.RETR_TREE
    };

    private static final String[] APPROX_METHODS = {"None", "Simple", "TC89_L1", "TC89_KCOS"};
    private static final int[] APPROX_VALUES = {
        Imgproc.CHAIN_APPROX_NONE, Imgproc.CHAIN_APPROX_SIMPLE,
        Imgproc.CHAIN_APPROX_TC89_L1, Imgproc.CHAIN_APPROX_TC89_KCOS
    };

    private static final String[] SORT_METHODS = {
        "None", "Top-Bottom, Left-Right", "Left-Right, Top-Bottom",
        "Area (Descending)", "Area (Ascending)",
        "Perimeter (Descending)", "Perimeter (Ascending)",
        "Circularity (Descending)", "Circularity (Ascending)",
        "Aspect Ratio (Descending)", "Aspect Ratio (Ascending)",
        "Extent (Descending)", "Extent (Ascending)",
        "Solidity (Descending)", "Solidity (Ascending)"
    };

    private static final String[] DRAW_MODES = {
        "Contours", "Bounding Rectangles", "Rotated Rectangles",
        "Enclosing Circles", "Fitted Ellipses", "Convex Hulls", "Centroids"
    };

    public ContoursNode(Display display, Shell shell, int x, int y) {
        super(display, shell, "Contours", x, y);
    }

    // Getters/setters for serialization
    public int getThresholdValue() { return thresholdValue; }
    public void setThresholdValue(int v) { thresholdValue = v; }
    public int getRetrievalMode() { return retrievalMode; }
    public void setRetrievalMode(int v) { retrievalMode = v; }
    public int getApproxMethod() { return approxMethod; }
    public void setApproxMethod(int v) { approxMethod = v; }
    public int getThickness() { return thickness; }
    public void setThickness(int v) { thickness = v; }
    public int getColorR() { return colorR; }
    public void setColorR(int v) { colorR = v; }
    public int getColorG() { return colorG; }
    public void setColorG(int v) { colorG = v; }
    public int getColorB() { return colorB; }
    public void setColorB(int v) { colorB = v; }
    public boolean getShowOriginal() { return showOriginal; }
    public void setShowOriginal(boolean v) { showOriginal = v; }
    public int getSortMethod() { return sortMethod; }
    public void setSortMethod(int v) { sortMethod = v; }
    public int getMinIndex() { return minIndex; }
    public void setMinIndex(int v) { minIndex = v; }
    public int getMaxIndex() { return maxIndex; }
    public void setMaxIndex(int v) { maxIndex = v; }
    public int getDrawMode() { return drawMode; }
    public void setDrawMode(int v) { drawMode = v; }

    @Override
    public Mat process(Mat input) {
        if (!enabled || input == null || input.empty()) return input;

        // Convert to grayscale
        Mat gray = new Mat();
        if (input.channels() == 3) {
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        } else {
            gray = input.clone();
        }

        // Apply threshold
        Mat binary = new Mat();
        Imgproc.threshold(gray, binary, thresholdValue, 255, Imgproc.THRESH_BINARY);

        // Find contours
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binary, contours, hierarchy,
            RETRIEVAL_VALUES[retrievalMode], APPROX_VALUES[approxMethod]);

        // Sort contours if needed
        if (sortMethod > 0 && contours.size() > 0) {
            contours = sortContours(contours, sortMethod);
        }

        // Filter contours by index range
        List<MatOfPoint> filteredContours = filterContoursByIndex(contours);

        // Create output image
        Mat result;
        if (showOriginal) {
            if (input.channels() == 1) {
                result = new Mat();
                Imgproc.cvtColor(input, result, Imgproc.COLOR_GRAY2BGR);
            } else {
                result = input.clone();
            }
        } else {
            result = Mat.zeros(input.size(), CvType.CV_8UC3);
        }

        // Draw based on selected mode
        Scalar color = new Scalar(colorB, colorG, colorR);
        drawContours(result, filteredContours, color);

        // Cleanup
        gray.release();
        binary.release();
        hierarchy.release();

        return result;
    }

    private List<MatOfPoint> filterContoursByIndex(List<MatOfPoint> contours) {
        if (contours.isEmpty()) return contours;

        int min = Math.max(0, minIndex);
        int max = maxIndex < 0 ? contours.size() - 1 : Math.min(maxIndex, contours.size() - 1);

        if (min > max || min >= contours.size()) {
            return new ArrayList<>();
        }

        return new ArrayList<>(contours.subList(min, max + 1));
    }

    private List<MatOfPoint> sortContours(List<MatOfPoint> contours, int method) {
        List<MatOfPoint> sorted = new ArrayList<>(contours);

        switch (method) {
            case 1: // Top-Bottom, Left-Right
                Collections.sort(sorted, (c1, c2) -> {
                    Rect r1 = Imgproc.boundingRect(c1);
                    Rect r2 = Imgproc.boundingRect(c2);
                    int yCompare = Integer.compare(r1.y, r2.y);
                    return yCompare != 0 ? yCompare : Integer.compare(r1.x, r2.x);
                });
                break;
            case 2: // Left-Right, Top-Bottom
                Collections.sort(sorted, (c1, c2) -> {
                    Rect r1 = Imgproc.boundingRect(c1);
                    Rect r2 = Imgproc.boundingRect(c2);
                    int xCompare = Integer.compare(r1.x, r2.x);
                    return xCompare != 0 ? xCompare : Integer.compare(r1.y, r2.y);
                });
                break;
            case 3: // Area Descending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(Imgproc.contourArea(c2), Imgproc.contourArea(c1)));
                break;
            case 4: // Area Ascending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(Imgproc.contourArea(c1), Imgproc.contourArea(c2)));
                break;
            case 5: // Perimeter Descending
                Collections.sort(sorted, (c1, c2) -> {
                    MatOfPoint2f c1f = new MatOfPoint2f(c1.toArray());
                    MatOfPoint2f c2f = new MatOfPoint2f(c2.toArray());
                    double p1 = Imgproc.arcLength(c1f, true);
                    double p2 = Imgproc.arcLength(c2f, true);
                    return Double.compare(p2, p1);
                });
                break;
            case 6: // Perimeter Ascending
                Collections.sort(sorted, (c1, c2) -> {
                    MatOfPoint2f c1f = new MatOfPoint2f(c1.toArray());
                    MatOfPoint2f c2f = new MatOfPoint2f(c2.toArray());
                    double p1 = Imgproc.arcLength(c1f, true);
                    double p2 = Imgproc.arcLength(c2f, true);
                    return Double.compare(p1, p2);
                });
                break;
            case 7: // Circularity Descending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateCircularity(c2), calculateCircularity(c1)));
                break;
            case 8: // Circularity Ascending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateCircularity(c1), calculateCircularity(c2)));
                break;
            case 9: // Aspect Ratio Descending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateAspectRatio(c2), calculateAspectRatio(c1)));
                break;
            case 10: // Aspect Ratio Ascending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateAspectRatio(c1), calculateAspectRatio(c2)));
                break;
            case 11: // Extent Descending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateExtent(c2), calculateExtent(c1)));
                break;
            case 12: // Extent Ascending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateExtent(c1), calculateExtent(c2)));
                break;
            case 13: // Solidity Descending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateSolidity(c2), calculateSolidity(c1)));
                break;
            case 14: // Solidity Ascending
                Collections.sort(sorted, (c1, c2) ->
                    Double.compare(calculateSolidity(c1), calculateSolidity(c2)));
                break;
        }

        return sorted;
    }

    private double calculateCircularity(MatOfPoint contour) {
        double area = Imgproc.contourArea(contour);
        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        double perimeter = Imgproc.arcLength(contour2f, true);
        if (perimeter == 0) return 0;
        return 4 * Math.PI * area / (perimeter * perimeter);
    }

    private double calculateAspectRatio(MatOfPoint contour) {
        Rect rect = Imgproc.boundingRect(contour);
        if (rect.height == 0) return 0;
        return (double) rect.width / rect.height;
    }

    private double calculateExtent(MatOfPoint contour) {
        double area = Imgproc.contourArea(contour);
        Rect rect = Imgproc.boundingRect(contour);
        double rectArea = rect.width * rect.height;
        if (rectArea == 0) return 0;
        return area / rectArea;
    }

    private double calculateSolidity(MatOfPoint contour) {
        double area = Imgproc.contourArea(contour);
        MatOfInt hull = new MatOfInt();
        Imgproc.convexHull(contour, hull);

        // Convert hull indices to points
        org.opencv.core.Point[] contourPoints = contour.toArray();
        int[] hullIndices = hull.toArray();
        org.opencv.core.Point[] hullPoints = new org.opencv.core.Point[hullIndices.length];
        for (int i = 0; i < hullIndices.length; i++) {
            hullPoints[i] = contourPoints[hullIndices[i]];
        }
        MatOfPoint hullContour = new MatOfPoint(hullPoints);
        double hullArea = Imgproc.contourArea(hullContour);

        if (hullArea == 0) return 0;
        return area / hullArea;
    }

    private void drawContours(Mat result, List<MatOfPoint> contours, Scalar color) {
        switch (drawMode) {
            case 0: // Contours
                Imgproc.drawContours(result, contours, -1, color, thickness);
                break;
            case 1: // Bounding Rectangles
                for (MatOfPoint contour : contours) {
                    Rect rect = Imgproc.boundingRect(contour);
                    Imgproc.rectangle(result, rect, color, thickness);
                }
                break;
            case 2: // Rotated Rectangles
                for (MatOfPoint contour : contours) {
                    if (contour.toArray().length >= 5) {
                        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                        RotatedRect rotRect = Imgproc.minAreaRect(contour2f);
                        org.opencv.core.Point[] vertices = new org.opencv.core.Point[4];
                        rotRect.points(vertices);
                        for (int i = 0; i < 4; i++) {
                            Imgproc.line(result, vertices[i], vertices[(i + 1) % 4], color, thickness);
                        }
                    }
                }
                break;
            case 3: // Enclosing Circles
                for (MatOfPoint contour : contours) {
                    MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                    org.opencv.core.Point center = new org.opencv.core.Point();
                    float[] radius = new float[1];
                    Imgproc.minEnclosingCircle(contour2f, center, radius);
                    Imgproc.circle(result, center, (int) radius[0], color, thickness);
                }
                break;
            case 4: // Fitted Ellipses
                for (MatOfPoint contour : contours) {
                    if (contour.toArray().length >= 5) {
                        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
                        RotatedRect ellipse = Imgproc.fitEllipse(contour2f);
                        Imgproc.ellipse(result, ellipse, color, thickness);
                    }
                }
                break;
            case 5: // Convex Hulls
                for (MatOfPoint contour : contours) {
                    MatOfInt hullIndices = new MatOfInt();
                    Imgproc.convexHull(contour, hullIndices);

                    org.opencv.core.Point[] contourPoints = contour.toArray();
                    int[] indices = hullIndices.toArray();
                    org.opencv.core.Point[] hullPoints = new org.opencv.core.Point[indices.length];
                    for (int i = 0; i < indices.length; i++) {
                        hullPoints[i] = contourPoints[indices[i]];
                    }

                    List<MatOfPoint> hullList = new ArrayList<>();
                    hullList.add(new MatOfPoint(hullPoints));
                    Imgproc.drawContours(result, hullList, -1, color, thickness);
                }
                break;
            case 6: // Centroids
                for (MatOfPoint contour : contours) {
                    Moments moments = Imgproc.moments(contour);
                    if (moments.m00 != 0) {
                        int cx = (int) (moments.m10 / moments.m00);
                        int cy = (int) (moments.m01 / moments.m00);
                        Imgproc.circle(result, new org.opencv.core.Point(cx, cy), 5, color, -1);
                    }
                }
                break;
        }
    }

    @Override
    public String getDescription() {
        return "Contour Detection\ncv2.findContours(image, mode, method)";
    }

    @Override
    public String getDisplayName() {
        return "Contours";
    }

    @Override
    public String getCategory() {
        return "Detection";
    }

    @Override
    public void showPropertiesDialog() {
        Shell dialog = new Shell(shell, SWT.DIALOG_TRIM | SWT.APPLICATION_MODAL);
        dialog.setText("Contours Properties");
        dialog.setLayout(new GridLayout(3, false));

        // Method signature
        Label sigLabel = new Label(dialog, SWT.NONE);
        sigLabel.setText(getDescription());
        sigLabel.setForeground(dialog.getDisplay().getSystemColor(SWT.COLOR_DARK_GRAY));
        GridData sigGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sigGd.horizontalSpan = 3;
        sigLabel.setLayoutData(sigGd);

        // Separator
        Label sep1 = new Label(dialog, SWT.SEPARATOR | SWT.HORIZONTAL);
        GridData sepGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        sepGd.horizontalSpan = 3;
        sep1.setLayoutData(sepGd);

        // Show Original checkbox
        Button showOrigBtn = new Button(dialog, SWT.CHECK);
        showOrigBtn.setText("Show Original Background");
        showOrigBtn.setSelection(showOriginal);
        GridData showGd = new GridData(SWT.FILL, SWT.CENTER, true, false);
        showGd.horizontalSpan = 3;
        showOrigBtn.setLayoutData(showGd);

        // Threshold
        new Label(dialog, SWT.NONE).setText("Threshold:");
        Scale threshScale = new Scale(dialog, SWT.HORIZONTAL);
        threshScale.setMinimum(0);
        threshScale.setMaximum(255);
        // Clamp slider position to valid range, but keep actual value
        int threshSliderPos = Math.min(Math.max(thresholdValue, 0), 255);
        threshScale.setSelection(threshSliderPos);
        threshScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label threshLabel = new Label(dialog, SWT.NONE);
        threshLabel.setText(String.valueOf(thresholdValue)); // Show real value
        threshScale.addListener(SWT.Selection, e -> threshLabel.setText(String.valueOf(threshScale.getSelection())));

        // Retrieval Mode
        new Label(dialog, SWT.NONE).setText("Retrieval Mode:");
        Combo modeCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        modeCombo.setItems(RETRIEVAL_MODES);
        modeCombo.select(retrievalMode);
        GridData comboGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        comboGd.horizontalSpan = 2;
        modeCombo.setLayoutData(comboGd);

        // Approximation Method
        new Label(dialog, SWT.NONE).setText("Approximation:");
        Combo approxCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        approxCombo.setItems(APPROX_METHODS);
        approxCombo.select(approxMethod);
        GridData approxGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        approxGd.horizontalSpan = 2;
        approxCombo.setLayoutData(approxGd);

        // Sort Method
        new Label(dialog, SWT.NONE).setText("Sort By:");
        Combo sortCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        sortCombo.setItems(SORT_METHODS);
        sortCombo.select(sortMethod);
        GridData sortGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        sortGd.horizontalSpan = 2;
        sortCombo.setLayoutData(sortGd);

        // Min Index
        new Label(dialog, SWT.NONE).setText("Min Index:");
        Spinner minSpinner = new Spinner(dialog, SWT.BORDER);
        minSpinner.setMinimum(0);
        minSpinner.setMaximum(1000);
        minSpinner.setSelection(minIndex);
        GridData minGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        minGd.horizontalSpan = 2;
        minSpinner.setLayoutData(minGd);

        // Max Index
        new Label(dialog, SWT.NONE).setText("Max Index (-1=all):");
        Spinner maxSpinner = new Spinner(dialog, SWT.BORDER);
        maxSpinner.setMinimum(-1);
        maxSpinner.setMaximum(1000);
        maxSpinner.setSelection(maxIndex);
        GridData maxGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        maxGd.horizontalSpan = 2;
        maxSpinner.setLayoutData(maxGd);

        // Draw Mode
        new Label(dialog, SWT.NONE).setText("Draw Mode:");
        Combo drawCombo = new Combo(dialog, SWT.DROP_DOWN | SWT.READ_ONLY);
        drawCombo.setItems(DRAW_MODES);
        drawCombo.select(drawMode);
        GridData drawGd = new GridData(SWT.FILL, SWT.CENTER, false, false);
        drawGd.horizontalSpan = 2;
        drawCombo.setLayoutData(drawGd);

        // Thickness
        new Label(dialog, SWT.NONE).setText("Line Thickness:");
        Scale thickScale = new Scale(dialog, SWT.HORIZONTAL);
        thickScale.setMinimum(1);
        thickScale.setMaximum(10);
        // Clamp slider position to valid range, but keep actual value
        int thickSliderPos = Math.min(Math.max(thickness, 1), 10);
        thickScale.setSelection(thickSliderPos);
        thickScale.setLayoutData(new GridData(200, SWT.DEFAULT));
        Label thickLabel = new Label(dialog, SWT.NONE);
        thickLabel.setText(String.valueOf(thickness)); // Show real value
        thickScale.addListener(SWT.Selection, e -> thickLabel.setText(String.valueOf(thickScale.getSelection())));

        // Buttons
        Composite buttonComp = new Composite(dialog, SWT.NONE);
        buttonComp.setLayout(new GridLayout(2, true));
        GridData gd = new GridData(SWT.RIGHT, SWT.CENTER, true, false);
        gd.horizontalSpan = 3;
        buttonComp.setLayoutData(gd);

        Button okBtn = new Button(buttonComp, SWT.PUSH);
        okBtn.setText("OK");
        okBtn.addListener(SWT.Selection, e -> {
            showOriginal = showOrigBtn.getSelection();
            thresholdValue = threshScale.getSelection();
            retrievalMode = modeCombo.getSelectionIndex();
            approxMethod = approxCombo.getSelectionIndex();
            sortMethod = sortCombo.getSelectionIndex();
            minIndex = minSpinner.getSelection();
            maxIndex = maxSpinner.getSelection();
            drawMode = drawCombo.getSelectionIndex();
            thickness = thickScale.getSelection();
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
        json.addProperty("thresholdValue", thresholdValue);
        json.addProperty("retrievalMode", retrievalMode);
        json.addProperty("approxMethod", approxMethod);
        json.addProperty("thickness", thickness);
        json.addProperty("colorR", colorR);
        json.addProperty("colorG", colorG);
        json.addProperty("colorB", colorB);
        json.addProperty("showOriginal", showOriginal);
        json.addProperty("sortMethod", sortMethod);
        json.addProperty("minIndex", minIndex);
        json.addProperty("maxIndex", maxIndex);
        json.addProperty("drawMode", drawMode);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        if (json.has("thresholdValue")) thresholdValue = json.get("thresholdValue").getAsInt();
        if (json.has("retrievalMode")) retrievalMode = json.get("retrievalMode").getAsInt();
        if (json.has("approxMethod")) approxMethod = json.get("approxMethod").getAsInt();
        if (json.has("thickness")) thickness = json.get("thickness").getAsInt();
        if (json.has("colorR")) colorR = json.get("colorR").getAsInt();
        if (json.has("colorG")) colorG = json.get("colorG").getAsInt();
        if (json.has("colorB")) colorB = json.get("colorB").getAsInt();
        if (json.has("showOriginal")) showOriginal = json.get("showOriginal").getAsBoolean();
        if (json.has("sortMethod")) sortMethod = json.get("sortMethod").getAsInt();
        if (json.has("minIndex")) minIndex = json.get("minIndex").getAsInt();
        if (json.has("maxIndex")) maxIndex = json.get("maxIndex").getAsInt();
        if (json.has("drawMode")) drawMode = json.get("drawMode").getAsInt();
    }
}
