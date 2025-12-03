# Processor Conversion Guide

This guide documents how to convert legacy monolithic processors to the new modular FXProcessor architecture.

## Architecture Overview

The new system uses:
- **`FXProcessor`** - Interface defining the contract for all processors
- **`FXProcessorBase`** - Abstract base class with helper methods
- **`FXDualInputProcessor`** - Base class for dual-input processors
- **`@FXProcessorInfo`** - Annotation for runtime discovery
- **`FXProcessorRegistry`** - Dynamic registry using classpath scanning

## File Locations

- Processor classes go in: `src/main/java/com/ttennebkram/pipeline/fx/processors/`
- Must end with `Processor.java` (e.g., `GaussianBlurProcessor.java`)

## Step-by-Step Conversion

### 1. Create the Processor Class

```java
package com.ttennebkram.pipeline.fx.processors;

import com.google.gson.JsonObject;
import com.ttennebkram.pipeline.fx.FXNode;
import com.ttennebkram.pipeline.fx.FXPropertiesDialog;
import org.opencv.core.Mat;
// ... other imports

@FXProcessorInfo(nodeType = "YourNodeType", category = "YourCategory")
public class YourProcessor extends FXProcessorBase {

    // Properties with defaults
    private int someProperty = 10;
    private double anotherProperty = 0.5;

    @Override
    public String getNodeType() {
        return "YourNodeType";  // Must match @FXProcessorInfo nodeType
    }

    @Override
    public String getCategory() {
        return "YourCategory";  // Must match @FXProcessorInfo category
    }

    @Override
    public String getDescription() {
        return "Description shown in dialog\nOpenCV method signature";
    }

    @Override
    public Mat process(Mat input) {
        if (isInvalidInput(input)) {
            return input;
        }

        Mat output = new Mat();
        // Your OpenCV processing here
        return output;
    }

    @Override
    public void buildPropertiesDialog(FXPropertiesDialog dialog) {
        dialog.addDescription(getDescription());

        // Add UI controls
        Slider slider = dialog.addSlider("Label:", min, max, currentValue, "%.0f");

        // Save callback - updates processor fields from UI
        dialog.setOnOk(() -> {
            someProperty = (int) slider.getValue();
        });
    }

    @Override
    public void serializeProperties(JsonObject json) {
        json.addProperty("someProperty", someProperty);
        json.addProperty("anotherProperty", anotherProperty);
    }

    @Override
    public void deserializeProperties(JsonObject json) {
        someProperty = getJsonInt(json, "someProperty", 10);
        anotherProperty = getJsonDouble(json, "anotherProperty", 0.5);
    }

    @Override
    public void syncFromFXNode(FXNode node) {
        someProperty = getInt(node.properties, "someProperty", 10);
        anotherProperty = getDouble(node.properties, "anotherProperty", 0.5);
    }

    @Override
    public void syncToFXNode(FXNode node) {
        node.properties.put("someProperty", someProperty);
        node.properties.put("anotherProperty", anotherProperty);
    }
}
```

### 2. For Dual-Input Processors

Extend `FXDualInputProcessor` instead:

```java
@FXProcessorInfo(nodeType = "YourDualInput", category = "Dual Input", dualInput = true)
public class YourDualInputProcessor extends FXDualInputProcessor {

    @Override
    public Mat processDual(Mat input1, Mat input2) {
        // Handle null inputs
        if (input1 == null && input2 == null) return null;
        if (input1 == null) return input2.clone();
        if (input2 == null) return input1.clone();

        // Prepare input2 to match input1 (resize and convert type)
        PreparedInput prep = prepareInput2(input1, input2);

        Mat output = new Mat();
        // Your dual-input processing here

        prep.releaseIfNeeded();
        return output;
    }

    // ... rest same as single-input
}
```

### 3. Remove from Legacy Switch Statements

In `FXNodePropertiesHelper.java`, remove/comment out the case from BOTH switch statements:

**In `addPropertiesForNode()`:**
```java
// case "YourNodeType":
//     addYourNodeTypeProperties(dialog, props);
//     return true;
```

**In `savePropertiesForNode()`:**
The modular processors are automatically skipped due to the registry check at the top.

### 4. Remove from ProcessorFactory (Optional)

The legacy switch in `ProcessorFactory.createImageProcessor()` will be skipped automatically because the registry check comes first. You can optionally remove/comment the legacy case for cleanliness.

## Key Points to Remember

1. **Property names must be consistent** across:
   - `serializeProperties()` / `deserializeProperties()` (JSON keys)
   - `syncFromFXNode()` / `syncToFXNode()` (node.properties keys)
   - They should use the same key names

2. **The `onOk` callback chain**:
   - Your processor's `dialog.setOnOk()` updates processor fields from UI controls
   - `FXNodePropertiesHelper` wraps this to also call `syncToFXNode()`
   - `PipelineEditorApp` wraps again for common node properties (label, etc.)

3. **Helper methods available in FXProcessorBase**:
   - `isInvalidInput(Mat)` - checks for null/empty
   - `getInt(Map, key, default)` - safe int extraction
   - `getDouble(Map, key, default)` - safe double extraction
   - `getJsonInt(JsonObject, key, default)` - safe JSON int
   - `getJsonDouble(JsonObject, key, default)` - safe JSON double

4. **Dialog helper methods in FXPropertiesDialog**:
   - `addSlider(label, min, max, value, format)` - basic slider
   - `addOddKernelSlider(label, value)` - kernel size (odd only, 1-31)
   - `addOddKernelSlider(label, value, max)` - kernel size with custom max
   - `addCheckbox(label, checked)` - checkbox
   - `addComboBox(label, options, selected)` - dropdown
   - `addRadioButtons(label, options, selectedIndex)` - radio group
   - `addSpinner(label, min, max, value)` - integer spinner
   - `addTextField(label, value)` - text input
   - `addDescription(text)` - description label

5. **Runtime discovery**:
   - Classes are discovered automatically via classpath scanning
   - Must be in `com.ttennebkram.pipeline.fx.processors` package
   - Must end with `Processor.java`
   - Must have `@FXProcessorInfo` annotation
   - Must not be abstract

## Already Converted Processors

- `GaussianBlurProcessor` - Blur category
- `InvertProcessor` - Basic category (no properties)
- `ThresholdProcessor` - Basic category
- `CannyEdgeProcessor` - Edge Detection category
- `AddWeightedProcessor` - Dual Input category

## Testing Your Conversion

1. Run `mvn clean compile` to ensure it compiles
2. Start the app and check console for: `[FXProcessorRegistry] Discovered N FX processors: [...]`
3. Your processor should appear in the list
4. Test the properties dialog - change values, click OK
5. Re-open dialog - values should persist
6. Save pipeline, reload - values should persist
7. Test the actual image processing works correctly
