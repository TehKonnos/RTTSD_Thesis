package thesis.rttsd_thesis;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.util.Log;


import androidx.annotation.RequiresApi;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

import thesis.rttsd_thesis.model.entity.ClassificationEntity;

public class SpeedLimitClassifier {

    private final Interpreter interpreter;

    public static String MODEL_FILENAME = "model_224.tflite";

    public static final int INPUT_IMG_SIZE_WIDTH = 224;
    public static final int INPUT_IMG_SIZE_HEIGHT = 224;
    private static final int FLOAT_TYPE_SIZE = 4;
    private static final int PIXEL_SIZE = 3;
    private static final int MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE;
    private static final int IMAGE_MEAN = 0;
    private static final float IMAGE_STD = 255;
    private static final int MAX_CLASSIFICATION_RESULTS = 3;
    public static final float CLASSIFICATION_THRESHOLD = 0.0f;

    public static final List<String> OUTPUT_LABELS = Collections.unmodifiableList(
            Arrays.asList(
                    "Speed limit (20km/h)",
                    "Speed limit (30km/h)",
                    "Speed limit (50km/h)",
                    "Speed limit (60km/h)",
                    "Speed limit (70km/h)",
                    "Speed limit (80km/h)",
                    "End of speed limit (80km/h)",
                    "Speed limit (100km/h)",
                    "Speed limit (120km/h)",
                    "No passing",
                    "No passing for vehicles over 3.5 metric tons",
                    "Right-of-way at the next intersection",
                    "Priority road",
                    "Yield",
                    "Stop",
                    "No vehicles",
                    "Vehicles over 3.5 metric tons prohibited",
                    "No entry",
                    "General caution",
                    "Dangerous curve to the left",
                    "Dangerous curve to the right",
                    "Double curve",
                    "Bumpy road",
                    "Slippery road",
                    "Road narrows on the right",
                    "Road work",
                    "Traffic signals",
                    "Pedestrians",
                    "Children crossing",
                    "Bicycles crossing",
                    "Beware of ice/snow",
                    "Wild animals crossing",
                    "End of all speed and passing limits",
                    "Turn right ahead",
                    "Turn left ahead",
                    "Ahead only",
                    "Go straight or right",
                    "Go straight or left",
                    "Keep right",
                    "Keep left",
                    "Roundabout mandatory",
                    "End of no passing",
                    "End of no passing by vehicles over 3.5 metric tons"
            ));

    private SpeedLimitClassifier(Interpreter interpreter) {
        this.interpreter = interpreter;
    }

    public static SpeedLimitClassifier classifier(AssetManager assetManager, String modelPath) throws IOException {
        ByteBuffer byteBuffer = loadModelFile(assetManager, modelPath);
        Interpreter interpreter = new Interpreter(byteBuffer);
        return new SpeedLimitClassifier(interpreter);
    }

    static ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public List<ClassificationEntity> recognizeImage(Bitmap bitmap,AssetManager assetManager) throws IOException {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float[][] result = new float[1][OUTPUT_LABELS.size()];
        interpreter.run(byteBuffer, result);
        return getSortedResult(result);
    }

    static ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(MODEL_INPUT_SIZE);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < INPUT_IMG_SIZE_WIDTH; ++i) {
            for (int j = 0; j < INPUT_IMG_SIZE_HEIGHT; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    private List<ClassificationEntity> getSortedResult(float[][] resultsArray) {
        PriorityQueue<ClassificationEntity> sortedResults = new PriorityQueue<>(
                MAX_CLASSIFICATION_RESULTS,
                (lhs, rhs) -> Float.compare(rhs.getConfidence(), lhs.getConfidence())
        );

        for (int i = 0; i < OUTPUT_LABELS.size(); ++i) {
            float confidence = resultsArray[0][i];
            if (confidence > CLASSIFICATION_THRESHOLD) {
                OUTPUT_LABELS.size();
                sortedResults.add(new ClassificationEntity(OUTPUT_LABELS.get(i), confidence));
            }
        }

        return new ArrayList<>(sortedResults);
    }


}
