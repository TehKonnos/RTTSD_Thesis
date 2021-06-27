package thesis.rttsd_thesis;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;


import androidx.annotation.RequiresApi;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Scanner;
import java.util.stream.Collectors;

import thesis.rttsd_thesis.model.entity.ClassificationEntity;

public class SpeedLimitClassifier {

    private final Interpreter interpreter;

    public static String MODEL_FILENAME = "43signs.tflite";

    public static final int INPUT_IMG_SIZE_WIDTH = 32;
    public static final int INPUT_IMG_SIZE_HEIGHT = 32;
    private static final int FLOAT_TYPE_SIZE = 4;
    private static final int PIXEL_SIZE = 3;
    private static final int MODEL_INPUT_SIZE = FLOAT_TYPE_SIZE * INPUT_IMG_SIZE_WIDTH * INPUT_IMG_SIZE_HEIGHT * PIXEL_SIZE;
    private static final int IMAGE_MEAN = 0;
    private static final float IMAGE_STD = 255.0f;

    //This list can be taken from notebooks/output/labels_readable.txt
   /* public static final List<String> OUTPUT_LABELS = Collections.unmodifiableList(
            Arrays.asList(
                    "speed limit 10",
                    "speed limit 20",
                    "speed limit 30",
                    "speed limit 40",
                    "speed limit 5",
                    "speed limit 50",
                    "speed limit 60",
                    "speed limit 70",
                    "speed limit 80"
            ));*/
    public static String OUTPUT_LABELS = "43signs.txt";

    private static final int MAX_CLASSIFICATION_RESULTS = 3;
    private static final float CLASSIFICATION_THRESHOLD = 0.1f;

    List<String> labels_arr;

    private SpeedLimitClassifier(Interpreter interpreter) {
        this.interpreter = interpreter;
    }

    public static SpeedLimitClassifier classifier(AssetManager assetManager, String modelPath) throws IOException {
        ByteBuffer byteBuffer = loadModelFile(assetManager, modelPath);
        Interpreter interpreter = new Interpreter(byteBuffer);
        return new SpeedLimitClassifier(interpreter);
    }

    private static ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @RequiresApi(api = Build.VERSION_CODES.O)
    public List<ClassificationEntity> recognizeImage(Bitmap bitmap,AssetManager assetManager) throws IOException {
        InputStream inputreader = assetManager.open(OUTPUT_LABELS);
        BufferedReader buffreader = new BufferedReader(new InputStreamReader(inputreader));

        List<String> labels_arr = new ArrayList<>();

        while (buffreader.readLine() != null) {
            labels_arr.add(buffreader.readLine());
        }

      /*  Scanner s = new Scanner(new File(String.valueOf(assetManager.openFd("43signs.txt"))));
        ArrayList<String> labels_arr = new ArrayList<String>();
        while (s.hasNext()){
            labels_arr.add(s.next());
        }
        s.close();
        */
       // labels_arr = Files.readAllLines(Paths.get(OUTPUT_LABELS));

        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        float[][] result = new float[1][labels_arr.size()];
        interpreter.run(byteBuffer, result);
        return getSortedResult(result);
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
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

        for (int i = 0; i < labels_arr.size(); ++i) {
            float confidence = resultsArray[0][i];
            if (confidence > CLASSIFICATION_THRESHOLD) {
                labels_arr.size();
                sortedResults.add(new ClassificationEntity(labels_arr.get(i), confidence));
            }
        }

        return new ArrayList<>(sortedResults);
    }


}