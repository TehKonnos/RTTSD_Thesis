/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package thesis.rttsd_thesis;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.widget.SwitchCompat;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.task.vision.classifier.Classifications;
import org.tensorflow.lite.task.vision.classifier.ImageClassifier;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import thesis.rttsd_thesis.Detection.Classifier.Recognition;
import thesis.rttsd_thesis.Detection.YoloV5Classifier;
import thesis.rttsd_thesis.customview.OverlayView;
import thesis.rttsd_thesis.env.BorderedText;
import thesis.rttsd_thesis.env.ImageUtils;
import thesis.rttsd_thesis.env.Logger;
import thesis.rttsd_thesis.mediaplayer.MediaPlayerHolder;
import thesis.rttsd_thesis.tracking.MultiBoxTracker;

import static thesis.rttsd_thesis.ImageUtils.prepareImageForClassification;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 640;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "sign_recognitionQ.tflite";
  public static final String TF_OD_API_LABELS_FILE = "sign_recognition.txt";
  // Minimum detection confidence to track a detection.
  public static float MINIMUM_CONFIDENCE_TF_OD_API = 0.6f;
  private static final boolean MAINTAIN_ASPECT = true;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  //For Classification
  public static float CLASSIFICATION_THRESHOLD = 0.6f;
  public static String MODEL_FILENAME = "model82Q.tflite";
  private static SwitchCompat notification;
  private MediaPlayerHolder mediaPlayerHolder;

  private int maximumResults = 3;
  OverlayView trackingOverlay;

  private YoloV5Classifier detector;
  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  protected void onSaveInstanceState(@NonNull Bundle outState) {
    super.onSaveInstanceState(outState);
  }

  protected void onRestoreInstanceState(Bundle savedInstanceState) {
    super.onRestoreInstanceState(savedInstanceState);
  }

  @SuppressLint("DefaultLocale")
  public void setupViews() {
    TextView confidence = findViewById(R.id.confidence_value);
    confidence.setText(String.format("%.2f", CLASSIFICATION_THRESHOLD));

    mediaPlayerHolder = getMediaPlayerHolder();


    notification = findViewById(R.id.notification_switch);
    notification.setOnCheckedChangeListener((buttonView, isChecked) -> {
      if (!isChecked)
          mediaPlayerHolder.reset();
    });

    SeekBar confidenceSeekBar = findViewById(R.id.confidence_seek);
    confidenceSeekBar.setMax(99);
    confidenceSeekBar.setProgress((int) (MINIMUM_CONFIDENCE_TF_OD_API * 100));

    confidenceSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
      @Override
      public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
          CLASSIFICATION_THRESHOLD = progress / 100.0F;
        confidence.setText(String.format("%.2f", CLASSIFICATION_THRESHOLD));
      }

      @Override
      public void onStartTrackingTouch(SeekBar seekBar) {
      }

      @Override
      public void onStopTrackingTouch(SeekBar seekBar) {
      }
    });
  }

    @Override
    public void onPreviewSizeChosen ( final Size size, final int rotation){
      final float textSizePx =
              TypedValue.applyDimension(
                      TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        BorderedText borderedText = new BorderedText(textSizePx);
      borderedText.setTypeface(Typeface.MONOSPACE);

      tracker = new MultiBoxTracker(this);

      int cropSize = TF_OD_API_INPUT_SIZE;
        try {
            detector =
                    YoloV5Classifier.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_IS_QUANTIZED,
                            TF_OD_API_INPUT_SIZE);
        } catch (final IOException e) {
            e.printStackTrace();
            LOGGER.e(e, "Exception initializing classifier!");
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }

      previewWidth = size.getWidth();
      previewHeight = size.getHeight();

      int sensorOrientation = rotation - getScreenOrientation();
      LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

      LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
      rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
      croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

      frameToCropTransform =
              ImageUtils.getTransformationMatrix(
                      previewWidth, previewHeight,
                      cropSize, cropSize,
                      sensorOrientation, MAINTAIN_ASPECT);

      cropToFrameTransform = new Matrix();
      frameToCropTransform.invert(cropToFrameTransform);

      trackingOverlay = findViewById(R.id.tracking_overlay);
      trackingOverlay.addCallback(
              canvas -> {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              });

      tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    }

    @Override
    protected void processImage () {
      ++timestamp;
      final long currTimestamp = timestamp;
      trackingOverlay.postInvalidate();

      // No mutex needed as this method is not reentrant.
      if (computingDetection) {
        readyForNextImage();
        return;
      }
      computingDetection = true;
      LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

      rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

      readyForNextImage();

      final Canvas canvas = new Canvas(croppedBitmap);
      canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
      // For examining the actual TF input.
      if (SAVE_PREVIEW_BITMAP) {
        ImageUtils.saveBitmap(croppedBitmap);
      }

      runInBackground(
              () -> {

                LOGGER.i("Running detection on image " + currTimestamp);
                final long startTime = SystemClock.uptimeMillis();
                List<Recognition> results = detector.recognizeImage(croppedBitmap);


                cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                  final Canvas canvas1 = new Canvas(cropCopyBitmap);
                  final Paint paint = new Paint();
                  paint.setColor(Color.RED);
                  paint.setStyle(Paint.Style.STROKE);
                  paint.setStrokeWidth(2.0f);

                float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;

                  final List<Recognition> mappedRecognitions =
                        new ArrayList<>();

                int cResults = 0;
                for (Recognition result : results) {
                  RectF location = result.getLocation();
                  if (location != null && result.getConfidence() >= minimumConfidence) {
                      classify(result);

                      cResults++;
                    if (cResults > maximumResults) break;
                    canvas1.drawRect(location, paint);

                    cropToFrameTransform.mapRect(location);

                    result.setLocation(location);
                    mappedRecognitions.add(result);

                    checkSpeedLimit(result.getTitle().trim());
                    if(getNotificationSpeed() && notification.isChecked()) playSound(result.getTitle());

                  }
                }
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                tracker.trackResults(mappedRecognitions, currTimestamp);
                trackingOverlay.postInvalidate();

                computingDetection = false;

                runOnUiThread(
                        () -> {
                          showFrameInfo(previewWidth + "x" + previewHeight);
                          showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                          showInference(lastProcessingTimeMs + "ms");
                        });
              });
    }

    private void checkSpeedLimit(String title){
      switch (title){
          case "Max speed limit 20km/h":
              setSpeedLimit(20);
              break;
          case "Max speed limit 30km/h":
              setSpeedLimit(30);
              break;
          case "Max speed limit 40km/h":
              setSpeedLimit(40);
              break;
          case "Max speed limit 50km/h":
              setSpeedLimit(50);
              break;
          case "Max speed limit 60km/h":
              setSpeedLimit(60);
              break;
          case "Max speed limit 70km/h":
              setSpeedLimit(70);
              break;
          case "Max speed limit 80km/h":
              setSpeedLimit(80);
              break;
          case "Max speed limit 90km/h":
              setSpeedLimit(90);
              break;
          case "Max speed limit 100km/h":
              setSpeedLimit(100);
              break;
          case "Max speed limit 110km/h":
              setSpeedLimit(110);
              break;
          case "Max speed limit 120km/h":
              setSpeedLimit(120);
              break;
          default:
              break;
      }
    }

    @SuppressLint("ResourceType")
    private void playSound(String title) {
        setNotificationSpeed(false);
        switch (title.trim()) {
            case "Priority road ends":
                mediaPlayerHolder.loadMedia(R.raw.priority_road_ends_3);
                break;
            case "Go straight ahead":
                mediaPlayerHolder.loadMedia(R.raw.go_straight_ahead_4);
                break;
            case "Go straight ahead or turn left":
                mediaPlayerHolder.loadMedia(R.raw.go_straight_ahead_or_turn_left_5);
                break;
            case "Go straight ahead or turn right":
                mediaPlayerHolder.loadMedia(R.raw.go_straight_ahead_or_turn_right_6);
                break;
            case "Passing left mandatory":
                mediaPlayerHolder.loadMedia(R.raw.passing_left_mandatory_8);
                break;
            case "Passing right mandatory":
                mediaPlayerHolder.loadMedia(R.raw.passing_right_mandatory_9);
                break;
            case "Max speed limit 20km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_20kmh_10);
                break;
            case "Max speed limit 30km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_30kmh_11);
                break;
            case "Max speed limit 40km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_40kmh_12);
                break;
            case "Max speed limit 50km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_50kmh_13);
                break;
            case "Max speed limit 60km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_60kmh_14);
                break;
            case "Max speed limit 70km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_70kmh_15);
                break;
            case "Max speed limit 80km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_80kmh_16);
                break;
            case "Max speed limit 90km/h":
                mediaPlayerHolder.loadMedia(R.raw.maxsl_90kmh_17);
                break;
            case "Max speed limit 100km/h":
                setSpeedLimit(100);
                mediaPlayerHolder.loadMedia(R.raw.maxsl_100kmh_18);
                break;
            case "Max speed limit 110km/h":
                setSpeedLimit(110);
                mediaPlayerHolder.loadMedia(R.raw.maxsl_110kmh_19);
                break;
            case "Max speed limit 120km/h":
                setSpeedLimit(120);
                mediaPlayerHolder.loadMedia(R.raw.maxsl_120kmh_20);
                break;
            case "Cyclists prohibited":
                mediaPlayerHolder.loadMedia(R.raw.cyclists_prohibited_21);
                break;
            case "No vehicle entry":
                mediaPlayerHolder.loadMedia(R.raw.no_vehicle_entry_22);
                break;
            case "Turning left prohibited":
                mediaPlayerHolder.loadMedia(R.raw.turning_left_prohibited_24);
                break;
            case "Cars prohibited":
                mediaPlayerHolder.loadMedia(R.raw.cars_prohibited_25);
                break;
            case "Motorcycles prohibited":
                mediaPlayerHolder.loadMedia(R.raw.motorcycles_prohibited_26);
                break;
            case "Turning right prohibited":
                mediaPlayerHolder.loadMedia(R.raw.turning_right_prohibited_31);
                break;
            case "Parking and stopping prohibited":
                mediaPlayerHolder.loadMedia(R.raw.parking_and_stopping_prohibited_32);
                break;
            case "U-turn prohibited":
                mediaPlayerHolder.loadMedia(R.raw.u_turn_prohibited_33);
                break;
            case "Mandatory one-way left":
                mediaPlayerHolder.loadMedia(R.raw.mandatory_one_way_34_35_36);
                break;
            case "Mandatory one-way right":
                mediaPlayerHolder.loadMedia(R.raw.mandatory_one_way_34_35_36);
                break;
            case "Mandatory one-way traffic":
                mediaPlayerHolder.loadMedia(R.raw.mandatory_one_way_34_35_36);
                break;
            case "Passing left or right mandatory":
                mediaPlayerHolder.loadMedia(R.raw.passing_left_or_right_37);
                break;
            case "Pedestrians only path":
                mediaPlayerHolder.loadMedia(R.raw.pedestrians_only_path_38);
                break;
            case "Priority road":
                mediaPlayerHolder.loadMedia(R.raw.priority_road_39);
                break;
            case "Stop":
                mediaPlayerHolder.loadMedia(R.raw.stop_43);
                break;
            case "Turning left mandatory":
                mediaPlayerHolder.loadMedia(R.raw.turning_left_mandatory_44);
                break;
            case "Turning right mandatory":
                mediaPlayerHolder.loadMedia(R.raw.turning_right_mandatory_45);
                break;
            case "Give way":
                mediaPlayerHolder.loadMedia(R.raw.give_way_47);
                break;
            case "Crossing for pedestrians":
                mediaPlayerHolder.loadMedia(R.raw.crossing_for_pedestrians_53);
                break;
            case "Children crossing":
                mediaPlayerHolder.loadMedia(R.raw.children_crossing_54);
                break;
            case "Crossroad w/ side roads left and right":
                mediaPlayerHolder.loadMedia(R.raw.crossroad_left_n_right_55);
                break;
            case "Curve left":
                mediaPlayerHolder.loadMedia(R.raw.curve_left_56);
                break;
            case "Curve right":
                mediaPlayerHolder.loadMedia(R.raw.curve_right_57);
                break;
            case "Double curve - first left":
                mediaPlayerHolder.loadMedia(R.raw.double_curve_58_59);
                break;
            case "Double curve - first right":
                mediaPlayerHolder.loadMedia(R.raw.double_curve_58_59);
                break;
            case "Crossroad w/ side road left":
                mediaPlayerHolder.loadMedia(R.raw.crossroad_left_side_road_60);
                break;
            case "Crossroad w/ side road right":
                mediaPlayerHolder.loadMedia(R.raw.crossroad_right_side_road_61);
                break;
            case "Other danger":
                mediaPlayerHolder.loadMedia(R.raw.other_danger_62);
                break;
            case "Pedestrians crossing":
                mediaPlayerHolder.loadMedia(R.raw.pedestrians_crossing_63);
                break;
            case "Railroad crossing":
                mediaPlayerHolder.loadMedia(R.raw.railroad_crossing_64);
                break;
            case "Railroad crossing without barriers":
                mediaPlayerHolder.loadMedia(R.raw.railroad_crossing_without_barriers_65);
                break;
            case "Speed bump":
                mediaPlayerHolder.loadMedia(R.raw.speed_bump_66);
                break;
            case "Road narrows":
                mediaPlayerHolder.loadMedia(R.raw.road_narrows_67);
                break;
            case "Road narrows left":
                mediaPlayerHolder.loadMedia(R.raw.road_narrows_left_68);
                break;
            case "Road narrows right":
                mediaPlayerHolder.loadMedia(R.raw.road_narrows_right_69);
                break;
            case "Roadworks":
                mediaPlayerHolder.loadMedia(R.raw.roadworks_70);
                break;
            case "Roundabout":
                mediaPlayerHolder.loadMedia(R.raw.roundabout_71);
                break;
            case "Slippery road surface":
                mediaPlayerHolder.loadMedia(R.raw.slippery_road_surface_72);
                break;
            case "Traffic light":
                mediaPlayerHolder.loadMedia(R.raw.traffic_light_74);
                break;
            case "Two-way traffic":
                mediaPlayerHolder.loadMedia(R.raw.two_way_traffic_75);
                break;
            case "Uneven road":
                mediaPlayerHolder.loadMedia(R.raw.uneven_road_76);
                break;
            default:
                break;
        }
    }

    //This method gets a recognised box of sign and returns the classified sign.
    private void classify (Recognition result){
        Matrix matrix = new Matrix();
        matrix.postRotate(0);

        Bitmap crop = Bitmap.createBitmap(croppedBitmap,
                (int) result.getLocation().left,
                (int) result.getLocation().top,
                (int) result.getLocation().width(),
                (int) result.getLocation().height(),
                matrix,
                true);

        if (crop != null) {
            try {
                ImageView view = findViewById(R.id.signImg);
                crop = prepareImageForClassification(crop);
                view.setImageBitmap(crop);

                // Initialization
                ImageClassifier.ImageClassifierOptions options =
                        ImageClassifier.ImageClassifierOptions.builder().setMaxResults(1).setScoreThreshold(CLASSIFICATION_THRESHOLD).setNumThreads(4).build();

                ImageClassifier imageClassifier = ImageClassifier.createFromFileAndOptions(
                        getApplicationContext(), MODEL_FILENAME, options);

                // Run inference
                List<Classifications> results2 = imageClassifier.classify(
                        TensorImage.fromBitmap(crop));

                result.setTitle(results2.get(0).getCategories().get(0).getLabel());
                result.setConfidence(results2.get(0).getCategories().get(0).getScore());
            } catch (Exception e) {
              Log.e("SLClassifier error:", e.getMessage(),e);
              result.setTitle("Σήμα");
            }
      }
    }

    @Override
    protected int getLayoutId () {
      return R.layout.tfe_od_camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize () {
      return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {}

    @Override
    protected void setNumThreads (final int numThreads){
      runInBackground(() -> detector.setNumThreads(numThreads));
    }
    public void setMaximumResults(int maximumResults) {
        this.maximumResults = maximumResults;
    }
}