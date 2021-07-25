package thesis.rttsd_thesis;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;

import static thesis.rttsd_thesis.SpeedLimitClassifier.INPUT_IMG_SIZE_HEIGHT;
import static thesis.rttsd_thesis.SpeedLimitClassifier.INPUT_IMG_SIZE_WIDTH;


public class ImageUtils {

    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        Paint paint = new Paint();
        Bitmap finalBitmap = Bitmap.createScaledBitmap(
                bitmap,
                INPUT_IMG_SIZE_WIDTH,
                INPUT_IMG_SIZE_HEIGHT,
                true);
        Canvas canvas = new Canvas(finalBitmap);
        canvas.drawBitmap(finalBitmap, 0, 0, paint);
        return finalBitmap;
    }

}
