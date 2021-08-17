package thesis.rttsd_thesis;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Paint;


public class ImageUtils {

    public static final int INPUT_IMG_SIZE_WIDTH = 64;
    public static final int INPUT_IMG_SIZE_HEIGHT = 64;

    public static Bitmap prepareImageForClassification(Bitmap bitmap) {
        Paint paint = new Paint();
        Bitmap finalBitmap = Bitmap.createScaledBitmap(
                bitmap,
                INPUT_IMG_SIZE_WIDTH,
                INPUT_IMG_SIZE_HEIGHT,
                false);
        Canvas canvas = new Canvas(finalBitmap);
        canvas.drawBitmap(finalBitmap, 0, 0, paint);
        return finalBitmap;
    }

}
