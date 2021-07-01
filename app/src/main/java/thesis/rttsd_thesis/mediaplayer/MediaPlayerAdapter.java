package thesis.rttsd_thesis.mediaplayer;

import androidx.annotation.IdRes;

public interface MediaPlayerAdapter {
    void loadMedia(@IdRes int resId);
    void reset();
}
