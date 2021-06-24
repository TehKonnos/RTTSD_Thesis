package thesis.rttsd_thesis.model.bus.model;


import thesis.rttsd_thesis.model.bus.EventModel;
import thesis.rttsd_thesis.model.entity.GpsStatusEntity;

public class EventUpdateStatus implements EventModel {

    private GpsStatusEntity status;

    public EventUpdateStatus(GpsStatusEntity status) {
        this.status = status;
    }

    public GpsStatusEntity getStatus() {
        return status;
    }

}

