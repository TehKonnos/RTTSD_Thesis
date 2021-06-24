package thesis.rttsd_thesis.model.bus.model;


import thesis.rttsd_thesis.model.bus.EventModel;
import thesis.rttsd_thesis.model.entity.Data;

public class EventUpdateLocation implements EventModel {

    private Data data;

    public EventUpdateLocation(Data data) {
        this.data = data;
    }

    public Data getData() {
        return data;
    }

}
