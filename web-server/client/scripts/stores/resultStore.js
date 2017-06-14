import _ from "lodash"
import alt from "../../alt";
import VideoActions from "../actions/audioActions";
import RouterActions from "../actions/routerActions";

class ResultStore {

  constructor() {
    this.bindActions(VideoActions);

    this.audio = null;
    this.predictions = null;
  }

  onReceivePrediction(response) {
    this.audio = response.audio;
    this.predictions = response.predictions;
    this.metadata = response.metadata;
    this.timesteps = response.timesteps;

    RouterActions.transition("result")

  }

  static getPredictions() {
    const predictions = this.getState().predictions;
    if (predictions) {
      return predictions;
    } else {
      return null;
    }
  }

  static getTimesteps() {
    const timesteps = this.getState().timesteps;
    if (timesteps) {
      return timesteps;
    } else {
      return null;
    }
  }

};

export default alt.createStore(ResultStore, "ResultStore");