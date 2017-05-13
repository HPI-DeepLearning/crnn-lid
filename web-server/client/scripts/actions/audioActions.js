import alt from "../../alt";

class AudioActions {
  constructor() {
    this.generateActions(
      "receivePrediction",
      "receiveUploadError",
      "uploadAudio",
      "useExample"
    )
  }
}

export default alt.createActions(AudioActions)