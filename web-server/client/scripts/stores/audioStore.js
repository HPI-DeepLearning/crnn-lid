import _ from "lodash"
import alt from '../../alt';
import AudioActions from '../actions/audioActions';
import API from "../lib/api"

import AudioThumbnail1 from "../../images/examples/audio.png"


class AudioStore {

  constructor() {
    this.bindActions(AudioActions);

    this.isUploading = false;
    this.isInvalidFile = false;
    this.exampleAudio = [
      {
        id : 1,
        thumbnail : "/dist/images/examples/audio.png",
        title : "CNN",
        language : "English",
      },
      {
        id : 2,
        thumbnail : "/dist/images/examples/audio.png",
        title : "Dubsmash 2",
        language : "German",
      },
      {
        id : 3,
        thumbnail : "/dist/images/examples/audio.png",
        title : "Voxforge",
        language : "English",
      }
    ];
  }

  static getExampleAudio() {
    return this.getState().exampleAudio;
  }

  onUploadAudio(audioFile) {

    const payload = {
      audio : audioFile
    };

    API.postAudio(payload)
    this.isUploading = true;
  }

  onUseExample(audioId) {

    API.getPredictionForExample(audioId);
    this.isUploading = true;
  }

  onReceivePrediction() {
    this.isUploading = false;
    this.isInvalidFile = false;
  }

  onReceiveUploadError() {
    this.isUploading = false;
    this.isInvalidFile = true;
  }

};

export default alt.createStore(AudioStore, "AudioStore");