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
        thumbnail : "/dist/images/gb.png",
        title : "CNN",
        language : "English",
      },
      {
        id : 2,
        thumbnail : "/dist/images/de.png",
        title : "Deutsche Welle",
        language : "German",
      },
      {
        id : 3,
        thumbnail : "/dist/images/fr.png",
        title : "France24",
        language : "French",
      },
      {
        id : 4,
        thumbnail : "/dist/images/es.png",
        title : "Antena3noticias",
        language : "Spanish",
      },
      {
        id : 5,
        thumbnail : "/dist/images/cn.png",
        title : "VOAChina",
        language : "Mandarin Chinese",
      },
      {
        id : 6,
        thumbnail : "/dist/images/ru.png",
        title : "Russia24TV",
        language : "Russian",
      },
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