import React from "react";
import Component from "./baseComponent.jsx";
import RecordRTC from "recordRTC";
import AudioActions from "../actions/audioActions"

class AudioCapture extends Component {

  constructor(props) {

    super(props);
    this.state = {
      isRecording : false,
      audioURL : null,
      audioOptions : {
        type: "audio",
        interval: 10 * 1000 //ms
      }
    };

    navigator.getUserMedia = navigator.getUserMedia ||
                             navigator.webkitGetUserMedia ||
                             navigator.mozGetUserMedia;

    navigator.getUserMedia(
      {audio : true},
      (mediaStream) => {

        this.mediaStream = mediaStream;
        //this.refs.daVideo.getDOMNode().src = window.URL.createObjectURL(mediaStream);
      },
      (error) => console.error(error)
    );

  }

  handleClick() {

    if (this.state.isRecording == false) {

      this.recordRTC = RecordRTC(this.mediaStream, this.state.audioOptions)
      this.recordRTC.startRecording();
      this.updateState({isRecording : {$set : true}})

    } else {

      this.recordRTC.stopRecording((audioURL) => {

        this.updateState({audioURL : {$set : audioURL}})
        this.updateState({isRecording : {$set : false}})

        const recordedBlob = this.recordRTC.getBlob();
        AudioActions.uploadAudio(recordedBlob);

        // Stop audio stream?
        //this.mediaStream.stop();
      });
    }
  }

  render() {

    const buttonText = this.state.isRecording ? "Stop Recording & Submit" : "Start Recording";

    return (
      <div className="center-align">
        <div>
          <a className="waves-effect waves-light btn" onClick={this.handleClick.bind(this)}> {buttonText} <i className="material-icons right">mic</i></a>
        </div>
      </div>
    )

  }

};


export default AudioCapture;



