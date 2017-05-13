import React from "react";
import _ from "lodash";
import C3 from "c3";
import C3CSS from "c3-css";
import Component from "./baseComponent.jsx";

class DropZone extends Component {

  constructor() {
    super();

    this.state = {
      isDragActive : false
    };
  }

  onDragLeave(evt) {

    this.setState({
      isDragActive: false
    });
  }

  onDragOver(evt) {

    evt.preventDefault();
    evt.dataTransfer.dropEffect = 'copy';

    this.setState({
      isDragActive: true
    });
  }

  onDrop(evt) {

    evt.preventDefault();

    this.setState({
      isDragActive: false
    });

    let files;
    if (evt.dataTransfer) {
      files = evt.dataTransfer.files;
    }

    this.props.onDrop(evt, files);

  }

  getDropzoneContent() {

    if (this.state.isDragActive) {

      return (
        <div className="dropzone valign-wrapper">
          <span className="valign center" style={{width : 100 + "%"}}>
            <i className="material-icons medium">queue</i>
          </span>
        </div>
      );
    } else {
      return this.props.children;
    }
  }

  render() {

    const children = this.getDropzoneContent();

    return (
      <div
        onDrop={this.onDrop.bind(this)}
        onDragLeave={this.onDragLeave.bind(this)}
        onDragOver={this.onDragOver.bind(this)}
        >
        {children}
      </div>
    );
  }

};

DropZone.propTypes = {
  onDrop : React.PropTypes.func.isRequired
}


export default DropZone;
