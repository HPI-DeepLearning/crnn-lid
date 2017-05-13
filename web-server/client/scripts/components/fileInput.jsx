import React from "react";
import Component from "./baseComponent.jsx";
import Dropzone from "./dropzone.jsx";

class FileInput extends Component {

  constructor(props) {
    super(props);
    this.state = {
      fileNames : "",
      files : []
    };
  }

  handleChange(evt, droppedFiles) {

    const files = droppedFiles ? droppedFiles : this.refs.fileInput.getDOMNode().files;
    let fileNames = [];

    for (var i=0; i < files.length; i++) {
      fileNames.push(files[i].name);
    }

    this.setState({
      fileNames : fileNames.join(", "),
      files : files
    });
  }

  handleClick(evt) {
    this.refs.fileInput.getDOMNode().click();
  }

  getFiles() {
    return this.state.files;
  }

  render() {

    return (
      <Dropzone onDrop={this.handleChange.bind(this)}>
        <div className="file-field input-field">
          <div className="btn">
            <span>File</span>
            <input
              type="file"
              name="audio"
              ref="fileInput"
              accept={this.props.fileFilter}
              onChange={this.handleChange.bind(this)} />
          </div>
          <div className="file-path-wrapper">
            <input
              readOnly
              value={this.state.fileNames}
              className="file-path validate"
              type="text"
              placeholder={this.props.placeholder}
              onClick={this.handleClick.bind(this)}
              ref="filePath"
               />
          </div>
        </div>
      </Dropzone>
    );
  }

};

FileInput.propTypes = {
  placeholder : React.PropTypes.string,
  fileFilter : React.PropTypes.string,
}

FileInput.defaultProps = {
  placeholder: "",
  fileFilter: ""
}

module.exports = FileInput;
