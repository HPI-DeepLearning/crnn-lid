import React from "react";
import Component from "./baseComponent.jsx";

class ProgressBar extends Component {

  render() {

    const styles = {
      width : this.props.progress + "%"
    }

    return (
      <div className="progress">
        <div className="determinate" style={styles}></div>
      </div>
    );
  }

};

ProgressBar.propTypes = {
  progress : React.PropTypes.number,
}

ProgressBar.defaultProps = {
  progress: 0
}

export default ProgressBar;
