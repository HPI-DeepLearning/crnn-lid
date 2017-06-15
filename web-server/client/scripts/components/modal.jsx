import React from "react";
import Component from "./baseComponent.jsx";
import Spinner from "./spinner.jsx";

class Modal extends Component {

  render() {
    const modalStyle = {
      "zIndex": 1003,
      "display": "block",
      "opacity": 1,
      "transform": "scaleX(1)",
      "top": "25%",
    };
    const overlayStyle = {
      "zIndex": 1002,
      "display": "block",
      "opacity": 0.5,
      "position": "fixed",
      "top": -100,
      "left": 0,
      "bottom": 0,
      "right": 0,
      "height": "125%",
      "width": "100%",
      "background": "#000",
    };

    return (
      <div>
        <div className="modal-overlay" id="materialize-modal-overlay-3" style={overlayStyle}></div>
        <div className="modal open" style={modalStyle}>
          <div className="modal-content center-align">
            <Spinner size="big"/>
            <p>Calculating predicition. Please wait.</p>
            <p>(this may take a few seconds)</p>
          </div>
        </div>
      </div>
    );
  }
};

export default Modal;


