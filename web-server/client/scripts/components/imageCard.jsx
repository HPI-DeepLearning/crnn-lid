import React from "react";
import Component from "./baseComponent.jsx";

class ImageCard extends Component {


  getCardActions() {

    if (this.props.actions.length > 0) {
      return (
        <div className="card-action">
          {this.props.actions}
        </div>
      )
    } else {
      return <span/>
    }
  }

  render() {

    const cardActions = this.getCardActions();

    return (
      <div className="card">
        <div className="card-image">
          <img src={this.props.image}/>
        </div>
        <div className="card-content">
          <h3 className="card-title">{this.props.title}</h3>
          <p>{this.props.content}</p>
        </div>
        {cardActions}
      </div>
    );
  }

};

ImageCard.propTypes = {
  title : React.PropTypes.string.isRequired,
  image : React.PropTypes.string.isRequired,
  content : React.PropTypes.string,
  actions : React.PropTypes.array
}

ImageCard.defaultProps = {
  content : "",
  actions : []
}

export default ImageCard;



