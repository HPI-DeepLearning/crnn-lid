import React from "react";
import connectToStores from "alt/utils/connectToStores";
import LineChart from "../components/linechart.jsx";
import BarChart from "../components/barchart.jsx";
import Component from "../components/baseComponent.jsx";
import ResultStore from "../stores/resultStore"
import ColorStore from "../stores/colorStore"

class Result extends Component {

  static getStores() {
    return [ResultStore];
  }

  static getPropsFromStores() {
    return ResultStore.getState();
  }

  onDataClicked(datum) {

    const audio = this.props.audio

    // Calculate audio timepoint from frame number (datum.x)
    const timepoint = datum.x / audio.framerate
    this.refs.audio.getDOMNode().currentTime = timepoint;
  }

  getBarChartData() {

    const predictions = ResultStore.getPredictions();
    const columns = _.chain(predictions)
      .map((value, key) => {
        return [key, value];
      })
      .sortBy(column => column[1])
      .reverse()
      .slice(0, 4)
      .value();

    const colors = _.mapValues(predictions, (value, key) => ColorStore.getColorForLabel(key))
    debugger

    return {
      columns : columns,
      colors : colors
    }

  }

  render() {

    return (
      <div className="result-page">
        <div className="row">
          <div className="col s12 m12">
            <div className="card-panel center-align prediction-panel">
              <BarChart data={this.getBarChartData()} />
              <audio
                src={this.props.audio.url}
                className="valign"
                controls
                />
            </div>
          </div>
        </div>
      </div>
    );
  }

};

export default connectToStores(Result);

