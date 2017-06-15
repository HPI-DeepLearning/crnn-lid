import React from "react";
import connectToStores from "alt/utils/connectToStores";
import LineChart from "../components/linechart.jsx";
import BarChart from "../components/barchart.jsx";
import Component from "../components/baseComponent.jsx";
import ResultStore from "../stores/resultStore"
import ColorStore from "../stores/colorStore"
import EnFlag from "../../images/gb.png"
import DeFlag from "../../images/de.png"
import FrFlag from "../../images/fr.png"
import EsFlag from "../../images/es.png"
import CnFlag from "../../images/cn.png"
import RuFlag from "../../images/ru.png"

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

    return {
      columns : columns,
      colors : colors
    }
  }

  getFlag() {
    const languageMap = {
      "English": "gb",
      "German": "de",
      "French": "fr",
      "Spanish": "es",
      "Chinese": "cn",
      "Russian": "ru",
    }
    const predictions = ResultStore.getPredictions();
    const highestPrediction = Object.keys(predictions).reduce(function(a, b){ return predictions[a] > predictions[b] ? a : b });

    return `/dist/images/${languageMap[highestPrediction]}.png`
  }

  getLineCharData() {

    const timesteps = ResultStore.getTimesteps();
    const columns = _.map(timesteps, (value, key) => [key].concat(value))
    columns.push(
      ["x"].concat(_.range(0, columns[0].length - 1 ).map((i) => i * 10))
    )

    const colors = _.mapValues(timesteps, (value, key) => ColorStore.getColorForLabel(key))

    return {
      x: "x",
      columns : columns,
      colors : colors
    }
  }

  render() {

    return (
      <div className="result-page">
        <div className="row">
          <div className="col s6 m6">
            <div className="card-panel center-align prediction-panel">
              <span className="card-title">Majority Voting Prediction</span>
              <BarChart data={this.getBarChartData()} />
            </div>
          </div>
          <div className="col s6 m6">
            <div className="card-panel center-align prediction-panel">
              <div className="row">
                <div className="col s3 m3">
                  <img className="flag" src={this.getFlag()} />
                </div>
                <div className="col s9 m9">
                  <audio
                    src={this.props.audio.url}
                    className="valign"
                    controls
                    />
                </div>
              </div>
              <table className="striped">
                <tbody></tbody>
                <tr>
                  <td>Channels</td><td>{this.props.metadata.channels}</td>
                </tr>
                <tr>
                  <td>Duration</td><td>{parseInt(this.props.metadata.duration)}s</td>
                </tr>
                <tr>
                  <td>Bit Rate</td><td>{this.props.metadata.bitrate}</td>
                </tr>
                <tr>
                  <td>Sample</td><td>{parseInt(this.props.metadata.sample_rate) / 1000}kHz</td>
                </tr>
                <tr>
                  <td>Encoding</td><td>{this.props.metadata.encoding}</td>
                </tr>
              </table>
            </div>
          </div>
        </div>
        <div className="row">
          <div className="col s12 m12">
            <div className="card-panel center-align">
              <span className="card-title">Predicitions per 10s Time Steps</span>
              <LineChart data={this.getLineCharData()}/>
            </div>
          </div>
        </div>
      </div>
    );
  }

};

export default connectToStores(Result);

