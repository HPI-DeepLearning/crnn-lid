import React from "react";
import Component from "../components/baseComponent.jsx";
import { RouteHandler } from 'react-router';
import Header from '../components/header.jsx'

class App extends Component {

  render() {

    return (
      <div>
        <Header />
        <div className="container">
          <RouteHandler/>
        </div>
      </div>
    );
  }

};

export default App;
