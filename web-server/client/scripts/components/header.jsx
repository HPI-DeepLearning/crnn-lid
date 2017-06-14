import React from "react";
import { Link } from "react-router";
import connectToStores from "alt/utils/connectToStores";
import Component from "./baseComponent.jsx";
import Logo from "../../images/favicon/android-icon-96x96.png";
import ResultStore from "../stores/resultStore.js";


class Header extends Component {

  static getStores() {
    return [ResultStore];
  }

  static getPropsFromStores() {
    return ResultStore.getState();
  }

  getResultLink() {
    if (this.props.audii)
      return (
        <li>
          <Link to="result">
            <i className="material-icons left">dashboard</i>
            Result
          </Link>
        </li>
      );
  }

  render() {

    const resultLink = this.getResultLink();

    return (
      <nav className="blue lighten-1">
        <div className="nav-wrapper">
          <div className="col s12">
            <a href="/" className="brand-logo">
              <img src="/dist/images/favicon/android-icon-96x96.png" width="48px" height="48px"/>
              Language Identification
            </a>
          </div>
        </div>
      </nav>
    );
  }

};

export default connectToStores(Header);
