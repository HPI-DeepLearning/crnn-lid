import alt from "../../alt";
import Router from "../lib/router";

class RouterActions {

  transition(nextPage) {

    Router.transitionTo("result", {stepUrl: nextPage});

  }
}

export default alt.createActions(RouterActions)