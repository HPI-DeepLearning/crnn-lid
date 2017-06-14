import React from "react";
import Router from "./scripts/lib/router";

import FavIcon2 from "./images/favicon/favicon.ico";


import Styles from "./styles/main.less";
import MaterializeCSS from "materialize-css";


Router.run(Handler => React.render(<Handler />, document.body));