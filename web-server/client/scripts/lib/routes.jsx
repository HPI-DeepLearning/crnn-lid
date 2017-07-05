import React from "react";
import Router from 'react-router';
var { Route, DefaultRoute, NotFoundRoute } = Router;

import App from '../pages/app.jsx';
import Home from '../pages/home.jsx';
import Result from '../pages/result.jsx';
import NotFound from '../pages/notFound.jsx';

var routes = (
  <Route name="app" path="/" handler={ App }>
    <Route name="result" handler={ Result } />
    <Route name="home" handler={ Home } />
    <DefaultRoute handler={ Home } />
    <NotFoundRoute handler={ NotFound } />
  </Route>
);

module.exports = routes;