import "whatwg-fetch";
import _ from "lodash";

const activeRequests = {};
const nullPromise = { then: _.noop, catch: _.noop };


function processStatus(response) {
  if (response.status >= 200 && response.status < 300) {
    return Promise.resolve(response);
  } else {
    return Promise.reject(response);
  }
}

function getJson(response) {
  return response.text()
      .then((text) => {
        // Handle empty response
        if (!_.isEmpty(text)) {
          return Promise.resolve(JSON.parse(text));
        }
        return Promise.resolve(null);
      });
}

function handleError(error) {
  console.error(error);

  error.text().then( (text) => {
    try {
      const errorJson = JSON.parse(text);
      console.error(errorJson.errors);
    } catch (e) {
      // no worries
    }
  });

  return Promise.reject(error);
}

function extendOptions(options, extensions) {
  options = options || {};
  return _.merge(options, extensions);
}

const FetchUtils = {
  fetchJson: function(url, _options) {
    let contentType;
    let options = _options || {};

    if (options.type === "formdata") {
      options.body = FetchUtils.getFormData(options.body);
    } else if (options.type === "json") {
      options.body = JSON.stringify(options.body);
      contentType = "application/json";
    }

    options = extendOptions(options, {
      headers: {
        "Accept": "application/json",
        "Content-Type":  contentType
      }
    });

    return this.triggerRequest(url, options);
  },

  triggerRequest: function(url, options) {
    const requestKey = JSON.stringify([url, options]);
    const isGetRequest = options.method === undefined ||
      options.method.toUpperCase() === "GET";

    if (isGetRequest && activeRequests[requestKey]) {
      return nullPromise;
    } else {
      activeRequests[requestKey] = true;
      return fetch(url, options)
        .then((response) => {
          delete activeRequests[requestKey];
          return response;
        }, (error) => {
          delete activeRequests[requestKey];
          return Promise.reject(error);
        }).then(processStatus)
        .then(getJson)
        .catch(handleError);
    }
  },

  getFormData: function(json) {
    const formData = new FormData();
    for (var key in json) {

      const value = json[key];

      // Rename blobs to something meaningful
      if (value.constructor.name == "Blob") {
        const filename = "recording.wav";
        formData.append(key, value, filename);

      } else {
        formData.append(key, value);
      }

    }
    return formData;
  }
};

module.exports = FetchUtils;
