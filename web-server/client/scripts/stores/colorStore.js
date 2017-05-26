import _ from "lodash"
import alt from "../../alt";

class ColorStore {

  constructor() {
    this.colors = ["#2196F3", "#4dd0e1", "#e57373 ", "#4db6ac",  "#fff176", "#7986cb", "#a1887f", "#90a4ae", "#2e7d32", "#4527a0", "#c2185b", "#1565c0", "#e64a19", "#455a64", "#8a716a", "#c2b8b2", "#197bbd", "#125e8a", "#204b57"];
  }

  static hash(text) {
    var hash = 0, i, chr, len;

    if (text.length == 0) return hash;
    for (i = 0, len = text.length; i < len; i++) {
      chr   = text.charCodeAt(i);
      hash  = ((hash << 5) - hash) + chr;
      hash |= 0; // Convert to 32bit integer
    }
    return hash;
  }

  static getColorForLabel(label) {

    const index = Math.abs(this.hash(label)) % this.getState().colors.length
    return this.getState().colors[index];
  }

};

export default alt.createStore(ColorStore, "ColorStore");