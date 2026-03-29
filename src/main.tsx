import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);

const hideBootSplash = () => {
  const splash = document.getElementById("boot-splash");
  if (!splash) {
    return;
  }

  window.setTimeout(() => {
    splash.classList.add("hidden");
    window.setTimeout(() => splash.remove(), 500);
  }, 380);
};

hideBootSplash();
