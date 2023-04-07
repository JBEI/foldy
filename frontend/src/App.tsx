import React, { useState, lazy, Suspense } from "react";
import "./App.scss";
import "react-tiny-fab/dist/styles.css";
import {
  Route,
  Routes,
  BrowserRouter,
  useLocation,
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import { useJwt } from "react-jwt";

import FoldsView from "./FoldsView";
import {
  authenticationService,
  currentJwtStringSubject,
  DecodedJwt,
  isFullDecodedJwt,
  LoginButton,
  redirectToLogin,
} from "./services/authentication.service";
import NewFold from "./NewFold/NewFold";
import UIkit from "uikit";
import SudoPage from "./SudoPage/SudoPage";
import About from "./About/About";
import TagView from "./TagView";
import { Foldy } from "./Util";

const AvatarFoldView = lazy(() => import("./FoldView/FoldView"));

function CheckForErrorQueryString(props: {
  setErrorText: (a: string | null) => void;
}) {
  const location = useLocation();
  const navigate = useNavigate();
  let params = new URLSearchParams(location.search);

  const queryParamErrorText = params.get("error_message");
  if (!queryParamErrorText) {
    return <div></div>;
  }

  props.setErrorText(queryParamErrorText);

  params.delete("error_message");
  navigate({
    pathname: location.pathname,
    search: params.toString(),
  });

  return <div></div>;
}

function NewUserWelcomePage() {
  const location = useLocation();
  const navigate = useNavigate();
  let params = new URLSearchParams(location.search);

  const isNewUser = params.get("new_user");
  if (!isNewUser) {
    return <div></div>;
  }

  params.delete("new_user");
  navigate({
    pathname: location.pathname,
    search: params.toString(),
  });

  UIkit.modal
    .alert(
      '<p>Welcome new user! Check out the <a href="/about">About</a> page for information about the service and updates as we make improvements.</p>'
    )
    .then(() => {
      navigate("/about");
    });
  return <div></div>;
}

function RoutedApp() {
  const [token, setToken] = useState<string>(
    authenticationService.currentJwtStringValue
  );
  const { decodedToken, isExpired } = useJwt(token);
  // const location = useLocation();
  // const navigate = useNavigate()
  let [searchParams, setSearchParams] = useSearchParams();

  var fullDecodedToken: DecodedJwt | null = null;
  if (isFullDecodedJwt(decodedToken)) {
    fullDecodedToken = decodedToken;
  }

  const setErrorText = (error_text: string | null): void => {
    UIkit.notification(error_text || "Danger!", { status: "danger" });
  };

  const renderLoader = () => {
    return (
      <div className="uk-text-center">
        <div uk-spinner="ratio: 4"></div>
      </div>
    );
  };

  // JBEI orange: CF4520
  // JBEI red: CF4420
  // const desktop_navbar = <nav className="uk-navbar" style={{background: 'linear-gradient(to left, #CF4420, #CF4520)'}}>
  const desktop_navbar = (
    <nav
      className="uk-navbar"
      style={{ background: "linear-gradient(to left, #28a5f5, #1e87f0)" }}
    >
      <div className="uk-navbar-left">
        <a
          href="/"
          className="uk-navbar-item uk-logo uk-margin-left"
          style={{ color: "#fff" }}
        >
          {process.env.REACT_APP_INSTITUTION} Foldy
        </a>
        <a href="/" className="uk-navbar-item" style={{ color: "#fff" }}>
          Dashboard
        </a>
        <a href="/about" className="uk-navbar-item" style={{ color: "#fff" }}>
          About
        </a>
      </div>

      <div
        className="uk-navbar-right uk-navbar-item uk-active uk-margin-small-right"
        style={{ color: "#fff" }}
      >
        <LoginButton
          decodedToken={fullDecodedToken}
          setToken={setToken}
          isExpired={isExpired}
        />
      </div>

      {fullDecodedToken && !isExpired ? null : (
        <Foldy
          text={`Welcome to ${process.env.REACT_APP_INSTITUTION} Foldy!`}
          moveTextAbove={false}
        />
      )}
    </nav>
  );

  const mobile_navbar = (
    <nav
      className="uk-navbar"
      style={{
        background: "linear-gradient(to left, #28a5f5, #1e87f0)",
        zIndex: 100,
      }}
    >
      <div className="uk-navbar-left">
        <a
          href="/"
          className="uk-navbar-item uk-logo uk-margin-small-left"
          style={{ color: "#fff" }}
        >
          {process.env.REACT_APP_INSTITUTION} Foldy
        </a>
      </div>

      <div className="uk-navbar-right uk-navbar-item uk-active uk-margin-small-right">
        <button
          className="uk-navbar-toggle"
          uk-navbar-toggle-icon={1}
          style={{ color: "#fff" }}
          uk-toggle="target: #off-canvas-navbar"
        ></button>
      </div>

      {fullDecodedToken && !isExpired ? null : (
        <Foldy
          text={`Welcome to ${process.env.REACT_APP_INSTITUTION} Foldy!`}
          moveTextAbove={true}
        />
      )}
    </nav>
  );

  // let params = new URLSearchParams(location.search);

  const jwtString = searchParams.get("access_token");
  if (jwtString) {
    // Remove access_token from the URL.
    searchParams.delete("access_token");

    setSearchParams({
      search: searchParams.toString(),
    });

    localStorage.setItem("currentJwtString", jwtString);
    currentJwtStringSubject.next(jwtString);
    setToken(jwtString);
  } else if (fullDecodedToken && isExpired) {
    redirectToLogin();
    return <div>Redirecting to refresh token...</div>;
  }

  return (
    <div>
      {/* <AcceptJWTAndRedirect setToken={setToken} decodedToken={fullDecodedToken} isExpired={isExpired} /> */}
      <div className="uk-visible@m">{desktop_navbar}</div>
      <div className="uk-hidden@m">{mobile_navbar}</div>

      <CheckForErrorQueryString
        setErrorText={setErrorText}
      ></CheckForErrorQueryString>
      <NewUserWelcomePage></NewUserWelcomePage>
      <div id="off-canvas-navbar" uk-offcanvas={1}>
        <div className="uk-offcanvas-bar uk-flex uk-flex-column">
          <button
            className="uk-offcanvas-close"
            type="button"
            uk-close={1}
          ></button>

          <h3>{process.env.REACT_APP_INSTITUTION} Foldy</h3>
          <p>
            {process.env.REACT_APP_INSTITUTION} Foldy is a web app for
            predicting and using protein structures based on AlphaFold.
          </p>

          <ul className="uk-nav uk-nav-primary uk-nav-center uk-margin-auto-vertical">
            <li className="uk-active">
              <a href="/">Dashboard</a>
            </li>
            <li className="uk-parent">
              <a href="/about">About</a>
            </li>
            <li>
              <LoginButton
                setToken={setToken}
                decodedToken={fullDecodedToken}
                isExpired={isExpired}
              />
            </li>
          </ul>
        </div>
      </div>

      <div className="uk-width-5-6@xl uk-container-center uk-align-center">
        <Routes>
          <Route
            path="/fold/:foldId"
            element={
              <Suspense fallback={renderLoader()}>
                <AvatarFoldView setErrorText={setErrorText} />
              </Suspense>
            }
          />
          <Route
            path="/tag/:tagStringParam"
            element={<TagView setErrorText={setErrorText} />}
          />
          <Route
            path="/newFold"
            element={<NewFold setErrorText={setErrorText} />}
          />
          <Route
            path="/sudopage"
            element={<SudoPage setErrorText={setErrorText} />}
          />
          <Route path="/about" element={<About />} />
          <Route
            path="/"
            element={
              <FoldsView
                setErrorText={setErrorText}
                decodedToken={fullDecodedToken}
              />
            }
          />
        </Routes>
      </div>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <RoutedApp />
    </BrowserRouter>
  );
}

export default App;
