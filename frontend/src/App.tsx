import React, { useState, lazy, Suspense, useEffect } from "react";
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

import DashboardView from "./DashboardView";
import {
  authenticationService,
  currentJwtStringSubject,
  DecodedJwt,
  getDescriptionOfUserType,
  isFullDecodedJwt,
  LoginButton,
} from "./services/authentication.service";
import NewFold from "./NewFold/NewFold";
import UIkit from "uikit";
import SudoPage from "./SudoPage/SudoPage";
import About from "./About/About";
import TagView from "./TagView";
import { GoogleLogin, GoogleOAuthProvider } from "@react-oauth/google";
import GoogleLoginButton from "./GoogleLogin";
import { FoldyMascot } from "./util/foldyMascot";

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

function RoutedApp() {
  const [token, setToken] = useState<string>(
    authenticationService.currentJwtStringValue
  );
  const { decodedToken, isExpired } = useJwt(token);
  let [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();

  var fullDecodedToken: DecodedJwt | null = null;
  if (isFullDecodedJwt(decodedToken)) {
    fullDecodedToken = decodedToken;

    const isNewUser = searchParams.get("new_user");
    if (isNewUser) {
      const newSearchParams = new URLSearchParams(searchParams);
      newSearchParams.delete("new_user");
      setSearchParams(newSearchParams);

      UIkit.modal
        .alert(
          `<p>Welcome new user!</p><p>${getDescriptionOfUserType(
            fullDecodedToken.user_claims.type || ""
          )}</p><p>Check out the <a href="/about">About</a> page for information about the service and updates as we make improvements.</p>`
        )
        .then(() => {
          navigate("/about");
        });
    }
  }

  useEffect(() => {
    const jwtString = searchParams.get("access_token");
    if (jwtString) {
      // Remove access_token from the URL.
      console.log(`Deleting access_token ${searchParams}`);
      const newSearchParams = new URLSearchParams(searchParams);
      newSearchParams.delete("access_token");
      console.log(`Deleted access_token ${newSearchParams}`);

      setSearchParams(newSearchParams);

      localStorage.setItem("currentJwtString", jwtString);
      currentJwtStringSubject.next(jwtString);
      setToken(jwtString);
    } else if (fullDecodedToken && isExpired) {
      UIkit.notification("Login expired.");
    }
  }, [searchParams, fullDecodedToken, isExpired]);

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

  const foldyTitle = (
    <span>
      {process.env.REACT_APP_INSTITUTION} Foldy
      <sub>
        <sub>
          {fullDecodedToken?.user_claims.type === "viewer" ? "View Only" : null}
          {fullDecodedToken?.user_claims.type === "editor"
            ? "Edit Access"
            : null}
          {fullDecodedToken?.user_claims.type === "admin"
            ? "Admin Access"
            : null}
        </sub>
      </sub>
    </span>
  );

  const foldyWelcomeText = `Welcome to ${process.env.REACT_APP_INSTITUTION} Foldy! Login with an ${process.env.REACT_APP_INSTITUTION} account for edit access, or any other account to view public structures.`;

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
          {foldyTitle}
        </a>
        <a href="/" className="uk-navbar-item" style={{ color: "#fff" }}>
          Dashboard
        </a>
        {fullDecodedToken?.user_claims.type === "admin" ? (
          <a
            href={`${process.env.REACT_APP_BACKEND_URL}/rq/`}
            className="uk-navbar-item"
            style={{ color: "#fff" }}
          >
            RQ
          </a>
        ) : null}
        {fullDecodedToken?.user_claims.type === "admin" ? (
          <a
            href={`${process.env.REACT_APP_BACKEND_URL}/admin/`}
            className="uk-navbar-item"
            style={{ color: "#fff" }}
          >
            DBs
          </a>
        ) : null}
        {fullDecodedToken?.user_claims.type === "admin" ? (
          <a
            href="/sudopage"
            className="uk-navbar-item"
            style={{ color: "#fff" }}
          >
            Sudo Page
          </a>
        ) : null}
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
        {/* <GoogleLogin
          onSuccess={(credentialResponse) => {
            console.log(credentialResponse);
          }}
          onError={() => {
            console.log("Login Failed");
          }}
        /> */}
      </div>

      {fullDecodedToken && !isExpired ? null : (
        <FoldyMascot text={foldyWelcomeText} moveTextAbove={false} />
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
          {foldyTitle}
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
        <FoldyMascot text={foldyWelcomeText} moveTextAbove={true} />
      )}
    </nav>
  );

  return (
    <div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      {/* <AcceptJWTAndRedirect setToken={setToken} decodedToken={fullDecodedToken} isExpired={isExpired} /> */}
      <div className="uk-visible@m">{desktop_navbar}</div>
      <div className="uk-hidden@m">{mobile_navbar}</div>

      <CheckForErrorQueryString
        setErrorText={setErrorText}
      ></CheckForErrorQueryString>
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
            {fullDecodedToken?.user_claims.type === "admin" ? (
              <li className="uk-parent">
                <a href={`${process.env.REACT_APP_BACKEND_URL}/rq/`}>RQ</a>
              </li>
            ) : null}
            {fullDecodedToken?.user_claims.type === "admin" ? (
              <li className="uk-parent">
                <a href={`${process.env.REACT_APP_BACKEND_URL}/admin/`}>DBs</a>
              </li>
            ) : null}
            {fullDecodedToken?.user_claims.type === "admin" ? (
              <li className="uk-parent">
                <a href="/sudopage">Sudo Page</a>
              </li>
            ) : null}
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

      <div
        className="uk-width-5-6@xl uk-container-center uk-align-center"
        style={{
          display: "flex",
          flexDirection: "column",
          flexGrow: 1,
          overflow: "hidden",
          marginTop: "0px",
          marginBottom: "0px",
        }}
      >
        <Routes>
          <Route
            path="/fold/:foldId"
            element={
              <Suspense fallback={renderLoader()}>
                <AvatarFoldView
                  setErrorText={setErrorText}
                  userType={
                    fullDecodedToken ? fullDecodedToken.user_claims.type : null
                  }
                />
              </Suspense>
            }
          />
          <Route
            path="/tag/:tagStringParam"
            element={<TagView setErrorText={setErrorText} />}
          />
          <Route
            path="/newFold"
            element={
              <NewFold
                setErrorText={setErrorText}
                userType={
                  fullDecodedToken ? fullDecodedToken.user_claims.type : null
                }
              />
            }
          />
          <Route
            path="/sudopage"
            element={<SudoPage setErrorText={setErrorText} />}
          />
          <Route
            path="/about"
            element={
              <About
                userType={
                  fullDecodedToken ? fullDecodedToken.user_claims.type : null
                }
              />
            }
          />
          <Route
            path="/"
            element={
              <DashboardView
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
  if (!process.env.REACT_APP_INSTITUTION) {
    console.error("REACT_APP_INSTITUTION is unset.");
  }
  if (!process.env.REACT_APP_BACKEND_URL) {
    console.error("REACT_APP_BACKEND_URL is unset.");
  }
  if (!process.env.REACT_APP_GOOGLE_CLIENT_ID) {
    console.error("REACT_APP_GOOGLE_CLIENT_ID is unset.");
  }
  return (
    <GoogleOAuthProvider
      clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID || ""}
    >
      <BrowserRouter>
        <RoutedApp />
      </BrowserRouter>
    </GoogleOAuthProvider>
  );
}

export default App;
