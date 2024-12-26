import React, { useEffect } from "react";
import { BehaviorSubject } from "rxjs";
import { map } from "rxjs/operators";

// Based on:
// https://jasonwatmore.com/post/2019/04/06/react-jwt-authentication-tutorial-example

export interface DecodedJwt {
  user_claims: {
    email: string;
    name: string;
    type: string | null;
  };
}

export function getDescriptionOfUserType(userType: string) {
  if (userType === "admin") {
    return `Your account has "admin" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy. You have edit access, plus access to debug tools available in the toolbar. The RQ page shows the status of the Redis Queue, which manages jobs. The DBs page allows direct edit access to all underlying databases, built on Flask-Admin. The Sudo Page contains some convenient buttons for manipulating folds.`;
  } else if (userType === "editor") {
    return `Your account has "editor" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy which means you can view any structure or submit your own. Check out the instructions in the About page for details.`;
  } else if (userType === "viewer") {
    return `Your account has "viewer" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy which means you can view any public folds and associated data, but cannot submit your own compute jobs. See the Foldy manuscript and codebase to set up a Foldy instance at your institution!`;
  } else {
    console.error(`Unknown user type ${userType}`);
    return "";
  }
}

export function isFullDecodedJwt(obj: any): obj is DecodedJwt {
  return (
    obj &&
    obj.user_claims &&
    typeof obj.user_claims.email === "string" &&
    typeof obj.user_claims.name === "string"
  );
}

export const currentJwtStringSubject: BehaviorSubject<string | null> = (() => {
  return new BehaviorSubject(localStorage.getItem("currentJwtString"));
})();

export const authenticationService = {
  logout,
  currentJwtString: currentJwtStringSubject.asObservable().pipe(
    map((v: string | null): string => {
      return v || "";
    })
  ),
  get currentJwtStringValue() {
    return currentJwtStringSubject.value || "";
  },
};

export function redirectToLogin() {
  const frontend_url = encodeURIComponent(window.location.href);
  window.open(
    `${import.meta.env.VITE_BACKEND_URL}/api/login?frontend_url=${frontend_url}`,
    "_self"
  );
}

export function redirectToLogout() {
  window.open(`${import.meta.env.VITE_BACKEND_URL}/api/logout`, "_self");
}

function logout() {
  // remove user from local storage to log user out
  localStorage.removeItem("currentJwtString");
  currentJwtStringSubject.next(null);

  redirectToLogout();
}

export function LoginButton(props: {
  setToken: (a: string) => void;
  decodedToken: DecodedJwt | null;
  isExpired: boolean;
}) {
  useEffect(() => {
    authenticationService.currentJwtString.subscribe(props.setToken);
  }, [props.setToken]);

  if (props.decodedToken && !props.isExpired) {
    return (
      <div>
        <span className="uk-visible@m">
          Logged in as {props.decodedToken.user_claims.name}.
        </span>
        <button
          className="uk-button uk-button-default uk-margin-left"
          onClick={(e) => {
            e.preventDefault();
            authenticationService.logout();
          }}
        >
          <span className="icon" style={{ color: "#fff" }}>
            Logout
          </span>
        </button>
      </div>
    );
  } else {
    return (
      <div>
        <button
          className="uk-button uk-button-default"
          onClick={(e) => {
            e.preventDefault();
            redirectToLogin();
          }}
        >
          <span className="icon" style={{ color: "#fff" }}>
            Login
          </span>
        </button>
      </div>
    );
  }
}
