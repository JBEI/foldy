import React, { useState } from "react";
import { CredentialResponse, GoogleLogin } from "@react-oauth/google";
import axios from "axios";

const YOUR_SERVER_URL = "http://localhost:5000";

const GoogleLoginButton = () => {
  const [jwtToken, setJwtToken] = useState(null);

  const exchangeCodeForToken = async (code: string) => {
    const response = await axios.post(`${YOUR_SERVER_URL}/auth/google`, {
      authorization_code: code,
    });
    const { access_token, id_token, jwt_token } = response.data;
    return jwt_token;
  };

  const onGoogleLoginSuccess = async (response: CredentialResponse) => {
    // TODO: double check that we want "credential" here.
    if (response.credential) {
      const jwtToken = await exchangeCodeForToken(response.credential);
      setJwtToken(jwtToken);
    }
  };

  const onGoogleLoginFailure = () => {
    console.log("Google login failed");
  };

  return (
    <div>
      <GoogleLogin
        // clientId={GOOGLE_CLIENT_ID}
        // buttonText="Sign in with Google"
        onSuccess={onGoogleLoginSuccess}
        onError={onGoogleLoginFailure}
      />
      {jwtToken && (
        <div>
          <h2>JWT Token:</h2>
          <p>{jwtToken}</p>
        </div>
      )}
    </div>
  );
};

export default GoogleLoginButton;
