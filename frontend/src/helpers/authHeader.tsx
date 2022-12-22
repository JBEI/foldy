import { authenticationService } from "../services/authentication.service";

export function authHeader(): Headers {
  // return authorization header with jwt token
  const currentJwtString = authenticationService.currentJwtStringValue;

  var headers = new Headers();
  if (currentJwtString) {
    headers.append("Authorization", `Bearer ${currentJwtString}`);
    return headers;
  } else {
    return headers;
  }
}
